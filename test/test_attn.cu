// Correctness + benchmark harness for single-head attention kernels.
//
//   O = softmax(Q . K^T / sqrt(d)) . V
//
//   Q, K, V, O are [batch][N][d], row-major fp32, contiguous.
//   batch = independent heads stacked; N = sequence length; d = head dim.
//
//   Build:  make build/test_attn   (from the repo root)
//   Run:    ./build/test_attn                 # correctness suite + sweep over N
//           ./build/test_attn --quick         # correctness suite only
//           ./build/test_attn --sweep         # sweep only
//           ./build/test_attn --d=128 --batch=8 --sweep
//           ./build/test_attn 8192            # one perf point at N=8192
//
// -------------------------------------------------------------- adding a kernel
// Write your kernel in its own .cu file (with its own main, like the SGEMM
// files), then:
//
//   1. include it in the KERNEL IMPORT section below with main renamed away
//   2. write a small run() wrapper with this signature:
//
//        void myRun(const float* Q, const float* K, const float* V, float* O,
//                   int batch, int N, int d, float* workspace);
//
//      and, if it needs scratch device memory, a workspace() function:
//
//        size_t myWorkspace(int batch, int N, int d);   // bytes, 0 if none
//
//   3. add a row to g_variants[]
//
// The harness allocates the workspace, poisons O, runs, and checks the result.
// Workspace bytes are what the memory column reports: a naive kernel that
// materialises S declares batch*N*N*4 and will show OOM at large N; a fused
// kernel declares 0. That column is the point of the project.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---------------------------------------------------------------- KERNEL IMPORT
#define main   naive_main_
#include "../attention/naive_attn.cu"
#undef main

#define main   online_main_
#include "../attention/online_attn.cu"          // onlineRowKernel: Br x Bc tiled, fused (FA-style)
#undef main

// ---------------------------------------------------------------------- helpers
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                            \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__, (int)st_); \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static cublasHandle_t g_cublas;

// Error is measured as max |got - ref| / max |ref| over the tensor. O is a
// weighted average of V rows, so individual elements can sit near zero from
// cancellation; a pure relative metric would flag those unfairly.
static const double TOL = 1e-3;

struct Diff { double maxAbs = 0, scale = 0, err = 0; bool finite = true; size_t worst = 0; };

static Diff compareResults(const float* got, const float* ref, size_t n) {
    Diff d;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i])) { d.finite = false; d.err = INFINITY; d.worst = i; return d; }
        double a = std::fabs((double)got[i] - (double)ref[i]);
        if (a > d.maxAbs) { d.maxAbs = a; d.worst = i; }
        double r = std::fabs((double)ref[i]);
        if (r > d.scale) d.scale = r;
    }
    d.err = d.scale > 0 ? d.maxAbs / d.scale : d.maxAbs;
    return d;
}

static const char* fmtBytes(size_t b, char* buf, size_t len) {
    if (b == 0)                snprintf(buf, len, "0");
    else if (b < (1u << 20))   snprintf(buf, len, "%zu KB", b >> 10);
    else if (b < (1u << 30))   snprintf(buf, len, "%.0f MB", b / 1048576.0);
    else                       snprintf(buf, len, "%.1f GB", b / 1073741824.0);
    return buf;
}

// ---------------------------------------------------------- reference: CPU double
// Full reference for small shapes, and a row-range version for spot-checking
// large ones: rows are independent, so 64 rows of a 65536-long sequence cost
// the same as a 64x65536 problem, and the fused kernel is verified at sizes
// where no N x N buffer fits.
static void cpuAttentionRows(const float* Q, const float* K, const float* V, float* O,
                             int N, int d, int row0, int rows) {
    std::vector<double> s(N);
    const double scale = 1.0 / std::sqrt((double)d);
    for (int i = row0; i < row0 + rows; ++i) {
        double m = -INFINITY;
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < d; ++k) acc += (double)Q[(size_t)i * d + k] * (double)K[(size_t)j * d + k];
            s[j] = acc * scale;
            if (s[j] > m) m = s[j];
        }
        double l = 0.0;
        for (int j = 0; j < N; ++j) { s[j] = std::exp(s[j] - m); l += s[j]; }
        for (int k = 0; k < d; ++k) {
            double acc = 0.0;
            for (int j = 0; j < N; ++j) acc += s[j] * (double)V[(size_t)j * d + k];
            O[(size_t)i * d + k] = (float)(acc / l);
        }
    }
}

static void cpuAttention(const float* Q, const float* K, const float* V, float* O,
                         int batch, int N, int d) {
    for (int b = 0; b < batch; ++b) {
        size_t off = (size_t)b * N * d;
        cpuAttentionRows(Q + off, K + off, V + off, O + off, N, d, 0, N);
    }
}

// ------------------------------------------------- reference: cuBLAS + softmax
// The unfused baseline: S = Q K^T (cuBLAS), softmax rows in place, O = P V
// (cuBLAS). Materialises one batch*N*N buffer. This is the oracle for shapes
// too big for the CPU, and the memory baseline every fused kernel is measured
// against. It is validated against the CPU reference in the self-test.

__device__ __forceinline__ float warpMax(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// one block per row; blockDim.x must be a multiple of 32, at most 1024
__global__ void softmaxRowsKernel(float* S, int N) {
    __shared__ float red[32];
    float* row = S + (size_t)blockIdx.x * N;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, nWarps = blockDim.x >> 5;

    float m = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) m = fmaxf(m, row[j]);
    m = warpMax(m);
    if (lane == 0) red[wid] = m;
    __syncthreads();
    m = (tid < nWarps) ? red[tid] : -INFINITY;
    if (wid == 0) m = warpMax(m);
    if (tid == 0) red[0] = m;
    __syncthreads();
    m = red[0];
    __syncthreads();                       // red is reused below

    float l = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float e = __expf(row[j] - m);
        row[j] = e;
        l += e;
    }
    l = warpSum(l);
    if (lane == 0) red[wid] = l;
    __syncthreads();
    l = (tid < nWarps) ? red[tid] : 0.0f;
    if (wid == 0) l = warpSum(l);
    if (tid == 0) red[0] = l;
    __syncthreads();
    const float inv = 1.0f / red[0];

    for (int j = tid; j < N; j += blockDim.x) row[j] *= inv;
}

static size_t cublasNaiveWorkspace(int batch, int N, int d) {
    (void)d;
    return (size_t)batch * N * N * sizeof(float);
}

static void cublasNaiveRun(const float* Q, const float* K, const float* V, float* O,
                           int batch, int N, int d, float* ws) {
    float* S = ws;
    const float scale = 1.0f / std::sqrt((float)d), zero = 0.0f, one = 1.0f;
    const long long sQKV = (long long)N * d, sS = (long long)N * N;

    // row-major S = scale * Q . K^T   ==  cuBLAS: C(N x N) = op(K)^T . Q
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d,
        &scale, K, d, sQKV, Q, d, sQKV, &zero, S, N, sS, batch));

    softmaxRowsKernel<<<batch * N, 256>>>(S, N);

    // row-major O = P . V   ==  cuBLAS: C(d x N) = V . P
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N,
        &one, V, d, sQKV, S, N, sS, &zero, O, d, sQKV, batch));
}

// ------------------------------------------------------------ fused launchers
// One block per BR query rows, 128 threads; the kernel knows one head, so the
// batch is a loop over pointer offsets. No workspace: S never leaves the chip.
static void onlineRun(const float* Q, const float* K, const float* V, float* O,
                      int batch, int N, int d, float* ws) {
    (void)ws;
    const int blocks = (N + BR - 1) / BR;
    for (int b = 0; b < batch; ++b) {
        const size_t off = (size_t)b * N * d;
        onlineRowKernel<<<blocks, 128>>>(Q + off, K + off, V + off, O + off, N);
    }
}

// ------------------------------------------------------------------- variants
typedef size_t (*WorkspaceFn)(int batch, int N, int d);
typedef void   (*RunFn)(const float* Q, const float* K, const float* V, float* O,
                        int batch, int N, int d, float* workspace);

struct AttnVariant {
    const char* name;
    WorkspaceFn workspace;   // scratch device bytes; may be nullptr for 0
    RunFn       run;
    int         alignN;      // N must be a multiple of this (1 = any)
    int         dReq;        // required d (0 = any)
};

static AttnVariant g_variants[] = {
    { "cublas-naive",  cublasNaiveWorkspace, cublasNaiveRun,  1, 0 },   // baseline, row 0
    { "naive",         naiveWorkspace,       naiveRun,       16, 0 },   // your GEMM + your softmax
    { "online",        nullptr,              onlineRun,       1, 64 },   // tiled fused, Br=64 Bc=32, any N
};
static const int NUM_VARIANTS = (int)(sizeof(g_variants) / sizeof(g_variants[0]));

static bool shapeOk(const AttnVariant& v, int N, int d) {
    return (N % v.alignN == 0) && (v.dReq == 0 || d == v.dReq);
}
static size_t wsBytes(const AttnVariant& v, int batch, int N, int d) {
    return v.workspace ? v.workspace(batch, N, d) : 0;
}

// -------------------------------------------------------------------- fixture
struct Fixture {
    int batch = 0, N = 0, d = 0;
    size_t n = 0;                                  // elements per tensor
    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    std::vector<float> hQ, hK, hV;

    bool setup(int b, int n_, int d_) {
        batch = b; N = n_; d = d_; n = (size_t)b * N * d;
        hQ.resize(n); hK.resize(n); hV.resize(n);
        std::mt19937 gen(7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (float& x : hQ) x = dist(gen);
        for (float& x : hK) x = dist(gen);
        for (float& x : hV) x = dist(gen);
        if (cudaMalloc(&dQ, n * 4) != cudaSuccess || cudaMalloc(&dK, n * 4) != cudaSuccess ||
            cudaMalloc(&dV, n * 4) != cudaSuccess || cudaMalloc(&dO, n * 4) != cudaSuccess) {
            cudaGetLastError();
            teardown();
            return false;
        }
        CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), n * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, hK.data(), n * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, hV.data(), n * 4, cudaMemcpyHostToDevice));
        return true;
    }
    void teardown() {
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
        dQ = dK = dV = dO = nullptr;
    }
};

// Allocate a variant's workspace; false (not fatal) if it doesn't fit in VRAM.
// On WSL/WDDM a cudaMalloc larger than the device can succeed by paging to host
// memory over PCIe (a 16 GB request on this 8 GB card ran ~300x slower rather
// than failing), so "fits" is decided against free device memory, not against
// whether cudaMalloc returns an error.
static bool allocWorkspace(size_t bytes, float** ws) {
    *ws = nullptr;
    if (bytes == 0) return true;
    size_t freeB = 0, totalB = 0;
    if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess && bytes > freeB) return false;
    cudaError_t e = cudaMalloc(ws, bytes);
    if (e != cudaSuccess) { cudaGetLastError(); *ws = nullptr; return false; }
    return true;
}

// Run once into fx.dO (poisoned first). err set on launch failure.
static bool runVariant(const AttnVariant& v, Fixture& fx, float* ws, const char** err) {
    *err = nullptr;
    CUDA_CHECK(cudaMemset(fx.dO, 0xAB, fx.n * 4));
    v.run(fx.dQ, fx.dK, fx.dV, fx.dO, fx.batch, fx.N, fx.d, ws);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { *err = cudaGetErrorString(e); return false; }
    return true;
}

// ------------------------------------------------------------------ self-tests
static bool selfTestComparator() {
    std::vector<float> a(1000, 0.5f), b(1000, 0.5f);
    if (compareResults(a.data(), b.data(), 1000).err > TOL) return false;
    a[500] = 0.6f;
    if (compareResults(a.data(), b.data(), 1000).err <= TOL) return false;
    a[500] = NAN;
    if (compareResults(a.data(), b.data(), 1000).finite) return false;
    return true;
}

static bool selfTestReference() {
    Fixture fx;
    if (!fx.setup(2, 256, 64)) return false;
    std::vector<float> ref(fx.n), got(fx.n);
    cpuAttention(fx.hQ.data(), fx.hK.data(), fx.hV.data(), ref.data(), fx.batch, fx.N, fx.d);
    float* ws; const char* err;
    allocWorkspace(wsBytes(g_variants[0], fx.batch, fx.N, fx.d), &ws);
    bool ok = runVariant(g_variants[0], fx, ws, &err);
    if (ok) CUDA_CHECK(cudaMemcpy(got.data(), fx.dO, fx.n * 4, cudaMemcpyDeviceToHost));
    Diff df = ok ? compareResults(got.data(), ref.data(), fx.n) : Diff();
    printf("  cublas-naive vs CPU double (2x256x64): max err %.2e  %s\n",
           ok ? df.err : INFINITY, (ok && df.err <= TOL) ? "ok" : "MISMATCH");
    cudaFree(ws); fx.teardown();
    return ok && df.err <= TOL;
}

// ------------------------------------------------------------ correctness suite
static int correctnessSuite() {
    struct Shape { int batch, N, d; const char* note; };
    const Shape shapes[] = {
        { 1,   64,  64, "tiny"                          },
        { 1,  128,  64, "one tile"                      },
        { 1,  256,  32, "d = 32"                        },
        { 2,  256,  64, "batch 2"                       },
        { 1,  512, 128, "d = 128"                       },
        { 1, 1000,  64, "N not a multiple of 64"        },
        { 3,  333,  64, "ragged N, batch 3"             },
        { 1, 2048,  64, "largest full CPU reference"    },
        { 1, 4096,  64, "reference = cublas-naive"      },
        { 2, 8192,  64, "reference = cublas-naive"      },
    };
    const int nShapes = (int)(sizeof(shapes) / sizeof(shapes[0]));
    const double CPU_BUDGET = 3.0e8;   // batch*N*N*d above this -> GPU reference

    printf("\n=== correctness (max |got-ref| / max |ref|, tolerance %.0e) ===\n\n", TOL);
    printf("%-20s %-8s", "batch x N x d", "ref");
    for (int v = 0; v < NUM_VARIANTS; ++v) printf("%16s", g_variants[v].name);
    printf("\n%-20s %-8s", "", "");
    for (int v = 0; v < NUM_VARIANTS; ++v) printf("%16s", "--------------");
    printf("\n");

    int failures = 0;
    for (int s = 0; s < nShapes; ++s) {
        Fixture fx;
        if (!fx.setup(shapes[s].batch, shapes[s].N, shapes[s].d)) {
            printf("%-20s  (could not allocate)\n", "");
            continue;
        }
        char label[48];
        snprintf(label, sizeof(label), "%d x %d x %d", fx.batch, fx.N, fx.d);

        // reference
        std::vector<float> ref(fx.n);
        bool cpuRef = (double)fx.batch * fx.N * fx.N * fx.d <= CPU_BUDGET;
        if (cpuRef) {
            cpuAttention(fx.hQ.data(), fx.hK.data(), fx.hV.data(), ref.data(), fx.batch, fx.N, fx.d);
        } else {
            float* ws; const char* err;
            if (!allocWorkspace(wsBytes(g_variants[0], fx.batch, fx.N, fx.d), &ws) ||
                !runVariant(g_variants[0], fx, ws, &err)) {
                printf("%-20s %-8s  (reference unavailable)\n", label, "-");
                cudaFree(ws); fx.teardown(); continue;
            }
            CUDA_CHECK(cudaMemcpy(ref.data(), fx.dO, fx.n * 4, cudaMemcpyDeviceToHost));
            cudaFree(ws);
        }
        printf("%-20s %-8s", label, cpuRef ? "cpu" : "cublas");

        std::vector<float> got(fx.n);
        for (int v = 0; v < NUM_VARIANTS; ++v) {
            char cell[32];
            if (!cpuRef && v == 0) { printf("%16s", "(is ref)"); continue; }
            if (!shapeOk(g_variants[v], fx.N, fx.d)) { printf("%16s", "skip"); continue; }
            float* ws; const char* err;
            if (!allocWorkspace(wsBytes(g_variants[v], fx.batch, fx.N, fx.d), &ws)) {
                printf("%16s", "> VRAM"); continue;
            }
            if (!runVariant(g_variants[v], fx, ws, &err)) {
                snprintf(cell, sizeof(cell), "LAUNCH ERR"); failures++;
                fprintf(stderr, "\n  [%s @ %s] %s\n", g_variants[v].name, label, err);
            } else {
                CUDA_CHECK(cudaMemcpy(got.data(), fx.dO, fx.n * 4, cudaMemcpyDeviceToHost));
                Diff df = compareResults(got.data(), ref.data(), fx.n);
                if (!df.finite)          { snprintf(cell, sizeof(cell), "NaN/Inf"); failures++; }
                else if (df.err <= TOL)  { snprintf(cell, sizeof(cell), "ok %.0e", df.err); }
                else { snprintf(cell, sizeof(cell), "FAIL %.1e", df.err); failures++; }
            }
            printf("%16s", cell);
            cudaFree(ws);
        }
        printf("   (%s)\n", shapes[s].note);
        fx.teardown();
    }
    return failures;
}

// --------------------------------------------------------------------- timing
// Same method as the SGEMM harness: settle the clock governor under sustained
// load, then interleaved rounds with the median reported and the spread shown.
// Short runs are batched so each timed sample is ~TARGET_SAMPLE_MS.
static const int   PERF_ROUNDS      = 7;
static const float SETTLE_MS        = 2500.0f;
static const float TARGET_SAMPLE_MS = 20.0f;
static const int   SPOT_ROWS        = 64;      // CPU-checked rows per sweep point

static float timeVariant(const AttnVariant& v, Fixture& fx, float* ws,
                         cudaEvent_t e0, cudaEvent_t e1, int iters) {
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) v.run(fx.dQ, fx.dK, fx.dV, fx.dO, fx.batch, fx.N, fx.d, ws);
    CUDA_CHECK(cudaEventRecord(e1));
    if (cudaEventSynchronize(e1) != cudaSuccess) { cudaGetLastError(); return -1.0f; }
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return ms / iters;
}

static int itersFor(float ms) {
    if (ms <= 0.0f) return 1;
    int it = (int)(TARGET_SAMPLE_MS / ms + 0.5f);
    return it < 1 ? 1 : (it > 200 ? 200 : it);
}

struct PerfRow { float ms = -1; float spread = 0; size_t ws = 0; const char* status = ""; double err = -1; bool alone = false; };

static std::vector<PerfRow> perfPoint(int batch, int N, int d, bool print) {
    std::vector<PerfRow> rows(NUM_VARIANTS);
    Fixture fx;
    if (!fx.setup(batch, N, d)) {
        for (auto& r : rows) r.status = "OOM (Q/K/V/O)";
        return rows;
    }
    const double flop = 4.0 * batch * (double)N * N * d;   // 2 N^2 d for QK^T, 2 N^2 d for PV

    // spot-check reference: first SPOT_ROWS rows of batch 0, CPU double
    const int spot = std::min(SPOT_ROWS, N);
    std::vector<float> spotRef((size_t)spot * d), spotGot((size_t)spot * d);
    cpuAttentionRows(fx.hQ.data(), fx.hK.data(), fx.hV.data(), spotRef.data(), N, d, 0, spot);

    std::vector<float*> ws(NUM_VARIANTS, nullptr);
    std::vector<bool> eligible(NUM_VARIANTS, false), deferred(NUM_VARIANTS, false);
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        rows[v].ws = wsBytes(g_variants[v], batch, N, d);
        if (!shapeOk(g_variants[v], N, d)) { rows[v].status = "skip"; continue; }
        // workspaces of all variants are held at once for interleaved timing;
        // one that doesn't fit alongside the others is retried by itself below
        if (!allocWorkspace(rows[v].ws, &ws[v])) { rows[v].status = "> VRAM"; deferred[v] = true; continue; }
        const char* err;
        if (!runVariant(g_variants[v], fx, ws[v], &err)) {
            rows[v].status = "LAUNCH ERR";
            fprintf(stderr, "  [%s @ N=%d] %s\n", g_variants[v].name, N, err);
            cudaFree(ws[v]); ws[v] = nullptr; continue;
        }
        CUDA_CHECK(cudaMemcpy(spotGot.data(), fx.dO, (size_t)spot * d * 4, cudaMemcpyDeviceToHost));
        Diff df = compareResults(spotGot.data(), spotRef.data(), (size_t)spot * d);
        rows[v].err = df.err;
        rows[v].status = (df.finite && df.err <= TOL) ? "ok" : "FAIL";
        eligible[v] = true;
    }

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    // batch sizes from one warm-up launch each, then settle the clock
    std::vector<int> iters(NUM_VARIANTS, 1);
    int settleWith = -1;
    for (int v = 0; v < NUM_VARIANTS; ++v)
        if (eligible[v]) { iters[v] = itersFor(timeVariant(g_variants[v], fx, ws[v], e0, e1, 1)); if (settleWith < 0) settleWith = v; }
    if (settleWith >= 0)
        for (float acc = 0; acc < SETTLE_MS;) {
            float t = timeVariant(g_variants[settleWith], fx, ws[settleWith], e0, e1, iters[settleWith]);
            if (t < 0) break;
            acc += t * iters[settleWith];
        }

    std::vector<std::vector<float>> samples(NUM_VARIANTS);
    for (int r = 0; r < PERF_ROUNDS; ++r)
        for (int v = 0; v < NUM_VARIANTS; ++v) {
            if (!eligible[v]) continue;
            float t = timeVariant(g_variants[v], fx, ws[v], e0, e1, iters[v]);
            if (t < 0) { eligible[v] = false; rows[v].status = "LAUNCH ERR"; continue; }
            samples[v].push_back(t);
        }
    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));

    for (int v = 0; v < NUM_VARIANTS; ++v) {
        if (!eligible[v] || samples[v].empty()) continue;
        std::vector<float> s(samples[v]); std::sort(s.begin(), s.end());
        rows[v].ms = s[s.size() / 2];
        rows[v].spread = 100.0f * (s.back() - s.front()) / rows[v].ms;
    }
    for (int v = 0; v < NUM_VARIANTS; ++v) { cudaFree(ws[v]); ws[v] = nullptr; }

    // second pass: variants that didn't fit alongside the others, one at a time
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        if (!deferred[v]) continue;
        float* w = nullptr;
        if (!allocWorkspace(rows[v].ws, &w)) continue;            // genuinely > VRAM
        const char* err;
        if (!runVariant(g_variants[v], fx, w, &err)) {
            rows[v].status = "LAUNCH ERR";
            fprintf(stderr, "  [%s @ N=%d] %s\n", g_variants[v].name, N, err);
            cudaFree(w); continue;
        }
        CUDA_CHECK(cudaMemcpy(spotGot.data(), fx.dO, (size_t)spot * d * 4, cudaMemcpyDeviceToHost));
        Diff df = compareResults(spotGot.data(), spotRef.data(), (size_t)spot * d);
        rows[v].err = df.err;
        rows[v].status = (df.finite && df.err <= TOL) ? "ok" : "FAIL";
        rows[v].alone = true;
        cudaEvent_t a0, a1; CUDA_CHECK(cudaEventCreate(&a0)); CUDA_CHECK(cudaEventCreate(&a1));
        int it = itersFor(timeVariant(g_variants[v], fx, w, a0, a1, 1));
        for (float acc = 0; acc < SETTLE_MS;) { float t = timeVariant(g_variants[v], fx, w, a0, a1, it); if (t < 0) break; acc += t * it; }
        std::vector<float> smp;
        for (int r = 0; r < PERF_ROUNDS; ++r) { float t = timeVariant(g_variants[v], fx, w, a0, a1, it); if (t < 0) break; smp.push_back(t); }
        CUDA_CHECK(cudaEventDestroy(a0)); CUDA_CHECK(cudaEventDestroy(a1));
        if (!smp.empty()) {
            std::sort(smp.begin(), smp.end());
            rows[v].ms = smp[smp.size() / 2];
            rows[v].spread = 100.0f * (smp.back() - smp.front()) / rows[v].ms;
        }
        cudaFree(w);
    }
    fx.teardown();

    if (print) {
        char b1[32];
        printf("\n=== performance  batch=%d  N=%d  d=%d  (median of %d interleaved rounds) ===\n\n", batch, N, d, PERF_ROUNDS);
        printf("%-16s %12s %12s %10s %8s %12s\n", "variant", "time (ms)", "TFLOP/s", "spread", "check", "workspace");
        printf("%-16s %12s %12s %10s %8s %12s\n", "----------------", "------------", "------------", "----------", "--------", "------------");
        for (int v = 0; v < NUM_VARIANTS; ++v) {
            const PerfRow& r = rows[v];
            fmtBytes(r.ws, b1, sizeof(b1));
            if (r.ms < 0) printf("%-16s %12s %12s %10s %8s %12s\n", g_variants[v].name, "-", "-", "-", r.status, b1);
            else printf("%-16s %12.3f %12.2f %9.1f%% %8s %12s\n", g_variants[v].name, r.ms,
                        flop / (r.ms / 1e3) / 1e12, r.spread, r.status, b1);
        }
    }
    return rows;
}

// ---------------------------------------------------------------------- sweep
static void sweepSuite(int batch, int d) {
    const int Ns[] = { 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
    const int nN = (int)(sizeof(Ns) / sizeof(Ns[0]));

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    char b1[32], b2[32];
    printf("\n=== sweep over N   batch=%d  d=%d   (device: %s free of %s) ===\n",
           batch, d, fmtBytes(freeB, b1, sizeof(b1)), fmtBytes(totalB, b2, sizeof(b2)));
    printf("    time = median of %d interleaved rounds (ms); check = first %d rows vs CPU double;\n"
           "    workspace = scratch memory the variant declares beyond Q/K/V/O\n\n", PERF_ROUNDS, SPOT_ROWS);

    printf("%8s", "N");
    for (int v = 0; v < NUM_VARIANTS; ++v) printf("  |  %-34s", g_variants[v].name);
    printf("\n%8s", "");
    for (int v = 0; v < NUM_VARIANTS; ++v) printf("  |  %8s %8s %6s %10s", "ms", "TFLOP/s", "check", "workspace");
    printf("\n%8s", "--------");
    for (int v = 0; v < NUM_VARIANTS; ++v) printf("  |  %-34s", "----------------------------------");
    printf("\n");

    for (int i = 0; i < nN; ++i) {
        std::vector<PerfRow> rows = perfPoint(batch, Ns[i], d, false);
        const double flop = 4.0 * batch * (double)Ns[i] * Ns[i] * d;
        printf("%8d", Ns[i]);
        for (int v = 0; v < NUM_VARIANTS; ++v) {
            const PerfRow& r = rows[v];
            fmtBytes(r.ws, b1, sizeof(b1));
            if (r.ms < 0) printf("  |  %8s %8s %6s %10s", "-", "-", r.status, b1);
            else printf("  |  %8.2f %8.2f %5s%c %10s", r.ms, flop / (r.ms / 1e3) / 1e12, r.status, r.alone ? '*' : ' ', b1);
        }
        printf("\n");
        fflush(stdout);
    }
    printf("\n  * timed alone: its workspace did not fit next to the other variants', so it\n"
           "    was not interleaved with them at that N.\n");
}

// ----------------------------------------------------------------------- main
int main(int argc, char** argv) {
    bool quick = false, sweep = false;
    int d = 64, batch = 1, singleN = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--quick")) quick = true;
        else if (!strcmp(argv[i], "--sweep")) sweep = true;
        else if (!strncmp(argv[i], "--d=", 4)) d = atoi(argv[i] + 4);
        else if (!strncmp(argv[i], "--batch=", 8)) batch = atoi(argv[i] + 8);
        else if (argv[i][0] != '-') singleN = atoi(argv[i]);
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs, CC %d.%d)\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    CUBLAS_CHECK(cublasCreate(&g_cublas));

    printf("\n=== harness self-test ===\n\n");
    bool cmpOk = selfTestComparator();
    printf("  comparator detects corruption:    %s\n", cmpOk ? "ok" : "BROKEN");
    bool refOk = selfTestReference();
    if (!cmpOk || !refOk) {
        fprintf(stderr, "\nHarness self-test failed; results below cannot be trusted.\n");
        return 2;
    }

    if (singleN > 0) { perfPoint(batch, singleN, d, true); cublasDestroy(g_cublas); return 0; }
    if (sweep)       { sweepSuite(batch, d);              cublasDestroy(g_cublas); return 0; }

    int failures = correctnessSuite();
    if (!quick) sweepSuite(batch, d);

    printf("\n%s\n", failures == 0 ? "All correctness cases passed."
                                   : "Some correctness cases failed (see table).");
    cublasDestroy(g_cublas);
    return failures == 0 ? 0 : 1;
}
