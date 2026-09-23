// Unit test + benchmark harness for the SGEMM ladder.
//
//   Build:  make build/test_sgemm   (from the repo root; binaries go to build/)
//   Run:    ./build/test_sgemm              # correctness suite + perf at 4096^3
//           ./build/test_sgemm 2048         # perf at 2048^3
//           ./build/test_sgemm 1024 512 256 # perf at N=1024 K=512 M=256
//           ./build/test_sgemm --quick      # correctness suite only, no big perf run
//           ./build/test_sgemm --sweep      # perf at several shapes + summary table
//
// The kernel files each have their own main() and matMul(). Rather than
// edit them, they are #included here with those two symbols renamed away, so
// each file still compiles and runs standalone exactly as before.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---------------------------------------------------------------- kernel import
#define main   naive_main_
#define matMul naive_matMul_
#include "../ladder/naive.cu"
#undef main
#undef matMul

#define main   sharedmem_main_
#define matMul sharedmem_matMul_
#include "../ladder/sharedmemtile.cu"
#undef main
#undef matMul
#undef TILE_SIZE

#define main   threadtile_main_
#define matMul threadtile_matMul_
#include "../ladder/threadtiling.cu"
#undef main
#undef matMul

#define main   sharedthread_main_
#define matMul sharedthread_matMul_
#include "../ladder/sharedthreadtile.cu"
#undef main
#undef matMul
#undef TILE_SIZE

#define main   sharedthreadv2_main_
#define matMul sharedthreadv2_matMul_
#include "../ladder/sharedthreadtilev2.cu"
// capture the tile size before the macros go out of scope, so the launcher
// below stays in sync if you change BM in that file
static constexpr int V2_BM = BM;
#undef main
#undef matMul
#undef BM
#undef BK
#undef TN

#define main   transpose_main_
#define matMul transpose_matMul_
#include "../ladder/transpose.cu"
static constexpr int TR_BM = BM;
#undef main
#undef matMul
#undef BM
#undef BK
#undef TN

#define main   doublebuf_main_
#define matMul doublebuf_matMul_
#include "../ladder/doublebuffer.cu"
static constexpr int DB_BM = BM;
#undef main
#undef matMul
#undef BM
#undef BK
#undef TN

#define main   bk16_main_
#define matMul bk16_matMul_
#include "../ladder/bk16.cu"
static constexpr int B16_BM = BM;
#undef main
#undef matMul
#undef BM
#undef BK
#undef TN
#undef NTHREADS
#undef STRIDE_A
#undef STRIDE_B
#undef PASSES_A
#undef PASSES_B

#define main   warptile_main_
#define matMul warptile_matMul_
#include "../ladder/warptiling.cu"
static constexpr int WT_BM = BM;
#undef main
#undef matMul
#undef BM
#undef BK
#undef TN
#undef NTHREADS
#undef STRIDE_A
#undef STRIDE_B
#undef PASSES_A
#undef PASSES_B

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

// A kernel that reads out of bounds would abort the whole suite with an
// illegal-access error, so the input buffers carry a zero-filled margin. This
// stops a crash from masking the results of every later test; it does NOT hide
// a wrong answer, which the comparison below still catches.
static const size_t PAD_ELEMS = 1 << 16;

static const double REL_TOL = 1e-3;   // relative error accepted vs the reference

struct Diff {
    double maxAbs = 0.0;
    double maxRel = 0.0;
    size_t worstIdx = 0;
    bool   finite  = true;
};

static Diff compareResults(const std::vector<float>& got, const std::vector<float>& ref) {
    Diff d;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) { d.finite = false; d.worstIdx = i; d.maxRel = INFINITY; return d; }
        double a = std::fabs((double)got[i] - (double)ref[i]);
        double r = a / (std::fabs((double)ref[i]) + 1e-6);
        if (a > d.maxAbs) d.maxAbs = a;
        if (r > d.maxRel) { d.maxRel = r; d.worstIdx = i; }
    }
    return d;
}

// --------------------------------------------------------------- kernel launchers
// Each of these reproduces the grid/block configuration from that kernel's own
// matMul() wrapper, and passes the arguments in the same order that wrapper used.
typedef void (*LaunchFn)(const float*, const float*, float*, int, int, int);

static void launchNaive(const float* dA, const float* dB, float* dC, int N, int K, int M) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    naiveMatMul<<<grid, block>>>((float*)dA, (float*)dB, dC, N, K, M);
}

static void launchSharedMem(const float* dA, const float* dB, float* dC, int N, int K, int M) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    sharedTileMatMul<<<grid, block>>>((float*)dA, (float*)dB, dC, N, K, M);
}

static void launchThreadTile(const float* dA, const float* dB, float* dC, int N, int K, int M) {
    dim3 block(16, 16);
    dim3 grid((M + block.x * 4 - 1) / (block.x * 4), (N + block.y * 4 - 1) / (block.y * 4));
    threadTileMatmul<<<grid, block>>>((float*)dA, (float*)dB, dC, N, K, M);
}

static void launchSharedThreadTile(const float* dA, const float* dB, float* dC, int N, int K, int M) {
    dim3 block(16, 16);
    dim3 grid((M + block.x * 4 - 1) / (block.x * 4), (N + block.y * 4 - 1) / (block.y * 4));
    sharedThreadTile<<<grid, block>>>((float*)dA, (float*)dB, dC, N, K, M);
}

// 1D block of 256 threads, 128x128 block tile (see that file's matMul wrapper)
static void launchSharedThreadTileV2(const float* dA, const float* dB, float* dC, int N, int K, int M) {
    dim3 block(256);
    dim3 grid((M + V2_BM - 1) / V2_BM, (N + V2_BM - 1) / V2_BM);
    sharedThreadTilev2<<<grid, block>>>((float*)dA, (float*)dB, dC, N, K, M);
}

// float4 loads: same launch shape as v2
static void launchTranspose(const float* dA, const float* dB, float* dC, int N, int K, int M) {
    dim3 block(256);
    dim3 grid((M + TR_BM - 1) / TR_BM, (N + TR_BM - 1) / TR_BM);
    vectorizeTranspose<<<grid, block>>>((float*)dA, (float*)dB, dC, N, K, M);
}

// double-buffered shared memory, same launch shape
static void launchDoubleBuffer(const float* dA, const float* dB, float* dC, int N, int K, int M) {
    dim3 block(256);
    dim3 grid((M + DB_BM - 1) / DB_BM, (N + DB_BM - 1) / DB_BM);
    doubleBuffer<<<grid, block>>>((float*)dA, (float*)dB, dC, N, K, M);
}

// double buffer with BK=16, same launch shape
static void launchBk16(const float* dA, const float* dB, float* dC, int N, int K, int M) {
    dim3 block(256);
    dim3 grid((M + B16_BM - 1) / B16_BM, (N + B16_BM - 1) / B16_BM);
    bk16<<<grid, block>>>((float*)dA, (float*)dB, dC, N, K, M);
}

// bk16 + warp-level remap of the compute mapping, same launch shape
static void launchWarpTile(const float* dA, const float* dB, float* dC, int N, int K, int M) {
    dim3 block(256);
    dim3 grid((M + WT_BM - 1) / WT_BM, (N + WT_BM - 1) / WT_BM);
    warpTile<<<grid, block>>>((float*)dA, (float*)dB, dC, N, K, M);
}

struct Kernel {
    const char* name;
    LaunchFn    launch;
    int         align;   // K and M must be multiples of this (float4 kernels need 4)
};

static Kernel g_kernels[] = {
    { "naive",          launchNaive,              1 },
    { "sharedmem",      launchSharedMem,          1 },
    { "threadtile",     launchThreadTile,         1 },
    { "sharedthread",   launchSharedThreadTile,   1 },
    { "sharedthreadv2", launchSharedThreadTileV2, 1 },
    { "transpose",      launchTranspose,          4 },
    { "doublebuffer",   launchDoubleBuffer,       4 },
    { "bk16",           launchBk16,               4 },
    { "warptile",       launchWarpTile,           4 },
};

// A misaligned float4 access is a sticky device error that would poison every
// later test, so shapes that violate a kernel's alignment precondition are
// skipped and reported rather than run.
static bool shapeOk(const Kernel& k, int K, int M) {
    return (K % k.align == 0) && (M % k.align == 0);
}
static const int NUM_KERNELS = (int)(sizeof(g_kernels) / sizeof(g_kernels[0]));

// ------------------------------------------------------------------- references
// row-major C(N x M) = A(N x K) * B(K x M)
// column-major equivalent: C^T = B^T * A^T
static void cublasReference(cublasHandle_t h, const float* dA, const float* dB,
                            float* dC, int N, int K, int M) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                &alpha, dB, M, dA, K, &beta, dC, M);
}

static void cpuReference(const float* A, const float* B, float* C, int N, int K, int M) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)A[i * K + k] * (double)B[k * M + j];
            C[i * M + j] = (float)acc;
        }
}

// ------------------------------------------------------------------- test fixture
struct Fixture {
    int N, K, M;
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dRef = nullptr;
    std::vector<float> hA, hB, ref;

    void setup(int n, int k, int m, cublasHandle_t h) {
        N = n; K = k; M = m;
        hA.resize((size_t)N * K);
        hB.resize((size_t)K * M);
        ref.resize((size_t)N * M);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (float& x : hA) x = dist(gen);
        for (float& x : hB) x = dist(gen);

        CUDA_CHECK(cudaMalloc(&dA, (hA.size() + PAD_ELEMS) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, (hB.size() + PAD_ELEMS) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, ref.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dRef, ref.size() * sizeof(float)));
        CUDA_CHECK(cudaMemset(dA, 0, (hA.size() + PAD_ELEMS) * sizeof(float)));
        CUDA_CHECK(cudaMemset(dB, 0, (hB.size() + PAD_ELEMS) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

        cublasReference(h, dA, dB, dRef, N, K, M);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(ref.data(), dRef, ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void teardown() {
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
        dA = dB = dC = dRef = nullptr;
    }
};

// Runs one kernel and returns its output. err is set if the launch failed.
static bool runKernel(const Kernel& kern, Fixture& fx, std::vector<float>& out,
                      const char** err) {
    *err = nullptr;
    CUDA_CHECK(cudaMemset(fx.dC, 0xAB, fx.ref.size() * sizeof(float)));  // poison
    kern.launch(fx.dA, fx.dB, fx.dC, fx.N, fx.K, fx.M);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { *err = cudaGetErrorString(e); return false; }
    out.resize(fx.ref.size());
    CUDA_CHECK(cudaMemcpy(out.data(), fx.dC, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return true;
}

static float timeKernel(const Kernel& kern, Fixture& fx, int iters) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int i = 0; i < 3; ++i) kern.launch(fx.dA, fx.dB, fx.dC, fx.N, fx.K, fx.M);
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0f;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) kern.launch(fx.dA, fx.dB, fx.dC, fx.N, fx.K, fx.M);
    CUDA_CHECK(cudaEventRecord(t1));
    if (cudaEventSynchronize(t1) != cudaSuccess) return -1.0f;
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / iters;
}

static float timeCublas(cublasHandle_t h, Fixture& fx, int iters) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int i = 0; i < 3; ++i) cublasReference(h, fx.dA, fx.dB, fx.dC, fx.N, fx.K, fx.M);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) cublasReference(h, fx.dA, fx.dB, fx.dC, fx.N, fx.K, fx.M);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / iters;
}

// -------------------------------------------------------------------- test suite
// Confirms the comparator can actually fail. A validator that always passes is
// worse than none, so prove it detects a single corrupted element.
static bool selfTestComparator() {
    std::vector<float> a(1000, 1.0f), b(1000, 1.0f);
    if (compareResults(a, b).maxRel > REL_TOL) return false;   // identical must pass
    a[500] = 2.0f;
    if (compareResults(a, b).maxRel <= REL_TOL) return false;  // corrupted must fail
    std::vector<float> c(10, 1.0f); c[3] = NAN;
    if (compareResults(c, b).finite) return false;             // NaN must be caught
    return true;
}

// Confirms cuBLAS agrees with a plain double-accumulate CPU matmul, so the
// reference the kernels are judged against is itself trustworthy.
static bool selfTestReference(cublasHandle_t h) {
    Fixture fx;
    fx.setup(96, 128, 64, h);
    std::vector<float> cpu((size_t)fx.N * fx.M);
    cpuReference(fx.hA.data(), fx.hB.data(), cpu.data(), fx.N, fx.K, fx.M);
    Diff d = compareResults(fx.ref, cpu);
    fx.teardown();
    printf("  cuBLAS vs CPU double-accumulate: max rel %.2e  %s\n",
           d.maxRel, d.maxRel <= REL_TOL ? "ok" : "MISMATCH");
    return d.maxRel <= REL_TOL;
}

static int correctnessSuite(cublasHandle_t h) {
    struct Shape { int N, K, M; const char* note; };
    const Shape shapes[] = {
        {  64,  64,  64, "small square"                 },
        { 128, 128, 128, "multiple of both tile sizes"  },
        { 256, 256, 256, "square"                       },
        { 512, 512, 512, "square"                       },
        { 129, 129, 129, "not a multiple of 16 or 64"   },
        { 200, 300, 400, "rectangular, K != N != M"     },
        {  48,  80,  96, "smaller than a 64 tile"       },
        {1024,1024,1024, "square"                       },
    };
    const int nShapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    printf("\n=== correctness (vs cuBLAS, relative tolerance %.0e) ===\n\n", REL_TOL);
    printf("%-22s", "shape  N x K x M");
    for (int k = 0; k < NUM_KERNELS; ++k) printf("%17s", g_kernels[k].name);
    printf("\n");
    printf("%-22s", "");
    for (int k = 0; k < NUM_KERNELS; ++k) printf("%17s", "---------------");
    printf("\n");

    int failures = 0;
    for (int s = 0; s < nShapes; ++s) {
        Fixture fx;
        fx.setup(shapes[s].N, shapes[s].K, shapes[s].M, h);
        char label[64];
        snprintf(label, sizeof(label), "%4d x%4d x%4d", shapes[s].N, shapes[s].K, shapes[s].M);
        printf("%-22s", label);

        std::vector<float> out;
        for (int k = 0; k < NUM_KERNELS; ++k) {
            const char* err = nullptr;
            char cell[32];
            if (!shapeOk(g_kernels[k], fx.K, fx.M)) {
                snprintf(cell, sizeof(cell), "skip (K,M %% %d)", g_kernels[k].align);
                printf("%17s", cell);
                continue;
            }
            if (!runKernel(g_kernels[k], fx, out, &err)) {
                snprintf(cell, sizeof(cell), "LAUNCH ERR");
                failures++;
                printf("%17s", cell);
                fprintf(stderr, "\n  [%s @ %s] %s\n", g_kernels[k].name, label, err);
                continue;
            }
            Diff d = compareResults(out, fx.ref);
            if (!d.finite)                    { snprintf(cell, sizeof(cell), "NaN/Inf"); failures++; }
            else if (d.maxRel <= REL_TOL)     { snprintf(cell, sizeof(cell), "ok"); }
            else { snprintf(cell, sizeof(cell), "FAIL %.1e", d.maxRel); failures++; }
            printf("%17s", cell);
        }
        printf("   (%s)\n", shapes[s].note);
        fx.teardown();
    }
    return failures;
}

// Timing method: every kernel (and cuBLAS) is launched once per round, in
// round-robin order, for ROUNDS rounds, and the MEDIAN per kernel is reported.
// On a laptop GPU the clock moves between launches; timing each kernel in one
// contiguous burst lets a fast or slow moment land on one kernel and not its
// neighbour, which can hide or invent a 10% difference. Interleaving spreads
// the noise evenly, and the median ignores the outliers.
static const int PERF_ROUNDS = 9;
static const float SETTLE_MS = 2500.0f;   // sustained load before timing

// One timed sample = `iters` back-to-back launches between two events, returned
// as ms per launch. Short kernels (a few ms) are batched so every sample lasts
// ~TARGET_SAMPLE_MS; a single 1-2 ms launch is otherwise at the mercy of one
// clock transition, which is what blew the spreads up at 2048^3.
static const float TARGET_SAMPLE_MS = 20.0f;

static float launchAndTime(const Kernel* kern, cublasHandle_t h, Fixture& fx,
                           cudaEvent_t e0, cudaEvent_t e1, int iters = 1) {
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) {
        if (kern) kern->launch(fx.dA, fx.dB, fx.dC, fx.N, fx.K, fx.M);
        else      cublasReference(h, fx.dA, fx.dB, fx.dC, fx.N, fx.K, fx.M);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    if (cudaEventSynchronize(e1) != cudaSuccess) return -1.0f;
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return ms / iters;
}

static int itersFor(float warmMs) {
    if (warmMs <= 0.0f) return 1;
    int it = (int)(TARGET_SAMPLE_MS / warmMs + 0.5f);
    return it < 1 ? 1 : (it > 200 ? 200 : it);
}

static std::vector<double> perfSuite(cublasHandle_t h, int N, int K, int M) {
    printf("\n=== performance  (N=%d  K=%d  M=%d;  median of %d interleaved rounds, ~%.0f ms per sample) ===\n\n",
           N, K, M, PERF_ROUNDS, TARGET_SAMPLE_MS);
    Fixture fx;
    fx.setup(N, K, M, h);
    const double flop = 2.0 * N * (double)K * M;

    // correctness verdict + eligibility, once per kernel
    std::vector<const char*> verdict(NUM_KERNELS, "skip");
    std::vector<bool> eligible(NUM_KERNELS, false);
    std::vector<float> out;
    for (int k = 0; k < NUM_KERNELS; ++k) {
        if (!shapeOk(g_kernels[k], fx.K, fx.M)) continue;
        const char* err = nullptr;
        if (!runKernel(g_kernels[k], fx, out, &err)) {
            verdict[k] = "LAUNCH ERR";
            fprintf(stderr, "  [%s] %s\n", g_kernels[k].name, err);
            continue;
        }
        Diff d = compareResults(out, fx.ref);
        verdict[k] = (!d.finite) ? "NaN/Inf" : (d.maxRel <= REL_TOL ? "ok" : "FAIL");
        eligible[k] = true;
    }

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // warm-up pass: one launch each; its time sets the batch size per kernel
    std::vector<int> iters(NUM_KERNELS + 1, 1);
    for (int k = 0; k < NUM_KERNELS; ++k)
        if (eligible[k]) iters[k] = itersFor(launchAndTime(&g_kernels[k], h, fx, e0, e1));
    iters[NUM_KERNELS] = itersFor(launchAndTime(nullptr, h, fx, e0, e1));

    // clock-settle: on a power/thermally limited GPU the clock ramps up at the
    // start of a run and is pulled back down over the next second or two. Keep
    // the GPU busy until that governor has reached its steady state, so the
    // timed rounds see one clock rather than a moving one. (Locking the clock
    // with `nvidia-smi -lgc` is better still, but needs admin.)
    {
        float settled = 0.0f;
        while (settled < SETTLE_MS) {
            float ms = launchAndTime(nullptr, h, fx, e0, e1);
            if (ms < 0) break;
            settled += ms;
        }
    }

    // interleaved timed rounds
    std::vector<std::vector<float>> samples(NUM_KERNELS + 1);
    for (int r = 0; r < PERF_ROUNDS; ++r) {
        for (int k = 0; k < NUM_KERNELS; ++k) {
            if (!eligible[k]) continue;
            float ms = launchAndTime(&g_kernels[k], h, fx, e0, e1, iters[k]);
            if (ms < 0) { eligible[k] = false; verdict[k] = "LAUNCH ERR"; continue; }
            samples[k].push_back(ms);
        }
        samples[NUM_KERNELS].push_back(launchAndTime(nullptr, h, fx, e0, e1, iters[NUM_KERNELS]));
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    auto median = [](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto spread = [](const std::vector<float>& v) {   // (max - min) / median, as a percent
        float lo = *std::min_element(v.begin(), v.end());
        float hi = *std::max_element(v.begin(), v.end());
        std::vector<float> c(v); std::sort(c.begin(), c.end());
        return 100.0f * (hi - lo) / c[c.size() / 2];
    };

    float msCublas = median(samples[NUM_KERNELS]);
    double gfCublas = flop / (msCublas / 1e3) / 1e9;

    printf("%-22s %12s %14s %12s %9s %10s\n",
           "kernel", "time (ms)", "GFLOP/s", "% of cuBLAS", "spread", "correct");
    printf("%-22s %12s %14s %12s %9s %10s\n",
           "----------------------", "------------", "--------------", "------------", "---------", "----------");
    for (int k = 0; k < NUM_KERNELS; ++k) {
        if (!eligible[k]) {
            printf("%-22s %12s %14s %12s %9s %10s\n", g_kernels[k].name, "-", "-", "-", "-", verdict[k]);
            continue;
        }
        float ms = median(samples[k]);
        double gf = flop / (ms / 1e3) / 1e9;
        printf("%-22s %12.3f %14.2f %11.1f%% %8.1f%% %10s\n",
               g_kernels[k].name, ms, gf, 100.0 * gf / gfCublas, spread(samples[k]), verdict[k]);
    }
    printf("%-22s %12.3f %14.2f %11.1f%% %8.1f%% %10s\n",
           "cuBLAS SGEMM", msCublas, gfCublas, 100.0, spread(samples[NUM_KERNELS]), "ref");
    printf("\n  spread = (max - min) / median across rounds. Large spreads mean the clock\n"
           "  was moving; differences between kernels smaller than the spread are noise.\n");

    std::vector<double> pct(NUM_KERNELS, -1.0);
    for (int k = 0; k < NUM_KERNELS; ++k)
        if (eligible[k]) pct[k] = 100.0 * (flop / (median(samples[k]) / 1e3) / 1e9) / gfCublas;
    fx.teardown();
    return pct;
}

// ---------------------------------------------------------------- shape sweep
// The headline "% of cuBLAS" is shape-specific: 4096^3 has no partial tiles and
// fills 14 waves. These shapes exercise the ways a real workload differs.
struct SweepShape { int N, K, M; const char* note; };
static const SweepShape g_sweep[] = {
    { 4096, 4096, 4096, "baseline, all tiles full"            },
    { 2048, 2048, 2048, "256 blocks: 3.6 waves, big tail"     },
    { 4000, 4000, 4000, "ragged: last tile row/col 25% used"  },
    { 1024, 4096, 1024, "long K, only 64 blocks (< 1 wave)"   },
    { 8192,  512, 8192, "short K, huge C: prologue/epilogue"  },
};
static const int NUM_SWEEP = (int)(sizeof(g_sweep) / sizeof(g_sweep[0]));

static void sweepSuite(cublasHandle_t h) {
    std::vector<std::vector<double>> table;
    for (int i = 0; i < NUM_SWEEP; ++i)
        table.push_back(perfSuite(h, g_sweep[i].N, g_sweep[i].K, g_sweep[i].M));

    printf("\n=== sweep summary: %% of cuBLAS, median of %d interleaved rounds ===\n\n", PERF_ROUNDS);
    printf("%-16s", "kernel");
    for (int i = 0; i < NUM_SWEEP; ++i) {
        char hdr[32];
        snprintf(hdr, sizeof(hdr), "%dx%dx%d", g_sweep[i].N, g_sweep[i].K, g_sweep[i].M);
        printf("%16s", hdr);
    }
    printf("\n%-16s", "");
    for (int i = 0; i < NUM_SWEEP; ++i) printf("%16s", "--------------");
    printf("\n");
    for (int k = 0; k < NUM_KERNELS; ++k) {
        printf("%-16s", g_kernels[k].name);
        for (int i = 0; i < NUM_SWEEP; ++i) {
            if (table[i][k] < 0) printf("%16s", "skip");
            else                 printf("%15.1f%%", table[i][k]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < NUM_SWEEP; ++i)
        printf("  %dx%dx%d  %s\n", g_sweep[i].N, g_sweep[i].K, g_sweep[i].M, g_sweep[i].note);
}

int main(int argc, char** argv) {
    bool quick = false, sweep = false;
    int N = 4096, K = 4096, M = 4096;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--quick")) quick = true;
        if (!strcmp(argv[i], "--sweep")) sweep = true;
    }
    int nums[3], nNums = 0;
    for (int i = 1; i < argc && nNums < 3; ++i)
        if (argv[i][0] != '-') nums[nNums++] = atoi(argv[i]);
    if (nNums == 1) N = K = M = nums[0];
    else if (nNums == 3) { N = nums[0]; K = nums[1]; M = nums[2]; }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs, CC %d.%d)\n", prop.name,
           prop.multiProcessorCount, prop.major, prop.minor);
    printf("Note: input buffers carry a %zu-element zero margin so an out-of-bounds\n"
           "      read reports a wrong answer instead of aborting the whole suite.\n",
           PAD_ELEMS);

    cublasHandle_t h;
    cublasCreate(&h);

    printf("\n=== harness self-test ===\n\n");
    bool cmpOk = selfTestComparator();
    printf("  comparator detects corruption:    %s\n", cmpOk ? "ok" : "BROKEN");
    bool refOk = selfTestReference(h);
    if (!cmpOk || !refOk) {
        fprintf(stderr, "\nHarness self-test failed; kernel results below cannot be trusted.\n");
        cublasDestroy(h);
        return 2;
    }

    if (sweep) {
        sweepSuite(h);
        cublasDestroy(h);
        return 0;
    }

    int failures = correctnessSuite(h);
    if (!quick) perfSuite(h, N, K, M);

    printf("\n%s\n", failures == 0 ? "All correctness cases passed."
                                   : "Some correctness cases failed (see table).");
    cublasDestroy(h);
    return failures == 0 ? 0 : 1;
}
