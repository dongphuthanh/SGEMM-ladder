// Unit test + benchmark harness for the SGEMM ladder.
//
//   Build:  make build/test_sgemm   (from the repo root; binaries go to build/)
//   Run:    ./build/test_sgemm              # correctness suite + perf at 4096^3
//           ./build/test_sgemm 2048         # perf at 2048^3
//           ./build/test_sgemm 1024 512 256 # perf at N=1024 K=512 M=256
//           ./build/test_sgemm --quick      # correctness suite only, no big perf run
//           ./build/test_sgemm --sweep      # perf at several shapes + summary table
//
// Tensor-core kernels (TF32 / FP16 inputs) are held to a looser tolerance and
// are also compared against cuBLAS at the same input precision -- the
// like-for-like baseline -- as well as against FP32 cuBLAS.
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
#include <cuda_fp16.h>

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

// tensor-core GEMM (FP16 inputs, FP32 accumulate, WMMA), kernel tcgemm
#include "../tensor_core/tc_gemm.cu"
static constexpr int TC_BM_ = BM;
#undef BM
#undef BK
#undef TN
#undef NTHREADS
#undef PAD
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

static const double REL_TOL  = 1e-3;  // relative error accepted vs the reference
static const double LOWP_TOL = 1e-2;  // TF32/FP16 inputs keep 10 mantissa bits: ~1e-4..1e-3 error expected

// input precision of a kernel, and which cuBLAS it is compared against
enum Prec { P_FP32 = 0, P_TF32 = 1, P_FP16 = 2, NUM_PREC = 3 };
static const char* PREC_NAME[NUM_PREC] = { "FP32", "TF32", "FP16" };

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

// FP16 tensor cores (WMMA): 256 threads = 8 warps, each a 64x32 warp tile of
// 16x16 fragments. The epilogue stores whole fragments with no bounds check, so
// the rows and cols of C must be multiples of the 128 block tile.
static void launchTcGemm(const float* dA, const float* dB, float* dC, int N, int K, int M) {
    dim3 block(256);
    dim3 grid((M + TC_BM_ - 1) / TC_BM_, (N + TC_BM_ - 1) / TC_BM_);
    tcgemm<<<grid, block>>>((float*)dA, (float*)dB, dC, N, K, M);
}

struct Kernel {
    const char* name;
    LaunchFn    launch;
    int         align;      // K and M must be multiples of this (float4 kernels need 4)
    int         tileAlign;  // N and M (rows and cols of C) must be multiples of this
    int         prec;       // Prec: input precision; non-FP32 gets LOWP_TOL and a matching cuBLAS
};

static Kernel g_kernels[] = {
    { "naive",          launchNaive,              1,   1, P_FP32 },
    { "sharedmem",      launchSharedMem,          1,   1, P_FP32 },
    { "threadtile",     launchThreadTile,         1,   1, P_FP32 },
    { "sharedthread",   launchSharedThreadTile,   1,   1, P_FP32 },
    { "sharedthreadv2", launchSharedThreadTileV2, 1,   1, P_FP32 },
    { "transpose",      launchTranspose,          4,   1, P_FP32 },
    { "doublebuffer",   launchDoubleBuffer,       4,   1, P_FP32 },
    { "bk16",           launchBk16,               4,   1, P_FP32 },
    { "warptile",       launchWarpTile,           4,   1, P_FP32 },
    { "tc_gemm (fp16)", launchTcGemm,             4, 128, P_FP16 },
};

// A misaligned float4 access is a sticky device error that would poison every
// later test, so shapes that violate a kernel's alignment precondition are
// skipped and reported rather than run.
static bool shapeOk(const Kernel& k, int N, int K, int M) {
    return (K % k.align == 0) && (M % k.align == 0) &&
           (N % k.tileAlign == 0) && (M % k.tileAlign == 0);
}
static const char* skipReason(const Kernel& k, int N, int K, int M, char* buf, size_t len) {
    if (K % k.align || M % k.align) snprintf(buf, len, "skip (K,M %% %d)", k.align);
    else                            snprintf(buf, len, "skip (N,M %% %d)", k.tileAlign);
    (void)N;
    return buf;
}
static double tolFor(const Kernel& k) { return k.prec != P_FP32 ? LOWP_TOL : REL_TOL; }
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

// cuBLAS handles, created in main(): g_fp32 in default math (true FP32 SGEMM),
// g_tf32 in TF32 tensor-op mode. The FP16 baseline is cublasGemmEx on FP16
// copies of A and B (made once per fixture, not timed) with FP32 accumulate.
static cublasHandle_t g_fp32 = nullptr;
static cublasHandle_t g_tf32 = nullptr;

__global__ void floatToHalfKernel(const float* in, __half* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(in[i]);
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
    __half *dAh = nullptr, *dBh = nullptr;     // FP16 copies, for the cuBLAS FP16 baseline
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
        CUDA_CHECK(cudaMalloc(&dAh, hA.size() * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dBh, hB.size() * sizeof(__half)));
        floatToHalfKernel<<<(unsigned)((hA.size() + 255) / 256), 256>>>(dA, dAh, hA.size());
        floatToHalfKernel<<<(unsigned)((hB.size() + 255) / 256), 256>>>(dB, dBh, hB.size());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(ref.data(), dRef, ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void teardown() {
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef); cudaFree(dAh); cudaFree(dBh);
        dA = dB = dC = dRef = nullptr; dAh = dBh = nullptr;
    }
};

// cuBLAS at a given input precision, into fx.dC
static void cublasRun(int prec, Fixture& fx) {
    const float one = 1.0f, zero = 0.0f;
    if (prec == P_FP32) cublasReference(g_fp32, fx.dA, fx.dB, fx.dC, fx.N, fx.K, fx.M);
    else if (prec == P_TF32) cublasReference(g_tf32, fx.dA, fx.dB, fx.dC, fx.N, fx.K, fx.M);
    else  // FP16 in, FP32 accumulate and out; same column-major trick as cublasReference
        cublasGemmEx(g_fp32, CUBLAS_OP_N, CUBLAS_OP_N, fx.M, fx.N, fx.K, &one,
                     fx.dBh, CUDA_R_16F, fx.M, fx.dAh, CUDA_R_16F, fx.K, &zero,
                     fx.dC, CUDA_R_32F, fx.M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

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

// Confirms the low-precision baselines really run at low precision: TF32/FP16
// rounding makes them differ from FP32 by far more than FP32 rounding does
// (~1e-7). A ~0 difference would mean the "TF32" baseline was secretly FP32.
static void selfTestLowPrec(cublasHandle_t h) {
    Fixture fx;
    fx.setup(512, 512, 512, h);
    std::vector<float> out(fx.ref.size());
    for (int p = P_TF32; p < NUM_PREC; ++p) {
        cublasRun(p, fx);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), fx.dC, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        Diff d = compareResults(out, fx.ref);
        printf("  cuBLAS %s vs cuBLAS FP32:      max rel %.2e  %s\n", PREC_NAME[p], d.maxRel,
               d.maxRel > 1e-6 ? "(low precision active)" : "(no difference: NOT running at low precision)");
    }
    fx.teardown();
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
        { 256, 100, 384, "K not a multiple of 16"       },
        {1024,1024,1024, "square"                       },
    };
    const int nShapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    printf("\n=== correctness (vs cuBLAS FP32, relative tolerance %.0e; TF32/FP16 kernels %.0e, error shown) ===\n\n",
           REL_TOL, LOWP_TOL);
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
            if (!shapeOk(g_kernels[k], fx.N, fx.K, fx.M)) {
                printf("%17s", skipReason(g_kernels[k], fx.N, fx.K, fx.M, cell, sizeof(cell)));
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
            const double tol = tolFor(g_kernels[k]);
            if (!d.finite)              { snprintf(cell, sizeof(cell), "NaN/Inf"); failures++; }
            else if (d.maxRel <= tol)   { if (g_kernels[k].prec != P_FP32) snprintf(cell, sizeof(cell), "ok %.1e", d.maxRel);
                                          else                   snprintf(cell, sizeof(cell), "ok"); }
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

// kern == nullptr times cuBLAS at input precision `prec`
static float launchAndTime(const Kernel* kern, int prec, Fixture& fx,
                           cudaEvent_t e0, cudaEvent_t e1, int iters = 1) {
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) {
        if (kern) kern->launch(fx.dA, fx.dB, fx.dC, fx.N, fx.K, fx.M);
        else      cublasRun(prec, fx);
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

struct PerfResult {
    std::vector<double> pct;        // % of cuBLAS FP32, per kernel (-1 = not run)
    std::vector<double> pctMatch;   // % of cuBLAS at the kernel's own precision (low-precision kernels only)
    double refPct[NUM_PREC] = {100.0, -1, -1};   // each cuBLAS precision as % of cuBLAS FP32
};

static PerfResult perfSuite(cublasHandle_t h, int N, int K, int M) {
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
        if (!shapeOk(g_kernels[k], fx.N, fx.K, fx.M)) continue;
        const char* err = nullptr;
        if (!runKernel(g_kernels[k], fx, out, &err)) {
            verdict[k] = "LAUNCH ERR";
            fprintf(stderr, "  [%s] %s\n", g_kernels[k].name, err);
            continue;
        }
        Diff d = compareResults(out, fx.ref);
        verdict[k] = (!d.finite) ? "NaN/Inf" : (d.maxRel <= tolFor(g_kernels[k]) ? "ok" : "FAIL");
        eligible[k] = true;
    }

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // warm-up pass: one launch each; its time sets the batch size per kernel.
    // slots NUM_KERNELS + p are cuBLAS at precision p (FP32, TF32, FP16).
    const int CUB = NUM_KERNELS;
    std::vector<int> iters(NUM_KERNELS + NUM_PREC, 1);
    for (int k = 0; k < NUM_KERNELS; ++k)
        if (eligible[k]) iters[k] = itersFor(launchAndTime(&g_kernels[k], P_FP32, fx, e0, e1));
    for (int p = 0; p < NUM_PREC; ++p)
        iters[CUB + p] = itersFor(launchAndTime(nullptr, p, fx, e0, e1));

    // clock-settle: on a power/thermally limited GPU the clock ramps up at the
    // start of a run and is pulled back down over the next second or two. Keep
    // the GPU busy until that governor has reached its steady state, so the
    // timed rounds see one clock rather than a moving one. (Locking the clock
    // with `nvidia-smi -lgc` is better still, but needs admin.)
    {
        float settled = 0.0f;
        while (settled < SETTLE_MS) {
            float ms = launchAndTime(nullptr, P_FP32, fx, e0, e1);
            if (ms < 0) break;
            settled += ms;
        }
    }

    // interleaved timed rounds
    std::vector<std::vector<float>> samples(NUM_KERNELS + NUM_PREC);
    for (int r = 0; r < PERF_ROUNDS; ++r) {
        for (int k = 0; k < NUM_KERNELS; ++k) {
            if (!eligible[k]) continue;
            float ms = launchAndTime(&g_kernels[k], P_FP32, fx, e0, e1, iters[k]);
            if (ms < 0) { eligible[k] = false; verdict[k] = "LAUNCH ERR"; continue; }
            samples[k].push_back(ms);
        }
        for (int p = 0; p < NUM_PREC; ++p)
            samples[CUB + p].push_back(launchAndTime(nullptr, p, fx, e0, e1, iters[CUB + p]));
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

    double gfRef[NUM_PREC];
    for (int p = 0; p < NUM_PREC; ++p) gfRef[p] = flop / (median(samples[CUB + p]) / 1e3) / 1e9;
    const double gfCublas = gfRef[P_FP32];

    printf("%-22s %12s %14s %12s %12s %9s %10s\n",
           "kernel", "time (ms)", "GFLOP/s", "% of cuBLAS", "% same-prec", "spread", "correct");
    printf("%-22s %12s %14s %12s %12s %9s %10s\n",
           "----------------------", "------------", "--------------", "------------", "------------",
           "---------", "----------");
    for (int k = 0; k < NUM_KERNELS; ++k) {
        if (!eligible[k]) {
            printf("%-22s %12s %14s %12s %12s %9s %10s\n", g_kernels[k].name, "-", "-", "-", "-", "-", verdict[k]);
            continue;
        }
        float ms = median(samples[k]);
        double gf = flop / (ms / 1e3) / 1e9;
        char tf[16] = "-";
        if (g_kernels[k].prec != P_FP32) snprintf(tf, sizeof(tf), "%.1f%%", 100.0 * gf / gfRef[g_kernels[k].prec]);
        printf("%-22s %12.3f %14.2f %11.1f%% %12s %8.1f%% %10s\n",
               g_kernels[k].name, ms, gf, 100.0 * gf / gfCublas, tf, spread(samples[k]), verdict[k]);
    }
    for (int p = 0; p < NUM_PREC; ++p) {
        char nm[32];
        snprintf(nm, sizeof(nm), "cuBLAS %s", p == P_FP32 ? "SGEMM (FP32)" : PREC_NAME[p]);
        printf("%-22s %12.3f %14.2f %11.1f%% %12s %8.1f%% %10s\n",
               nm, median(samples[CUB + p]), gfRef[p], 100.0 * gfRef[p] / gfCublas,
               p == P_FP32 ? "-" : "100.0%", spread(samples[CUB + p]), "ref");
    }
    printf("\n  spread = (max - min) / median across rounds. Large spreads mean the clock\n"
           "  was moving; differences between kernels smaller than the spread are noise.\n"
           "  %% same-prec = vs cuBLAS at the kernel's own input precision (TF32 or FP16\n"
           "  inputs, FP32 accumulate) -- the like-for-like baseline for tensor-core kernels.\n");

    PerfResult res;
    res.pct.assign(NUM_KERNELS, -1.0);
    res.pctMatch.assign(NUM_KERNELS, -1.0);
    for (int k = 0; k < NUM_KERNELS; ++k) {
        if (!eligible[k]) continue;
        const double gf = flop / (median(samples[k]) / 1e3) / 1e9;
        res.pct[k] = 100.0 * gf / gfCublas;
        if (g_kernels[k].prec != P_FP32) res.pctMatch[k] = 100.0 * gf / gfRef[g_kernels[k].prec];
    }
    for (int p = 0; p < NUM_PREC; ++p) res.refPct[p] = 100.0 * gfRef[p] / gfCublas;
    fx.teardown();
    return res;
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
    std::vector<PerfResult> table;
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
            if (table[i].pct[k] < 0) printf("%16s", "skip");
            else                     printf("%15.1f%%", table[i].pct[k]);
        }
        printf("\n");
    }
    for (int p = P_TF32; p < NUM_PREC; ++p) {
        char nm[32]; snprintf(nm, sizeof(nm), "cuBLAS %s", PREC_NAME[p]);
        printf("%-16s", nm);
        for (int i = 0; i < NUM_SWEEP; ++i) printf("%15.1f%%", table[i].refPct[p]);
        printf("\n");
    }

    printf("\n  tensor-core kernels as %% of cuBLAS at the same input precision:\n\n");
    for (int k = 0; k < NUM_KERNELS; ++k) {
        if (g_kernels[k].prec == P_FP32) continue;
        printf("%-16s", g_kernels[k].name);
        for (int i = 0; i < NUM_SWEEP; ++i) {
            if (table[i].pctMatch[k] < 0) printf("%16s", "skip");
            else                          printf("%15.1f%%", table[i].pctMatch[k]);
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
    g_fp32 = h;
    cublasCreate(&g_tf32);
    cublasSetMathMode(g_tf32, CUBLAS_TF32_TENSOR_OP_MATH);

    printf("\n=== harness self-test ===\n\n");
    bool cmpOk = selfTestComparator();
    printf("  comparator detects corruption:    %s\n", cmpOk ? "ok" : "BROKEN");
    bool refOk = selfTestReference(h);
    selfTestLowPrec(h);
    if (!cmpOk || !refOk) {
        fprintf(stderr, "\nHarness self-test failed; kernel results below cannot be trusted.\n");
        cublasDestroy(h); cublasDestroy(g_tf32);
        return 2;
    }

    if (sweep) {
        sweepSuite(h);
        cublasDestroy(h); cublasDestroy(g_tf32);
        return 0;
    }

    int failures = correctnessSuite(h);
    if (!quick) perfSuite(h, N, K, M);

    printf("\n%s\n", failures == 0 ? "All correctness cases passed."
                                   : "Some correctness cases failed (see table).");
    cublasDestroy(h); cublasDestroy(g_tf32);
    return failures == 0 ? 0 : 1;
}
