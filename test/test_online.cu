// test_online.cu -- debugging harness for online_attn.cu (the tiled fused kernel).
//
//   Build/run:  make online
//               ./build/test_online              all stages, all shapes
//               ./build/test_online --map        print the tile map for every failure, not just the first per stage
//               ./build/test_online 256          only N = 256 (batch 1), all stages
//               ./build/test_online --tc         test tensor_core/tc_attn.cu (FP16) instead,
//                                                at the FP16 tolerance (1e-2)
//
// Launch: one block per BR query rows, 128 threads. The kernel only knows one
// head, so the launcher loops over the batch and offsets the pointers.
// d is fixed at D = 64 (the kernel takes no d argument). Ragged N is
// tested: partial last query tile and partial last KV chunk.
//
// Stages, in the order worth debugging them. Each one isolates part of the kernel:
//
//   1. V = 1     softmax weights sum to 1, so every output must be exactly 1,
//                whatever the scores are. Tests only that o and l are
//                accumulated and normalised consistently, and that every
//                element of O gets written.
//   2. Q = 0     every score is 0, so P is uniform and O[i][c] = mean_j V[j][c]
//                for every row i. Tests the V load, P.V and the store, with
//                S = Q.K^T taken out of the picture.
//   3. random    everything: S = Q.K^T, the max, and the rescale across chunks.
//
// On a failure it prints:
//   - the first wrong element (batch, row, col) and the max error
//   - how many elements were never written (still the 0xAB poison) and NaN/inf
//   - a map of the first failing BR x d tile, one character per 4x4 patch:
//        map rows = 4-row groups    -> (warpId, laneRow)
//        map cols = 4-col groups    -> laneCol 0..7 for cols 0-31, then again for cols 32-63
//        '.' ok   'x' wrong   '-' never written   'N' NaN/inf
//   - for a few wrong rows, whether got / ref is one constant factor along the row
//
// A launch error (illegal address etc.) is sticky: the CUDA context is dead
// after it, so the test stops at the first one.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

// ---------------------------------------------------------------- KERNEL IMPORT
#define main online_main_          // in case online_attn.cu grows a main later
#include "../attention/online_attn.cu"
#undef main

// tensor-core attention (FP16 WMMA), kernel TcAttn. Its macros (BR, BC, D, ...)
// have the same values as online_attn.cu's, so the redefinitions are identical
// and legal. tc_attn.cu uses wmma:: unqualified, so the namespace comes first.
#include <mma.h>
using namespace nvcuda;
#define main tc_attn_main_
#include "../tensor_core/tc_attn.cu"
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

static const int    THREADS = 128;
// D (head dim) comes from online_attn.cu
static double       TOL     = 1e-3;     // max |got - ref| / max |ref|, same metric as test_attn
                                        // (--tc raises it to 1e-2 for FP16 inputs)

// which kernel is under test: onlineRowKernel (default) or TcAttn (--tc)
typedef void (*AttnKernel)(const float*, const float*, const float*, float*, int);
static AttnKernel g_kernel = onlineRowKernel;
static const char* g_kernelName = "online_attn";
static const uint32_t POISON = 0xABABABABu;

static void launchOnline(const float* Q, const float* K, const float* V, float* O,
                         int batch, int N, int d) {
    const int blocks = (N + BR - 1) / BR;
    for (int b = 0; b < batch; ++b) {
        const size_t off = (size_t)b * N * d;
        g_kernel<<<blocks, THREADS>>>(Q + off, K + off, V + off, O + off, N);
    }
}

// ------------------------------------------------------------ CPU double reference
static void cpuAttention(const float* Q, const float* K, const float* V, float* O,
                         int batch, int N, int d) {
    std::vector<double> s(N);
    const double scale = 1.0 / std::sqrt((double)d);
    for (int b = 0; b < batch; ++b) {
        const size_t off = (size_t)b * N * d;
        const float *q = Q + off, *k = K + off, *v = V + off;
        float* o = O + off;
        for (int i = 0; i < N; ++i) {
            double m = -INFINITY;
            for (int j = 0; j < N; ++j) {
                double acc = 0.0;
                for (int t = 0; t < d; ++t) acc += (double)q[(size_t)i * d + t] * k[(size_t)j * d + t];
                s[j] = acc * scale;
                m = std::max(m, s[j]);
            }
            double l = 0.0;
            for (int j = 0; j < N; ++j) { s[j] = std::exp(s[j] - m); l += s[j]; }
            for (int t = 0; t < d; ++t) {
                double acc = 0.0;
                for (int j = 0; j < N; ++j) acc += s[j] * v[(size_t)j * d + t];
                o[(size_t)i * d + t] = (float)(acc / l);
            }
        }
    }
}

// -------------------------------------------------------------------- stages
enum Stage { V_ONES, Q_ZERO, RANDOM, NUM_STAGES };
static const char* STAGE_NAME[] = { "V = 1", "Q = 0", "random" };
static const char* STAGE_TESTS[] = {
    "o and l consistent, every output written",
    "V load, P.V and the store (S taken out)",
    "everything incl. S, max, rescale",
};

static void fillInputs(Stage st, std::vector<float>& Q, std::vector<float>& K, std::vector<float>& V) {
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : Q) x = dist(gen);
    for (float& x : K) x = dist(gen);
    for (float& x : V) x = dist(gen);
    if (st == V_ONES) std::fill(V.begin(), V.end(), 1.0f);
    if (st == Q_ZERO) std::fill(Q.begin(), Q.end(), 0.0f);
}

// ------------------------------------------------------------------ checking
static bool isPoison(float f) { uint32_t u; memcpy(&u, &f, 4); return u == POISON; }

struct Check {
    double scale = 0, maxAbs = 0, err = 0;
    size_t wrong = 0, unwritten = 0, nonfinite = 0;
    long long firstBad = -1;
};

static Check check(const std::vector<float>& got, const std::vector<float>& ref) {
    Check c;
    for (float r : ref) c.scale = std::max(c.scale, (double)std::fabs(r));
    const double tolAbs = TOL * (c.scale > 0 ? c.scale : 1.0);
    for (size_t i = 0; i < got.size(); ++i) {
        bool bad = false;
        if (isPoison(got[i]))                { ++c.unwritten; bad = true; }
        else if (!std::isfinite(got[i]))     { ++c.nonfinite; bad = true; }
        else {
            const double a = std::fabs((double)got[i] - ref[i]);
            c.maxAbs = std::max(c.maxAbs, a);
            if (a > tolAbs) bad = true;
        }
        if (bad) { ++c.wrong; if (c.firstBad < 0) c.firstBad = (long long)i; }
    }
    c.err = (c.unwritten || c.nonfinite) ? INFINITY : c.maxAbs / (c.scale > 0 ? c.scale : 1.0);
    return c;
}

// One character per 4x4 patch of the BR x d tile that contains `row`.
static void printTileMap(const std::vector<float>& got, const std::vector<float>& ref,
                         double scale, int b, int row, int N, int d) {
    const int tile = row / BR;
    const double tolAbs = TOL * (scale > 0 ? scale : 1.0);
    printf("      tile map: batch %d, query rows %d-%d  (one char per 4x4 patch of O)\n",
           b, tile * BR, tile * BR + BR - 1);
    printf("                        laneCol  01234567   01234567\n");
    printf("                        O cols   0-31       32-63\n");
    for (int rg = 0; rg < BR / 4; ++rg) {
        const int r0 = tile * BR + rg * 4;
        if (r0 >= N) break;
        printf("        rows %4d-%-4d  w%d r%d    ", r0, r0 + 3, rg / 4, rg % 4);
        for (int cg = 0; cg < d / 4; ++cg) {
            if (cg == 8) printf("   ");
            char ch = '.';
            for (int i = 0; i < 4 && r0 + i < N; ++i)
                for (int j = 0; j < 4; ++j) {
                    const size_t idx = ((size_t)b * N + r0 + i) * d + cg * 4 + j;
                    const float g = got[idx];
                    char e = '.';
                    if (isPoison(g))                               e = '-';
                    else if (!std::isfinite(g))                    e = 'N';
                    else if (std::fabs((double)g - ref[idx]) > tolAbs) e = 'x';
                    // precedence: N > - > x > .
                    if (e == 'N' || (e == '-' && ch != 'N') || (e == 'x' && ch == '.')) ch = e;
                }
            putchar(ch);
        }
        putchar('\n');
    }
}

// For up to `maxRows` wrong rows in the tile: is got = factor * ref along the row?
static void printRowRatios(const std::vector<float>& got, const std::vector<float>& ref,
                           double scale, int b, int row, int N, int d, int maxRows) {
    const int tile = row / BR;
    const double tolAbs = TOL * (scale > 0 ? scale : 1.0);
    int shown = 0;
    for (int r = tile * BR; r < std::min(N, tile * BR + BR) && shown < maxRows; ++r) {
        const size_t base = ((size_t)b * N + r) * d;
        bool rowBad = false, rowUsable = true;
        for (int c = 0; c < d; ++c) {
            const float g = got[base + c];
            if (isPoison(g) || !std::isfinite(g)) rowUsable = false;
            else if (std::fabs((double)g - ref[base + c]) > tolAbs) rowBad = true;
        }
        if (!rowBad || !rowUsable) continue;
        double lo = INFINITY, hi = -INFINITY, sum = 0; int n = 0;
        for (int c = 0; c < d; ++c) {
            if (std::fabs(ref[base + c]) < 0.05 * scale) continue;   // ratio meaningless near 0
            const double q = got[base + c] / (double)ref[base + c];
            lo = std::min(lo, q); hi = std::max(hi, q); sum += q; ++n;
        }
        printf("      row %4d: got[0..3] = %9.4f %9.4f %9.4f %9.4f\n", r,
               got[base], got[base + 1], got[base + 2], got[base + 3]);
        printf("                ref[0..3] = %9.4f %9.4f %9.4f %9.4f",
               ref[base], ref[base + 1], ref[base + 2], ref[base + 3]);
        if (n > 0) {
            const double mean = sum / n;
            if ((hi - lo) <= 1e-3 * std::fabs(mean)) printf("   got/ref = %.5g on every column (one factor for the row)\n", mean);
            else                                     printf("   got/ref ranges %.3g .. %.3g (not one factor)\n", lo, hi);
        } else printf("\n");
        ++shown;
    }
}

// ---------------------------------------------------------------------- run one
// returns false on a launch error (the context is then unusable)
static bool runCase(Stage st, int batch, int N, bool mapAlways, bool* printedMap, bool* passed) {
    const size_t n = (size_t)batch * N * D;
    std::vector<float> hQ(n), hK(n), hV(n), got(n), ref(n);
    fillInputs(st, hQ, hK, hV);

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, n * 4)); CUDA_CHECK(cudaMalloc(&dK, n * 4));
    CUDA_CHECK(cudaMalloc(&dV, n * 4)); CUDA_CHECK(cudaMalloc(&dO, n * 4));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dO, 0xAB, n * 4));

    launchOnline(dQ, dK, dV, dO, batch, N, D);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    printf("  %-7s  batch %d  N %5d   ", STAGE_NAME[st], batch, N);
    if (e != cudaSuccess) {
        printf("LAUNCH ERR: %s\n", cudaGetErrorString(e));
        return false;
    }
    CUDA_CHECK(cudaMemcpy(got.data(), dO, n * 4, cudaMemcpyDeviceToHost));
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);

    cpuAttention(hQ.data(), hK.data(), hV.data(), ref.data(), batch, N, D);
    Check c = check(got, ref);
    *passed = (c.wrong == 0);

    if (*passed) { printf("ok     err %.2e\n", c.err); return true; }

    const size_t fb = (size_t)c.firstBad;
    const int fbB = (int)(fb / ((size_t)N * D)), fbR = (int)(fb / D % N), fbC = (int)(fb % D);
    char errBuf[32];
    snprintf(errBuf, sizeof errBuf, std::isfinite(c.err) ? "%.2e" : "inf", c.err);
    printf("FAIL   err %s  wrong %zu/%zu  unwritten %zu  nan/inf %zu\n",
           errBuf, c.wrong, n, c.unwritten, c.nonfinite);
    printf("      first wrong: batch %d row %d col %d   got %.6g  ref %.6g\n",
           fbB, fbR, fbC, got[fb], ref[fb]);
    if (mapAlways || !*printedMap) {
        printTileMap(got, ref, c.scale, fbB, fbR, N, D);
        printRowRatios(got, ref, c.scale, fbB, fbR, N, D, 3);
        *printedMap = true;
    }
    return true;
}

// ------------------------------------------------------------------------ main
int main(int argc, char** argv) {
    bool mapAlways = false;
    int onlyN = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--map")) mapAlways = true;
        else if (!strcmp(argv[i], "--tc")) { g_kernel = TcAttn; g_kernelName = "tc_attn (fp16)"; TOL = 1e-2; }
        else onlyN = atoi(argv[i]);
    }

    cudaFuncAttributes fa;
    CUDA_CHECK(cudaFuncGetAttributes(&fa, g_kernel));
    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, g_kernel, THREADS, 0));
    printf("%s: BR=%d BC=%d  %d threads  d=%d   regs/thread %d  static smem %zu B  local (spill) %zu B\n"
           "             occupancy: %d blocks/SM = %d warps/SM\n\n",
           g_kernelName, BR, BC, THREADS, D, fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes,
           blocksPerSM, blocksPerSM * THREADS / 32);

    struct Shape { int batch, N; };
    std::vector<Shape> shapes = { {1, 64}, {1, 128}, {1, 256}, {2, 256}, {1, 1024}, {4, 1024}, {1, 4096},
                                  // ragged: partial last query tile and partial last KV chunk
                                  {1, 1}, {1, 31}, {1, 33}, {1, 100}, {3, 333}, {1, 1000} };
    if (onlyN) shapes = { {1, onlyN} };

    int stagesPassed = 0;
    for (int s = 0; s < NUM_STAGES; ++s) {
        printf("stage %d: %s   -- tests %s\n", s + 1, STAGE_NAME[s], STAGE_TESTS[s]);
        bool printedMap = false, stageOk = true;
        for (const Shape& sh : shapes) {
            bool passed = false;
            if (!runCase((Stage)s, sh.batch, sh.N, mapAlways, &printedMap, &passed)) {
                printf("\nlaunch error is sticky: the CUDA context is gone, stopping here.\n");
                return 1;
            }
            stageOk &= passed;
        }
        stagesPassed += stageOk;
        printf("\n");
    }
    printf("%s  (tolerance %.0e): %d / %d stages pass\n", g_kernelName, TOL, stagesPassed, (int)NUM_STAGES);
    return stagesPassed == NUM_STAGES ? 0 : 1;
}
