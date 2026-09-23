// Unit test for softmaxRows(float* S, int N) in softmax.cu.
//
// Launch model: ONE BLOCK PER ROW.
//   grid  = N                       (blockIdx.x = row)
//   block = N / TN                  (each thread owns TN consecutive elements
//                                    of its row; threadIdx.x = column chunk)
// With TN = 16 that is launchable up to N = 16384 (1024 threads per block).
//
// If you change the kernel to a fixed block size that strides across the row,
// pass --block=<threads> and the test launches that instead of N / TN.
//
// The test assumes nothing about correctness. For each N it reports, per row,
// the row sum (softmax rows must sum to 1) and the max error against a
// double-precision CPU softmax, then lists the first few rows that are off.
//
//   Build:  make build/test_softmax   (from the repo root)
//   Run:    ./build/test_softmax                 # N = 16, 64, 256, 1024, 4096
//           ./build/test_softmax 512             # one size
//           ./build/test_softmax 4096 --block=256

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#include "../attention/softmax.cu"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const double TOL = 1e-4;            // max |got - ref| per row; softmax values are in [0, 1]
static const int    ELEMS_PER_THREAD = 16; // must match TN in softmax.cu

static void cpuSoftmaxRows(const std::vector<float>& in, std::vector<double>& out, int N) {
    out.resize((size_t)N * N);
    for (int r = 0; r < N; ++r) {
        const float* row = &in[(size_t)r * N];
        double m = -INFINITY;
        for (int j = 0; j < N; ++j) m = std::max(m, (double)row[j]);
        double l = 0.0;
        for (int j = 0; j < N; ++j) { out[(size_t)r * N + j] = std::exp((double)row[j] - m); l += out[(size_t)r * N + j]; }
        for (int j = 0; j < N; ++j) out[(size_t)r * N + j] /= l;
    }
}

static bool testSize(int N, int blockOverride, float range, const char* rangeNote) {
    printf("=== N = %d   scores in [%g, %g]  %s ===\n", N, -range, range, rangeNote);
    int blockSize;
    if (blockOverride > 0) {
        blockSize = blockOverride;
        printf("  launch: grid = N = %d blocks (one per row), block = %d threads (--block override)\n", N, blockSize);
    } else {
        if (N % ELEMS_PER_THREAD != 0) {
            printf("  N must be a multiple of %d for the N/TN mapping; skipping\n\n", ELEMS_PER_THREAD);
            return true;
        }
        // round up to whole warps: shuffle reductions need all 32 lanes of a
        // warp to exist. Threads with tx*16 >= N must be inactive in the kernel.
        blockSize = ((N / ELEMS_PER_THREAD + 31) / 32) * 32;
        printf("  launch: grid = N = %d blocks (one per row), block = N/%d rounded up to warps = %d threads"
               " (%d active)\n", N, ELEMS_PER_THREAD, blockSize, N / ELEMS_PER_THREAD);
    }
    if (blockSize > 1024) {
        printf("  exceeds the 1024-thread block limit; not launchable with this mapping\n\n");
        return true;
    }

    // input: scores in a realistic range, seeded so runs are repeatable
    std::vector<float> h((size_t)N * N);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-range, range);
    for (float& x : h) x = dist(gen);

    std::vector<double> ref;
    cpuSoftmaxRows(h, ref, N);

    float* dS;
    CUDA_CHECK(cudaMalloc(&dS, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dS, h.data(), (size_t)N * N * sizeof(float), cudaMemcpyHostToDevice));

    softmaxRows<<<N, blockSize>>>(dS, N);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("  LAUNCH ERROR: %s\n", cudaGetErrorString(e));
        printf("  (this fault is sticky: the CUDA context is now dead, so no later size can run\n"
               "   in this process. Fix this size first, or run a single size to test the others.)\n\n");
        printf("FAILURES PRESENT\n");
        exit(1);
    }

    std::vector<float> got((size_t)N * N);
    CUDA_CHECK(cudaMemcpy(got.data(), dS, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dS);

    // per-row diagnostics
    int rowsOk = 0, rowsBadSum = 0, rowsBadVal = 0, rowsNonFinite = 0, shown = 0;
    double worstErr = 0.0;
    for (int r = 0; r < N; ++r) {
        double sum = 0.0, err = 0.0;
        bool finite = true;
        for (int j = 0; j < N; ++j) {
            float g = got[(size_t)r * N + j];
            if (!std::isfinite(g)) { finite = false; break; }
            sum += g;
            err = std::max(err, std::fabs((double)g - ref[(size_t)r * N + j]));
        }
        bool sumOk = finite && std::fabs(sum - 1.0) < 1e-3;
        bool valOk = finite && err <= TOL;
        if (!finite) rowsNonFinite++;
        else if (!sumOk) rowsBadSum++;
        else if (!valOk) rowsBadVal++;
        else rowsOk++;
        if (finite) worstErr = std::max(worstErr, err);
        if (!(sumOk && valOk) && shown < 6) {
            if (!finite) printf("  row %5d: NaN/Inf\n", r);
            else printf("  row %5d: sum = %.6f   max |got-ref| = %.3e   got[0..3] = %.4f %.4f %.4f %.4f   ref[0..3] = %.4f %.4f %.4f %.4f\n",
                        r, sum, err,
                        got[(size_t)r * N + 0], got[(size_t)r * N + 1], got[(size_t)r * N + 2], got[(size_t)r * N + 3],
                        ref[(size_t)r * N + 0], ref[(size_t)r * N + 1], ref[(size_t)r * N + 2], ref[(size_t)r * N + 3]);
            shown++;
        }
    }
    if (shown == 6 && rowsOk < N - 6) printf("  ... (%d more rows off)\n", N - rowsOk - 6);

    printf("  rows: %d ok, %d sum != 1, %d values off, %d NaN/Inf   (worst |got-ref| = %.3e)\n",
           rowsOk, rowsBadSum, rowsBadVal, rowsNonFinite, worstErr);
    bool pass = rowsOk == N;
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

int main(int argc, char** argv) {
    int singleN = 0, blockOverride = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strncmp(argv[i], "--block=", 8)) blockOverride = atoi(argv[i] + 8);
        else if (argv[i][0] != '-') singleN = atoi(argv[i]);
    }

    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s   max threads per block: %d\n\n", p.name, p.maxThreadsPerBlock);

    // two input ranges: gentle scores, and scores large enough that exp()
    // overflows float unless the row max is subtracted first
    const float ranges[]     = { 3.0f, 100.0f };
    const char* rangeNote[]  = { "(gentle)", "(large: exp overflows without max-subtraction)" };
    bool all = true;
    for (int ri = 0; ri < 2; ++ri) {
        if (singleN > 0) {
            all = testSize(singleN, blockOverride, ranges[ri], rangeNote[ri]) && all;
        } else {
            const int sizes[] = { 16, 64, 256, 1024, 4096 };
            for (int n : sizes) all = testSize(n, blockOverride, ranges[ri], rangeNote[ri]) && all;
        }
    }
    printf("%s\n", all ? "ALL PASSED" : "FAILURES PRESENT");
    return all ? 0 : 1;
}
