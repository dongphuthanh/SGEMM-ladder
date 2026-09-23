// naive_attn.cu -- unfused attention built from your GEMM and your softmax.
//
//   O = softmax(Q . K^T / sqrt(d)) . V
//
// Four launches per batch element, in order:
//
//   1. transposeScale   Kt = K^T * (1/sqrt(d))        [d x N]   (scratch)
//   2. gemm::run        S  = Q . Kt                   [N x N]   (scratch)
//   3. softmaxRun       S  = softmax(S)  row-wise, in place
//   4. gemm::run        O  = S . V                    [N x d]
//
// Nothing here is a kernel launch you write yourself: each kernel file owns its
// launcher, and this file just calls them in sequence on one stream. Launches
// on the same stream execute in order, so no synchronisation is needed between
// steps -- step 2 cannot start before step 1 finishes.
//
// The 1/sqrt(d) is folded into the transpose (step 1) so the softmax kernel
// runs unchanged from its unit test.
//
// Materialises S, so workspace is batch*N*N floats: this is the baseline the
// fused kernel is measured against, on memory and on time.
//
// Standalone:  nvcc -O2 -arch=sm_120 attention/naive_attn.cu -o build/naive_attn && ./build/naive_attn
// In harness:  included by test/test_attn.cu; see the KERNEL IMPORT section there.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// Order matters. softmax.cu does `#define TN 16`, and a macro rewrites every
// later `TN` it sees -- including gemm::TN. gemm.cuh keeps its constants inside
// a namespace, so it is safe *before* the macro exists; then the macro is
// removed again so nothing after this line is affected. (Your stub already
// had the #undef -- this is why it was needed.)
#include "gemm.cuh"
#include "softmax.cu"
#undef TN

// ------------------------------------------------------- step 1: K^T * scale
// Kt[k][n] = K[n][k] * scale.  One thread per element; the read of K is
// coalesced (consecutive threads read consecutive k), the write of Kt is not.
// K is N x d -- tiny next to the N x N matrices -- so this doesn't matter here.
__global__ void transposeScaleKernel(const float* K, float* Kt, int N, int d, float scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // flat index over N*d
    if (idx >= N * d) return;
    const int n = idx / d;
    const int k = idx % d;
    Kt[k * N + n] = K[n * d + k] * scale;
}

static void transposeScale(const float* K, float* Kt, int N, int d, float scale,
                           cudaStream_t stream = 0) {
    const int threads = 256;
    const int blocks = (N * d + threads - 1) / threads;
    transposeScaleKernel<<<blocks, threads, 0, stream>>>(K, Kt, N, d, scale);
}

// ----------------------------------------------------------- harness interface
// Scratch memory this variant needs beyond Q/K/V/O: the N x N score matrix and
// the transposed K, per batch element.
size_t naiveWorkspace(int batch, int N, int d) {
    return (size_t)batch * N * N * sizeof(float)      // S
         + (size_t)batch * N * d * sizeof(float);     // Kt
}

void naiveRun(const float* Q, const float* K, const float* V, float* O,
              int batch, int N, int d, float* workspace) {
    float* S  = workspace;                                 // batch * N * N
    float* Kt = workspace + (size_t)batch * N * N;         // batch * N * d
    const float scale = 1.0f / std::sqrt((float)d);

    for (int b = 0; b < batch; ++b) {
        const size_t qkv = (size_t)b * N * d;              // offset into Q, K, V, O, Kt
        const size_t ss  = (size_t)b * N * N;              // offset into S

        transposeScale(K + qkv, Kt + qkv, N, d, scale);    // Kt = K^T / sqrt(d)
        gemm::run(Q + qkv, Kt + qkv, S + ss, N, d, N);     // S  = Q . Kt      [N x N]
        softmaxRun(S + ss, N, N);                          // S  = softmax rows
        gemm::run(S + ss, V + qkv, O + qkv, N, N, d);      // O  = S . V       [N x d]
    }
}

// ----------------------------------------------------------- standalone demo
// A small run with a CPU check on one row, so the file does something useful
// on its own. The full test is `make attn` from the repo root.
int main() {
    const int batch = 2, N = 256, d = 64;
    const size_t n = (size_t)batch * N * d;

    std::vector<float> hQ(n), hK(n), hV(n), hO(n);
    for (size_t i = 0; i < n; ++i) {
        hQ[i] = (float)((i * 7) % 13) / 13.0f - 0.5f;
        hK[i] = (float)((i * 5) % 11) / 11.0f - 0.5f;
        hV[i] = (float)((i * 3) % 17) / 17.0f - 0.5f;
    }

    float *dQ, *dK, *dV, *dO, *dWs;
    cudaMalloc(&dQ, n * 4); cudaMalloc(&dK, n * 4); cudaMalloc(&dV, n * 4); cudaMalloc(&dO, n * 4);
    cudaMalloc(&dWs, naiveWorkspace(batch, N, d));
    cudaMemcpy(dQ, hQ.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV.data(), n * 4, cudaMemcpyHostToDevice);

    naiveRun(dQ, dK, dV, dO, batch, N, d, dWs);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(hO.data(), dO, n * 4, cudaMemcpyDeviceToHost);

    // CPU check of row 0 of batch 1, in double
    const int b = 1, i = 0;
    const float* q = &hQ[(size_t)b * N * d];
    const float* k = &hK[(size_t)b * N * d];
    const float* v = &hV[(size_t)b * N * d];
    std::vector<double> s(N);
    double m = -INFINITY, scale = 1.0 / std::sqrt((double)d);
    for (int j = 0; j < N; ++j) {
        double acc = 0; for (int t = 0; t < d; ++t) acc += (double)q[i * d + t] * k[j * d + t];
        s[j] = acc * scale; m = std::max(m, s[j]);
    }
    double l = 0; for (int j = 0; j < N; ++j) { s[j] = std::exp(s[j] - m); l += s[j]; }
    double maxErr = 0;
    for (int t = 0; t < d; ++t) {
        double acc = 0; for (int j = 0; j < N; ++j) acc += s[j] * v[j * d + t];
        maxErr = std::max(maxErr, std::fabs(acc / l - hO[(size_t)b * N * d + i * d + t]));
    }
    printf("naive attention  batch=%d N=%d d=%d\n", batch, N, d);
    printf("  O[b=1][row 0][0..3] = %.5f %.5f %.5f %.5f\n",
           hO[(size_t)b * N * d + 0], hO[(size_t)b * N * d + 1], hO[(size_t)b * N * d + 2], hO[(size_t)b * N * d + 3]);
    printf("  max |O - cpu| on that row = %.3e  %s\n", maxErr, maxErr < 1e-4 ? "ok" : "MISMATCH");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dWs);
    return 0;
}
