#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <float.h>


#define TN 16

__global__ void softmaxRows(float* S, int N) {
    const int ty = blockIdx.x;
    const int tx = threadIdx.x;

    const int lane = tx % 32;
    const int warpId = tx / 32;
    const int nWarps = (blockDim.x + 31) / 32;

    __shared__ float shared_sum[32];
    __shared__ float red[32];



    float s[TN];

    const bool active = tx * TN < N;

    float v = -FLT_MAX;

    if (active) {
        for (int i = 0; i < TN; ++i) {
            s[i] = S[ty * N + tx * TN + i];
            v = fmaxf(v,s[i]);
        }

    }
    

    for (int offset = 16; offset > 0; offset /= 2) {
        float other =
            __shfl_xor_sync(0xffffffff, v, offset);

        v = fmaxf(v, other);
    }
    if (lane == 0) red[warpId] = v;

    __syncthreads();

    if (warpId == 0) {
        float x = (lane < nWarps)
                    ? red[lane]
                    : -FLT_MAX;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other =
                __shfl_xor_sync(0xffffffff, x, offset);

            x = fmaxf(x, other);
        }
        if (lane == 0)
            red[0] = x;
    }

    __syncthreads();
    float m = red[0];


    float sum = 0.0f;

    if (active) {
        for (int i = 0; i < TN; ++i) {
            s[i] = expf(s[i] - m);
            sum += s[i];
        }
    }
    

    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xffffffff, sum, offset);
        sum += other;
    }

    if (lane == 0) shared_sum[warpId] = sum;
    __syncthreads();

    if (warpId == 0) {
        float x = (lane < nWarps)
                    ? shared_sum[lane]
                    : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other =
                __shfl_xor_sync(0xffffffff, x, offset);

            x += other;
        }
        if (lane == 0)
            shared_sum[0] = x;
    }
    __syncthreads();
    float total = shared_sum[0];

    if (active) {
        for (int i = 0; i < TN; i++) {
            S[ty * N + tx * TN + i] = s[i] / total;
        }
    }

}

// ---- launcher, so other files can call the kernel without writing <<< >>> ----
// One block per row, N/TN threads rounded up to whole warps; the kernel's
// `active` guard idles the padding threads. rows = batch * N for a batched S.
// Requires N % TN == 0 and N / TN <= 1024.
inline void softmaxRun(float* S, int rows, int N, cudaStream_t stream = 0) {
    const int threads = ((N / TN + 31) / 32) * 32;
    softmaxRows<<<rows, threads, 0, stream>>>(S, N);
}
