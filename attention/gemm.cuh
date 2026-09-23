// gemm.cuh -- the SGEMM kernel from ../SGEMM/warptiling.cu, packaged so other
// files can call it.
//
// What changed from warptiling.cu, and why:
//   * no main(), no matMul(): a header holds only the kernel and its launcher.
//     The caller owns the device buffers; the kernel just computes.
//   * the #defines became constexpr ints inside a namespace. A bare
//     `#define TN 8` in a header would silently rewrite every `TN` in every
//     file that includes it -- and softmax.cuh has its own TN = 16.
//   * gemm::run() is the host-side launcher: it computes the grid and launches.
//     Callers never write <<< >>> themselves.
//
// Computes row-major  C[M x N] = A[M x K] . B[K x N]  in fp32.
// Requires K % 4 == 0 and N % 4 == 0 (float4 loads); M, K, N otherwise arbitrary.
#pragma once
#include <cuda_runtime.h>

namespace gemm {

constexpr int BM       = 128;
constexpr int BK       = 16;
constexpr int TN       = 8;
constexpr int NTHREADS = 256;
constexpr int STRIDE_A = NTHREADS / (BK / 4);   // 64 rows of As per pass
constexpr int STRIDE_B = NTHREADS / (BM / 4);   //  8 rows of Bs per pass
constexpr int PASSES_A = BM / STRIDE_A;         // 2
constexpr int PASSES_B = BK / STRIDE_B;         // 2

// ---- the kernel, unchanged apart from the constants above --------------------
__global__ void warpTile(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float shared_a[2][BK][BM];
    __shared__ float shared_b[2][BK][BM];

    float c[TN][TN]{};

    const int warpId  = threadIdx.x / 32;   const int lane    = threadIdx.x % 32;
    const int warpRow = warpId / 4;         const int warpCol = warpId % 4;
    const int laneRow = lane / 4;           const int laneCol = lane % 4;
    const int ty = warpRow * 8 + laneRow;
    const int tx = warpCol * 4 + laneCol;

    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BM / 4);
    const int innerColB = threadIdx.x % (BM / 4);
    const int cRow = blockIdx.y * BM  + innerRowA;
    const int cCol = blockIdx.x * BM + innerColB * 4;
    const int numTiles = (K + BK - 1) / BK;

    //Load slab 0
    int aCol = innerColA * 4;
    float4 tmpA[PASSES_A];
    for (int p = 0; p < PASSES_A; p++) {
        tmpA[p] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (cRow + p * STRIDE_A < M && aCol < K) {
            tmpA[p] = *reinterpret_cast<const float4*>(&A[(cRow + p * STRIDE_A) * K + aCol]);
        }
        shared_a[0][innerColA * 4 + 0][innerRowA + p * STRIDE_A] = tmpA[p].x;
        shared_a[0][innerColA * 4 + 1][innerRowA + p * STRIDE_A] = tmpA[p].y;
        shared_a[0][innerColA * 4 + 2][innerRowA + p * STRIDE_A] = tmpA[p].z;
        shared_a[0][innerColA * 4 + 3][innerRowA + p * STRIDE_A] = tmpA[p].w;
    }

    int bRow = innerRowB;

    float4 tmpB[PASSES_B];
    for (int p = 0; p < PASSES_B; p++) {
        tmpB[p] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (bRow + p * STRIDE_B < K && cCol < N) {
            tmpB[p] = *reinterpret_cast<const float4*>(&B[(bRow + p * STRIDE_B) * N + cCol]);
        }
        *reinterpret_cast<float4*>(&shared_b[0][innerRowB + p * STRIDE_B][innerColB * 4]) = tmpB[p];
    }
    __syncthreads();

    for (int i = 0; i < numTiles; i++) {
        const int cur = i % 2;
        const int next = (i + 1) % 2;

        if (i + 1 < numTiles) {
            aCol = (i + 1) * BK + innerColA * 4;
            for (int p = 0; p < PASSES_A; p++) {
                tmpA[p] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (cRow + p * STRIDE_A < M && aCol < K) {
                    tmpA[p] = *reinterpret_cast<const float4*>(&A[(cRow + p * STRIDE_A) * K + aCol]);
                }
            }

            bRow = (i + 1) * BK + innerRowB;
            for (int p = 0; p < PASSES_B; p++) {
                tmpB[p] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (bRow + p * STRIDE_B < K && cCol < N) {
                    tmpB[p] = *reinterpret_cast<const float4*>(&B[(bRow + p * STRIDE_B) * N + cCol]);
                }
            }
        }
        for (int h = 0; h < BK; h++) {
            float a[TN];
            float b[TN];
            for (int j = 0; j < TN; j++) {
                a[j] = shared_a[cur][h][ty * TN + j];
                b[j] = shared_b[cur][h][tx * TN + j];
            }

            for (int j = 0; j < TN; j++) {
                for (int k = 0; k < TN; k++) {
                    c[j][k] += a[j] * b[k];
                }
            }
        }

        for (int p = 0; p < PASSES_A; p++) {
            shared_a[next][innerColA * 4 + 0][innerRowA + p * STRIDE_A] = tmpA[p].x;
            shared_a[next][innerColA * 4 + 1][innerRowA + p * STRIDE_A] = tmpA[p].y;
            shared_a[next][innerColA * 4 + 2][innerRowA + p * STRIDE_A] = tmpA[p].z;
            shared_a[next][innerColA * 4 + 3][innerRowA + p * STRIDE_A] = tmpA[p].w;
        }

        for (int p = 0; p < PASSES_B; p++) {
            *reinterpret_cast<float4*>(&shared_b[next][innerRowB + p * STRIDE_B][innerColB * 4]) = tmpB[p];
        }

        __syncthreads();
    }

    for (int i = 0; i < TN; i++) {
        for (int j = 0; j < TN; j++) {
            int cRow = blockIdx.y * BM + ty * TN + i;
            int cCol = blockIdx.x * BM + tx * TN + j;

            if (cRow < M && cCol < N) {
                C[cRow * N + cCol] = c[i][j];
            }
        }
    }
}

// ---- the launcher ------------------------------------------------------------
// This is the only thing callers use. It owns the launch geometry, so the
// "128-wide tile, 256 threads" knowledge lives in exactly one place.
inline void run(const float* A, const float* B, float* C, int M, int K, int N,
                cudaStream_t stream = 0) {
    dim3 block(NTHREADS);
    dim3 grid((N + BM - 1) / BM, (M + BM - 1) / BM);
    warpTile<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}

}  // namespace gemm
