#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <mma.h>

using namespace nvcuda;

#define BM 128
#define BK 32
#define TN 8
#define NTHREADS 256
#define PAD 8

// BK doubled: every thread now owns 2 float4 per matrix per slab, so the loads
// and stores run in PASSES_A / PASSES_B passes with a row stride between them.
#define STRIDE_A (NTHREADS / (BK / 4))
#define STRIDE_B (NTHREADS / (BM / 4))
#define PASSES_A (BM / STRIDE_A)
#define PASSES_B (BK / STRIDE_B)


__global__ void __launch_bounds__(256, 2) tcgemm(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ __align__(32) half shared_a[2][BM][BK + PAD];  
    __shared__ __align__(32) half shared_b[2][BK][BM + PAD];


    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }
        

    const int warpId  = threadIdx.x / 32;   const int lane    = threadIdx.x % 32;
    const int warpRow = warpId / 4;         const int warpCol = warpId % 4;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BM;

    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BM / 4);
    const int innerColB = threadIdx.x % (BM / 4);
    const int cRow = blockIdx.y * BM  + innerRowA;
    const int cCol = blockIdx.x * BM + innerColB * 4;
    const int numTiles = (K + BK - 1) / BK;

    //Load slab 0
    half2 tmpA[PASSES_A][2], tmpB[PASSES_B][2];
    int aCol = innerColA * 4;
    
    for (int p = 0; p < PASSES_A; p++) {
        tmpA[p][0] = tmpA[p][1] = __floats2half2_rn(0.f, 0.f); 
        if (cRow + p * STRIDE_A < M && aCol < K) {
            const float4 v = *reinterpret_cast<const float4*>(&A[(cRow + p * STRIDE_A) * K + aCol]);
            tmpA[p][0] = __floats2half2_rn(v.x, v.y);
            tmpA[p][1] = __floats2half2_rn(v.z, v.w);
        }
        half2* dA = reinterpret_cast<half2*>(&shared_a[0][innerRowA + p * STRIDE_A][innerColA * 4]);
        dA[0] = tmpA[p][0];
        dA[1] = tmpA[p][1];
    }

    int bRow = innerRowB;

    for (int p = 0; p < PASSES_B; p++) {
        tmpB[p][0] = tmpB[p][1] = __floats2half2_rn(0.f, 0.f);
        if (bRow + p * STRIDE_B < K && cCol < N) {
            const float4 v = *reinterpret_cast<const float4*>(&B[(bRow + p * STRIDE_B) * N + cCol]);
            tmpB[p][0] = __floats2half2_rn(v.x, v.y);
            tmpB[p][1] = __floats2half2_rn(v.z, v.w);
        }
        half2* dB = reinterpret_cast<half2*>(&shared_b[0][innerRowB + p * STRIDE_B][innerColB * 4]);
        dB[0] = tmpB[p][0];
        dB[1] = tmpB[p][1];
    }
    __syncthreads();

    for (int i = 0; i < numTiles; i++) {
        const int cur = i % 2;
        const int next = (i + 1) % 2;

        if (i + 1 < numTiles) {
            aCol = (i + 1) * BK + innerColA * 4;
            for (int p = 0; p < PASSES_A; p++) {
                tmpA[p][0] = tmpA[p][1] = __floats2half2_rn(0.f, 0.f); 
                if (cRow + p * STRIDE_A < M && aCol < K) {
                    const float4 v = *reinterpret_cast<const float4*>(&A[(cRow + p * STRIDE_A) * K + aCol]);
                    tmpA[p][0] = __floats2half2_rn(v.x, v.y);
                    tmpA[p][1] = __floats2half2_rn(v.z, v.w);
                }
            }

            bRow = (i + 1) * BK + innerRowB;
            for (int p = 0; p < PASSES_B; p++) {
                tmpB[p][0] = tmpB[p][1] = __floats2half2_rn(0.f, 0.f);
                if (bRow + p * STRIDE_B < K && cCol < N) {
                    const float4 v = *reinterpret_cast<const float4*>(&B[(bRow + p * STRIDE_B) * N + cCol]);
                    tmpB[p][0] = __floats2half2_rn(v.x, v.y);
                    tmpB[p][1] = __floats2half2_rn(v.z, v.w);
                }
            }
        }

        for (int kk = 0; kk < BK; kk += 16) {                                        
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[2];
            for (int i = 0; i < 4; ++i)
                wmma::load_matrix_sync(a[i], &shared_a[cur][warpRow * 64 + i * 16][kk], BK + PAD);
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(b[j], &shared_b[cur][kk][warpCol * 32 + j * 16], BM + PAD);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }


        for (int p = 0; p < PASSES_A; p++) {
            half2* dA = reinterpret_cast<half2*>(&shared_a[next][innerRowA + p * STRIDE_A][innerColA * 4]);
            dA[0] = tmpA[p][0];
            dA[1] = tmpA[p][1];
        }

        for (int p = 0; p < PASSES_B; p++) {
            half2* dB = reinterpret_cast<half2*>(&shared_b[next][innerRowB + p * STRIDE_B][innerColB * 4]);
            dB[0] = tmpB[p][0];
            dB[1] = tmpB[p][1];
        }

        __syncthreads();


    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(&C[(row0 + warpRow * 64 + i * 16) * N + col0 + warpCol * 32 + j * 16],
                                    acc[i][j], N, wmma::mem_row_major);
        }
    }
}