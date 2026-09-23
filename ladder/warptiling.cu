#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>


using namespace std;

#define BM 128
#define BK 16
#define TN 8
#define NTHREADS 256

// BK doubled: every thread now owns 2 float4 per matrix per slab, so the loads
// and stores run in PASSES_A / PASSES_B passes with a row stride between them.
#define STRIDE_A (NTHREADS / (BK / 4))   // 64 rows of As per pass
#define STRIDE_B (NTHREADS / (BM / 4))   //  8 rows of Bs per pass
#define PASSES_A (BM / STRIDE_A)         // 2
#define PASSES_B (BK / STRIDE_B)         // 2


__global__ void warpTile(float* A, float* B, float* C, int M, int K, int N) {
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



void matMul(float* A, float* B, float* C, int N, int K, int M){
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;
    cudaMalloc(&devA, N * K * sizeof(float));
    cudaMalloc(&devB, M * K * sizeof(float));
    cudaMalloc(&devC, N * M * sizeof(float));

    cudaMemcpy(devA, A, N * K * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, M * K * sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, N * M * sizeof(float));

    dim3 blockDim(256);
    dim3 gridDim(((M +  BM - 1) / BM), ((N + BM - 1) / BM) );

    warpTile<<<gridDim, blockDim>>>(devA, devB, devC, N, K, M);

    cudaMemcpy(C, devC, N * M * sizeof(float), cudaMemcpyDefault);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

int main() {
    // Matrix dimensions:
    // A = N x K
    // B = K x M
    // C = N x M
    int N = 4096;
    int K = 4096;
    int M = 4096;

    // Allocate host memory
    vector<float> A(N * K);
    vector<float> B(K * M);
    vector<float> C(N * M);

    // Random initialization
    mt19937 gen(42);
    uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (float& x : A) {
        x = dist(gen);
    }

    for (float& x : B) {
        x = dist(gen);
    }

    cout << "Matrix sizes:\n";
    cout << "A: " << N << " x " << K << "\n";
    cout << "B: " << K << " x " << M << "\n";
    cout << "C: " << N << " x " << M << "\n";

    // Warmup
    matMul(A.data(), B.data(), C.data(), N, K, M);

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 10;

    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++) {
        matMul(A.data(), B.data(), C.data(), N, K, M);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);

    float avg_ms = total_ms / iterations;

    cout << "\nAverage GPU time: "
         << avg_ms << " ms\n";

    // Print a few values so we know something was computed
    cout << "C[0] = " << C[0] << "\n";
    cout << "C[1] = " << C[1] << "\n";
    cout << "C[last] = " << C[N * M - 1] << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
