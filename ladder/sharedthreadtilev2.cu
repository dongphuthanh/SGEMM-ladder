#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>


using namespace std;

#define BM 128
#define BK 8
#define TN 8


__global__ void sharedThreadTilev2(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float shared_a[BM][BK];
    __shared__ float shared_b[BK][BM];
    constexpr int NTHREADS = (BM / TN) * (BM / TN); 
    const int BDIM = BM / TN;
    float c[TN][TN]{};

    const int tx = threadIdx.x % (BDIM);
    const int ty = threadIdx.x / (BDIM);
    constexpr int strideA = NTHREADS / BK;
    constexpr int strideB = NTHREADS / BM; 
    
    

    const int numTiles = (K + BK - 1) / BK;

    for (int i = 0; i < numTiles; i++) {
        const int innerRowA = threadIdx.x / BK;    
        const int innerColA = threadIdx.x % BK;    
        const int innerRowB = threadIdx.x / BM;    
        const int innerColB = threadIdx.x % BM;
        for (int j = 0; j < 4; j++) {
            int row = innerRowA + j * strideA;
            int cRow = blockIdx.y * BM + row;
            int aCol = i * BK + innerColA;
            if (cRow < M && aCol < K) {
                    shared_a[row][innerColA] = A[cRow * K + aCol];
            }
            else {
                    shared_a[row][innerColA] = 0.0f;
            }
        }

        for (int k = 0; k < 4; k++) {
            int row = innerRowB + k * strideB;
            int cCol = blockIdx.x * BM + innerColB;
            int bRow = i * BK + row;
            if (bRow < K && cCol < N) {
                shared_b[row][innerColB] = B[bRow * N + cCol];
            }
            else {
                shared_b[row][innerColB] = 0.0f;
            }
        }
        __syncthreads();

        for (int h = 0; h < BK; h++) {
            float a[TN];
            float b[TN];
            for (int j = 0; j < TN; j++) {
                a[j] = shared_a[ty * TN + j][h];
            }

            for (int k = 0; k < TN; k++) {
                b[k] = shared_b[h][tx * TN + k];
            }

            for (int j = 0; j < TN; j++) {
                for (int k = 0; k < TN; k++) {
                    c[j][k] += a[j] * b[k];
                }
            }
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

    sharedThreadTilev2<<<gridDim, blockDim>>>(devA, devB, devC, N, K, M);
    
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