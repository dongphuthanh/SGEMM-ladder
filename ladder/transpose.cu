#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>


using namespace std;

#define BM 128
#define BK 8
#define TN 8


__global__ void vectorizeTranspose(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float shared_a[BK][BM];
    __shared__ float shared_b[BK][BM];

    const int BDIM = BM / TN;
    float c[TN][TN]{};

    const int tx = threadIdx.x % (BDIM);
    const int ty = threadIdx.x / (BDIM);
    
    

    const int numTiles = (K + BK - 1) / BK;

    for (int i = 0; i < numTiles; i++) {
        const int innerRowA = threadIdx.x / (BK / 4);    
        const int innerColA = threadIdx.x % (BK / 4);    
        const int innerRowB = threadIdx.x / (BM / 4);    
        const int innerColB = threadIdx.x % (BM / 4);
        const int cRow = blockIdx.y * BM  + innerRowA;
        const int aCol = i * BK + innerColA * 4;
        float4 tmp = make_float4(0.f, 0.f, 0.f, 0.f);
        if (cRow < M && aCol < K) {
            tmp = *reinterpret_cast<const float4*>(&A[cRow * K + aCol]);
        }
        shared_a[innerColA * 4 + 0][innerRowA] = tmp.x;
        shared_a[innerColA * 4 + 1][innerRowA] = tmp.y;
        shared_a[innerColA * 4 + 2][innerRowA] = tmp.z;
        shared_a[innerColA * 4 + 3][innerRowA] = tmp.w;
        
        const int cCol = blockIdx.x * BM + innerColB * 4;
        const int bRow = i * BK + innerRowB;
        
        tmp = make_float4(0.f, 0.f, 0.f, 0.f);
        if (bRow < K && cCol < N) {
            tmp = *reinterpret_cast<const float4*>(&B[bRow * N + cCol]);
        }
        *reinterpret_cast<float4*>(&shared_b[innerRowB][innerColB * 4]) = tmp;

        __syncthreads();

        for (int h = 0; h < BK; h++) {
            float a[TN];
            float b[TN];
            for (int j = 0; j < TN; j++) {
                a[j] = shared_a[h][ty * TN + j];
                b[j] = shared_b[h][tx * TN + j];
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

    vectorizeTranspose<<<gridDim, blockDim>>>(devA, devB, devC, N, K, M);
    
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