#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

using namespace std;





__global__ void naiveMatMul(float* A, float* B, float* C, int N, int K, int M) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int row = blockIdx.y * blockDim.y + ty;
        int col = blockIdx.x * blockDim.x + tx;

        if (row < N && col < M) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                sum += A[row * K + i] * B[i * M + col];
            }
            C[row * M + col] = sum;
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

    dim3 blockDim(16,16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y );

    naiveMatMul<<<gridDim, blockDim>>>(devA, devB, devC, N, K, M);
    
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

