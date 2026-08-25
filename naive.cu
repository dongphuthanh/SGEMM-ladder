#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>


using namespace std;



#define blockSize 256


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
    float a[6] = {1,3,4,5,6,7};
    float b[9] = {1,2,3,4,5,6,7,8,9};
    float* A = a;
    float* B = b;
    int N = 2;
    int K = 3;
    int M = 3;
    float* C = new float[N * M];
    matMul(A, B, C, N, K, M);
    for (int i = 0; i < N * M; i++) {
        cout << C[i] << " ";
    }
}

