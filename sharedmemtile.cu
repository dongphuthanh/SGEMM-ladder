#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>


using namespace std;
#define TILE_SIZE 16

__global__ void sharedTileMatMul(float* A, float* B, float* C, int N, int K, int M) {
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float acc = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int i = 0; i < numTiles; i++) {
        int aCol = i * TILE_SIZE + tx;
        int bRow = i * TILE_SIZE + ty;

        if (row < N && aCol < K) {
            shared_a[ty][tx] = A[row * K + aCol];
        } 
        else {
            shared_a[ty][tx] = 0.0f;
        }

        if (bRow < K && col < M) {
            shared_b[ty][tx] = B[bRow * M + col];
        }
        else {
            shared_b[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++) {
            acc += shared_a[ty][j] * shared_b[j][tx];
        }
        __syncthreads();
    }

    if (row < N && col < M) {
        C[row * M + col] = acc;
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

    sharedTileMatMul<<<gridDim, blockDim>>>(devA, devB, devC, N, K, M);
    
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