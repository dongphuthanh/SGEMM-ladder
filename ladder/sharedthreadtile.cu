#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>


using namespace std;
#define TILE_SIZE 64

__global__ void sharedThreadTile(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE];
    float c[4][4]{};

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int i = 0; i < numTiles; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                int row = blockIdx.y * blockDim.y * 4
                    + ty * 4 + j;
                
                int col = blockIdx.x * blockDim.x * 4
                        + tx * 4 + k;
                int aCol = i * TILE_SIZE + tx * 4 + k;
                int bRow = i * TILE_SIZE + ty * 4 + j;

                if (row < M && aCol < K) {
                    shared_a[ty * 4 + j][tx * 4 + k] = A[row * K + aCol];
                } 
                else {
                    shared_a[ty * 4 + j][tx * 4 + k] = 0.0f;
                }

                if (bRow < K && col < N) {
                    shared_b[ty * 4 + j][tx * 4 + k] = B[bRow * N + col];
                }
                else {
                    shared_b[ty * 4 + j][tx * 4 + k] = 0.0f;
                }

            }
        }
        __syncthreads();

        for (int h = 0; h < TILE_SIZE; h++) {
            float a[4];
            float b[4];
            for (int j = 0; j < 4; j++) {

                int row = ty * 4 + j;
                
                a[j] = shared_a[row][h];
            }

            for (int k = 0; k < 4; k++) {
                int col = tx * 4 + k;
                b[k] = shared_b[h][col];
            }

            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    c[j][k] += a[j] * b[k];
                }
            }
        }
        __syncthreads();

    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int row = blockIdx.y * blockDim.y * 4
                    + threadIdx.y * 4 + i;

            int col = blockIdx.x * blockDim.x * 4
                    + threadIdx.x * 4 + j;

            if (row < M && col < N) {
                C[row * N + col] = c[i][j];
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

    dim3 blockDim(16,16);
    dim3 gridDim((M + blockDim.x * 4 - 1) / (blockDim.x * 4), (N + blockDim.y * 4 - 1) / (blockDim.y * 4) );

    sharedThreadTile<<<gridDim, blockDim>>>(devA, devB, devC, N, K, M);
    
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