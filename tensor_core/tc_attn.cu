#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <float.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define BR 64
#define BC 32
#define TR 4
#define TC 8
#define D 64

#define PAD_H 8
#define PAD_F 4

__global__ void TcAttn(const float* Q, const float* K, const float* V, float* O, int N) {
    const int blockRow = blockIdx.x;


    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int laneRow = lane / 8;
    const int laneCol = lane % 8;
    const int startRow = warpId * 16 + laneRow * 4;
    const int startCol = laneCol * 8;
    const float scale = rsqrtf((float)D) * 1.44269504f; 

    __shared__ __align__(32) half  Qs[BR][D + PAD_H];
    __shared__ __align__(32) half  Ks[BC][D + PAD_H];
    __shared__ __align__(32) half  Vs[BC][D + PAD_H];
    __shared__ __align__(32) float Ss[BR][BC + PAD_F];
    __shared__ __align__(32) float Os[BR][D + PAD_F];
    half (*Ps)[BC + PAD_H] = reinterpret_cast<half (*)[BC + PAD_H]>(Ss);


    static_assert(BR == D, "Ps aliases Ks: both must be [64][BC]");

    const int numChunks = (N + BC - 1) / BC;

    //load chunk 0
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            const int qRow = blockRow * BR + startRow + i;
            Qs[startRow + i][startCol + j] = __float2half_rn((qRow < N) ? Q[qRow * D + startCol + j] * scale : 0.0f);
            Os[startRow + i][startCol + j] = 0.0f;
        }
    } 

    __syncthreads();


    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qa[D / 16];
    for (int k = 0; k < D / 16; ++k) {
        wmma::load_matrix_sync(qa[k], &Qs[warpId * 16][k * 16], D + PAD_H);
    }

    float m[4];
    for (int i = 0; i < 4; ++i) {
        m[i] = -FLT_MAX;
    }
    float l[4]{};

    for (int chunk = 0; chunk < numChunks; ++chunk) {
        float s[4][4]{};

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 8; ++j) {
                const int row = warpId * 8 + laneRow * 2 + i;
                const int key = chunk * BC + row;
                Ks[row][startCol + j] = __float2half_rn((key < N) ? K[key * D + startCol + j] : 0.0f);
                Vs[row][startCol + j] = __float2half_rn((key < N) ? V[key * D + startCol + j] : 0.0f);
                
            }
        }
        __syncthreads();
        //Calculate S
        for (int n = 0; n < BC / 16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf;
            wmma::fill_fragment(sf, 0.0f);
            for (int k = 0; k < D / 16; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb;
                wmma::load_matrix_sync(kb, &Ks[n * 16][k * 16], D + PAD_H); 
                wmma::mma_sync(sf, qa[k], kb, sf);
            }
            wmma::store_matrix_sync(&Ss[warpId * 16][n * 16], sf, BC + PAD_F, wmma::mem_row_major);
        }
        __syncwarp();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                s[i][j] = Ss[startRow + i][laneCol * 4 + j];
            }
        }



        for (int j = 0; j < 4; ++j) {
            const int key = chunk * BC + laneCol * 4 + j;
            if (key >= N) {
                for (int i = 0; i < 4; ++i) s[i][j] = -INFINITY;
            }
        }


        //Calculate the max
        float rmax[4];
        for (int i = 0; i < 4; ++i) {
            rmax[i] = s[i][0];
            for (int j = 1; j < 4; ++j)
                rmax[i] = fmaxf(rmax[i], s[i][j]);
        }

        for (int i = 0; i < 4; ++i) {
            for (int off = 4; off > 0; off >>= 1) {
                rmax[i] = fmaxf(rmax[i], __shfl_xor_sync(0xffffffff, rmax[i], off));
            }   
        }

        float m_new[4];
        float alpha[4];
        for (int i = 0; i < 4; ++i) {
            m_new[i] = fmaxf(m[i], rmax[i]);
            alpha[i] = exp2f(m[i] - m_new[i]);
        }

        //S -> P
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                s[i][j] = exp2f(s[i][j] - m_new[i]);
            }
        }

        //calculate the sum
        float rsum[4]{};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j)
                rsum[i] += s[i][j];
        }

        for (int i = 0; i < 4; ++i) {
            for (int off = 4; off > 0; off >>= 1) {
                rsum[i] += __shfl_xor_sync(0xffffffff, rsum[i], off);
            }   
        }

        //Update l
        for (int i = 0; i < 4; ++i) {
            l[i] = alpha[i] * l[i] + rsum[i];
        }


        //update O
         //update O
        for (int i = 0; i < 4; ++i) {
            for (int c = 0; c < 4; ++c) {
                Os[startRow + i][laneCol * 4 + c]      *= alpha[i];
                Os[startRow + i][32 + laneCol * 4 + c] *= alpha[i];
            }
            m[i] = m_new[i];
        }

        __syncthreads();

        //Write P to share memory
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                Ps[startRow + i][laneCol * 4 + j] = __float2half_rn(s[i][j]);
            }
        }
        __syncthreads();


        //Calculate O
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa[BC / 16];
        for (int k = 0; k < BC / 16; ++k)
            wmma::load_matrix_sync(pa[k], &Ps[warpId * 16][k * 16], BC + PAD_H);
        for (int n = 0; n < D / 16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> of;
            wmma::load_matrix_sync(of, &Os[warpId * 16][n * 16], D + PAD_F, wmma::mem_row_major);
            for (int k = 0; k < BC / 16; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb;
                wmma::load_matrix_sync(vb, &Vs[k * 16][n * 16], D + PAD_H);
                wmma::mma_sync(of, pa[k], vb, of);
            }
            wmma::store_matrix_sync(&Os[warpId * 16][n * 16], of, D + PAD_F, wmma::mem_row_major);
        }

        

        __syncthreads();

    }


    for (int i = 0; i < 4; ++i) {
        const int qRow = blockRow * BR + startRow + i;
        if (qRow >= N) continue;
        const float inv = 1.0f / l[i];
        for (int j = 0; j < 4; ++j) {
            O[blockRow * BR * D + (startRow + i) * D + laneCol * 4 + j] = Os[startRow + i][laneCol * 4 + j] * inv;
            O[blockRow * BR * D + (startRow + i) * D + laneCol * 4 + j + 32] = Os[startRow + i][32 + laneCol * 4 + j] * inv;
        }
    }





}