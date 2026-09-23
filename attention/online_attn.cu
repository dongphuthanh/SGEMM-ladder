#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <float.h>


#define BR 64
#define BC 32
#define TR 4
#define TC 8
#define D 64
__global__ void onlineRowKernel(const float* Q, const float* K, const float* V, float* O, int N) {
    const int blockRow = blockIdx.x;


    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int laneRow = lane / 8;
    const int laneCol = lane % 8;
    const int startRow = warpId * 16 + laneRow * 4;
    const int startCol = laneCol * 8;
    const float scale = rsqrtf((float)D) * 1.44269504f; 

    __shared__ __align__(16) float Qs[BR][D];
    __shared__ __align__(16) float Ks[D][BC];
    __shared__ __align__(16) float Vs[BC][D];
    float (*Ps)[BC] = Ks;

    static_assert(BR == D, "Ps aliases Ks: both must be [64][BC]");

    const int numChunks = (N + BC - 1) / BC;

    //load chunk 0
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            const int qRow = blockRow * BR + startRow + i;
            Qs[startRow + i][startCol + j] = (qRow < N) ? Q[qRow * D + startCol + j] * scale : 0.0f;
        }
    } 

    __syncthreads();

    float m[4];
    for (int i = 0; i < 4; ++i) {
        m[i] = -FLT_MAX;
    }
    float l[4]{};

    float o[4][8]{};
    for (int chunk = 0; chunk < numChunks; ++chunk) {
        float s[4][4]{};

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 8; ++j) {
                const int row = warpId * 8 + laneRow * 2 + i;
                const int key = chunk * BC + row;
                Ks[startCol + j][row] = (key < N) ? K[key * D + startCol + j] : 0.0f;
                Vs[row][startCol + j] = (key < N) ? V[key * D + startCol + j] : 0.0f;
                
            }
        }
        __syncthreads();
        //Calculate S
        for (int h = 0; h < D; h+=4) {
            float q[4][4];
            float k[4][4];
            for (int i = 0; i < 4; ++i) {
                const float4 t4 = *reinterpret_cast<const float4*>(&Qs[startRow + i][h]);
                q[i][0] = t4.x; q[i][1] = t4.y; q[i][2] = t4.z; q[i][3] = t4.w;
            }

            for (int t = 0; t < 4; ++t) {
                const float4 t4 = *reinterpret_cast<const float4*>(&Ks[h + t][laneCol * 4]);
                k[t][0] = t4.x; k[t][1] = t4.y; k[t][2] = t4.z; k[t][3] = t4.w;
            }

            for (int t = 0; t < 4; ++t) {
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < 4; i++) {
                        s[j][i] += q[j][t] * k[t][i];
                    }
                }
            }
        }

        for (int j = 0; j < 4; ++j) {
            const int key = chunk * BC + laneCol * 4 + j;
            if (key >= N)
                for (int i = 0; i < 4; ++i) s[i][j] = -INFINITY;
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
        for (int i = 0; i < 4; ++i) {
            for (int c = 0; c < 8; ++c) {
                o[i][c] *= alpha[i];
            }
            m[i] = m_new[i];
        }

        __syncthreads();

        //Write P to share memory
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                Ps[startRow + i][laneCol * 4 + j] = s[i][j];
            }
        }
        __syncthreads();


        //Calculate O
        for (int h = 0; h < BC; h+=4) {
            float p[4][4];

            for (int i = 0; i < 4; ++i) {
                const float4 t4 = *reinterpret_cast<const float4*>(&Ps[startRow + i][h]);
                p[i][0] = t4.x; p[i][1] = t4.y; p[i][2] = t4.z; p[i][3] = t4.w;
            }

            for (int t = 0; t < 4; ++t) {                 // one V row at a time: keeps registers down
                const float4 va = *reinterpret_cast<const float4*>(&Vs[h + t][laneCol * 4]);
                const float4 vb = *reinterpret_cast<const float4*>(&Vs[h + t][32 + laneCol * 4]);
                const float v[8] = { va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w };
                for (int i = 0; i < 4; ++i) {
                    for (int c = 0; c < 8; ++c) {
                        o[i][c] += p[i][t] * v[c];
                    }
                }
            }
        }

        

        __syncthreads();

    }


    for (int i = 0; i < 4; ++i) {
        const int qRow = blockRow * BR + startRow + i;
        if (qRow >= N) continue;
        const float inv = 1.0f / l[i];
        for (int j = 0; j < 4; ++j) {
            O[blockRow * BR * D + (startRow + i) * D + laneCol * 4 + j] = o[i][j] * inv;
            O[blockRow * BR * D + (startRow + i) * D + laneCol * 4 + j + 32] = o[i][j + 4] * inv;
        }
    }





}


