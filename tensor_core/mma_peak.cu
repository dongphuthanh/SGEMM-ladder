// mma_peak.cu -- peak math throughput per number format on this GPU.
//
// No memory traffic at all: every warp issues back-to-back multiply-adds on
// constant operands, with 8-16 independent accumulators so latency is hidden.
// This is the ceiling a kernel could reach, per format:
//   FP32 FFMA on CUDA cores                 (what the SGEMM ladder uses)
//   TF32 tensor cores: wmma m16n16k8 and mma.sync m16n8k8 (PTX)
//   FP16 in / FP32 accumulate: mma.sync m16n8k16   (what tc_gemm, tc_attn use)
//   FP8 e4m3 in / FP32 accumulate: mma.sync m16n8k32
//
//   make peak      (or: nvcc -O2 -arch=sm_120 tensor_core/mma_peak.cu && ./a.out)
#include <cstdio>
#include <mma.h>
using namespace nvcuda;

__global__ void wmmaPeak(float* out, int iters) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c[8];
    wmma::fill_fragment(a, 0.001f); wmma::fill_fragment(b, 0.001f);
    for (int i = 0; i < 8; ++i) wmma::fill_fragment(c[i], 0.f);
    for (int it = 0; it < iters; ++it)
        for (int i = 0; i < 8; ++i) wmma::mma_sync(c[i], a, b, c[i]);
    float s = 0; for (int i = 0; i < 8; ++i) s += c[i].x[0];
    if (s == 12345.f) out[0] = s;
}

__global__ void mmaPeak(float* out, int iters) {
    unsigned a[4], b[2];
    for (int i = 0; i < 4; ++i) a[i] = __float_as_uint(0.001f);
    for (int i = 0; i < 2; ++i) b[i] = __float_as_uint(0.001f);
    float c[16][4];
    for (int i = 0; i < 16; ++i) for (int j = 0; j < 4; ++j) c[i][j] = 0.f;
    for (int it = 0; it < iters; ++it)
        for (int i = 0; i < 16; ++i)
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    float s = 0; for (int i = 0; i < 16; ++i) s += c[i][0];
    if (s == 12345.f) out[0] = s;
}

__global__ void mmaF16Peak(float* out, int iters) {
    unsigned a[4], b[2];
    for (int i = 0; i < 4; ++i) a[i] = 0x1c001c00u;   // two halves of ~0.004
    for (int i = 0; i < 2; ++i) b[i] = 0x1c001c00u;
    float c[16][4];
    for (int i = 0; i < 16; ++i) for (int j = 0; j < 4; ++j) c[i][j] = 0.f;
    for (int it = 0; it < iters; ++it)
        for (int i = 0; i < 16; ++i)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    float s = 0; for (int i = 0; i < 16; ++i) s += c[i][0];
    if (s == 12345.f) out[0] = s;
}

__global__ void mmaF8Peak(float* out, int iters) {
    unsigned a[4], b[2];
    for (int i = 0; i < 4; ++i) a[i] = 0x20202020u;   // four e4m3 values each
    for (int i = 0; i < 2; ++i) b[i] = 0x20202020u;
    float c[16][4];
    for (int i = 0; i < 16; ++i) for (int j = 0; j < 4; ++j) c[i][j] = 0.f;
    for (int it = 0; it < iters; ++it)
        for (int i = 0; i < 16; ++i)
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    float s = 0; for (int i = 0; i < 16; ++i) s += c[i][0];
    if (s == 12345.f) out[0] = s;
}

__global__ void ffmaPeak(float* out, int iters) {
    float c[16]; for (int i = 0; i < 16; ++i) c[i] = threadIdx.x;
    const float a = 1.0001f, b = 0.0001f;
    for (int it = 0; it < iters; ++it)
        for (int k = 0; k < 16; ++k)          // 16 FMA x 16 chains = 256 FMA per iteration
            for (int i = 0; i < 16; ++i) c[i] = fmaf(c[i], a, b);
    float s = 0; for (int i = 0; i < 16; ++i) s += c[i];
    if (s == 12345.f) out[0] = s;
}

int main() {
    float* out; cudaMalloc(&out, 4);
    const int blocks = 36 * 8, threads = 256, iters = 2000;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto time = [&](int v) {
        float best = 1e9;
        for (int r = 0; r < 7; ++r) {
            cudaEventRecord(e0);
            if (v == 0) wmmaPeak<<<blocks, threads>>>(out, iters);
            if (v == 1) mmaPeak<<<blocks, threads>>>(out, iters);
            if (v == 2) ffmaPeak<<<blocks, threads>>>(out, iters);
            if (v == 3) mmaF16Peak<<<blocks, threads>>>(out, iters);
            if (v == 4) mmaF8Peak<<<blocks, threads>>>(out, iters);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); if (r > 1 && ms < best) best = ms;
        }
        return best;
    };
    for (int w = 0; w < 30; ++w) { time(0); time(1); time(2); time(3); time(4); }       // settle clocks
    const double warps = blocks * threads / 32.0;
    double tw = time(0), tm = time(1), tf = time(2), th = time(3), t8 = time(4);
    double macW = warps * iters * 8.0 * 16 * 16 * 8;                  // 8 wmma 16x16x8 per iter
    double macM = warps * iters * 16.0 * 16 * 8 * 8;                  // 16 mma 16x8x8 per iter
    double macF = blocks * (double)threads * iters * 256.0;           // 256 FMA per thread per iter
    printf("wmma m16n16k8 tf32 (tc_gemm's path):  %6.2f TFLOP/s\n", 2 * macW / (tw / 1e3) / 1e12);
    printf("mma.sync m16n8k8 tf32 (PTX):          %6.2f TFLOP/s\n", 2 * macM / (tm / 1e3) / 1e12);
    printf("mma.sync m16n8k16 fp16 in, fp32 acc:  %6.2f TFLOP/s\n", 2 * (macM * 2) / (th / 1e3) / 1e12);
    printf("mma.sync m16n8k32 fp8 e4m3, fp32 acc: %6.2f TFLOP/s\n", 2 * (macM * 4) / (t8 / 1e3) / 1e12);
    printf("FP32 FFMA (SIMT):                     %6.2f TFLOP/s\n", 2 * macF / (tf / 1e3) / 1e12);
    printf("%s\n", cudaGetErrorString(cudaDeviceSynchronize()));
}
