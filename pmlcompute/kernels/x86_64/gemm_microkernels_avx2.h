#ifndef GEMM_MICROKERNELS_AVX2
#define GEMM_MICROKERNELS_AVX2

#include <stddef.h>

extern void __attribute__((sysv_abi))
sgemm_microkernel6x16_avx2(const float* pack_A, const float* pack_B, float* C, const size_t ldc);

extern void __attribute__((sysv_abi))
sgemm_microkernel8x8_avx2(const float* pack_A, const float* pack_B, float* C, const size_t ldc);

#endif
