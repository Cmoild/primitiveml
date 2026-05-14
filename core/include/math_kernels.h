#ifndef MATH_KERNELS_H
#define MATH_KERNELS_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void exp_avx2(const float* operand, float* result, size_t n);

void exp_avx2_intrin(float* operand, float* result, size_t n);

void gemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K,
          const bool a_transposed, const bool b_transposed, const float alpha, const float beta,
          const size_t lda, const size_t ldb, const size_t ldc);

void sgemm_microkernel_avx2(const float* pack_A, const float* pack_B, float* C, const size_t ldc,
                            float alpha);

void sgemm_microkernel6x16_avx2(const float* pack_A, const float* pack_B, float* C,
                                const size_t ldc);

#ifdef __cplusplus
}
#endif

#endif
