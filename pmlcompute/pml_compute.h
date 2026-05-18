#ifndef MATH_KERNELS_H
#define MATH_KERNELS_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void __attribute__((sysv_abi)) exp_avx2(const float* operand, float* result, size_t n);

void exp_avx2_intrin(float* operand, float* result, size_t n);

void pml_sgemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K,
               const bool a_transposed, const bool b_transposed, const float alpha,
               const float beta, const size_t lda, const size_t ldb, const size_t ldc);

#ifdef __cplusplus
}
#endif

#endif
