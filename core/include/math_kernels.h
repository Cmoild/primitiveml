#ifndef MATH_KERNELS_H
#define MATH_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void exp_avx2_aligned(float* operand, float* result, size_t n);

void exp_avx2_unaligned(float* operand, float* result, size_t n);

#ifdef __cplusplus
}
#endif

#endif
