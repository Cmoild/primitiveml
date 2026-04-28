#ifndef MATH_KERNELS_H
#define MATH_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void exp_avx2(float* operand, float* result, size_t n);

// void exp_avx2_intrin(float* operand, float* result, size_t n);

#ifdef __cplusplus
}
#endif

void exp_avx2_intrin(float* operand, float* result, size_t n);

#endif
