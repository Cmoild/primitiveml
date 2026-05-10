#include "gemm_config.h"

    .section .text
    .globl sgemm_microkernel_avx2
    .type sgemm_microkernel_avx2, @function
# const float* pack_A, const float* pack_B, float* C,
# const size_t ldc, float alpha,
# float beta
sgemm_microkernel_avx2:
    push %rcx
    vbroadcastss %xmm0, %ymm0 # alpha

    xor %r8, %r8 # pack_A, C row, i
.LMR_loop:

    xor %r9, %r9 # pack_B, C col, j
.LNR_loop:

    mov (%rsp), %rax # ldc
    imul %r8, %rax # ldc * i
    add %r9, %rax # ldc * i + j
    lea (%rdx, %rax, 4), %r10 # C + ldc * i + j
    vmovups (%r10), %ymm2 # C[ldc * i + j : ldc * i + j + 8]
    add (%rsp), %rax # ldc * (i + 1) + j
    lea (%rdx, %rax, 4), %r10
    vmovups (%r10), %ymm3
    add (%rsp), %rax # ldc * (i + 2) + j
    lea (%rdx, %rax, 4), %r10
    vmovups (%r10), %ymm4
    add (%rsp), %rax # ldc * (i + 3) + j
    lea (%rdx, %rax, 4), %r10
    vmovups (%r10), %ymm5

    xor %rcx, %rcx # pack_A col, pack_B row, k
.LKC_loop:
    mov $KC, %rax
    imul %r8, %rax
    add %rcx, %rax # KC * i + k

    mov $NC, %r10
    imul %rcx, %r10
    add %r9, %r10 # NC * k + j

    vbroadcastss (%rdi, %rax, 4), %ymm6 # pack_A[KC * i + k]
    vbroadcastss (KC * 4 * 1)(%rdi, %rax, 4), %ymm7 # pack_A[KC * (i + 1) + k]
    vbroadcastss (KC * 4 * 2)(%rdi, %rax, 4), %ymm8 # pack_A[KC * (i + 2) + k]
    vbroadcastss (KC * 4 * 3)(%rdi, %rax, 4), %ymm9 # pack_A[KC * (i + 3) + k]

    vmovups (%rsi, %r10, 4), %ymm10 # pack_B[NC * k + j : NC * k + j + 8]
    vmulps %ymm0, %ymm10, %ymm10 # alpha * pack_B[...]

    # C[ldc * i + j : ldc * i + j + 8] += pack_A[KC * i + k] * pack_B[NC * k + j : NC * k + j + 8]
    vfmadd231ps %ymm6, %ymm10, %ymm2
    # C[ldc * (i + 1) + j : ldc * (i + 1) + j + 8] += pack_A[KC * (i + 1) + k] * pack_B[NC * k + j : NC * k + j + 8]
    vfmadd231ps %ymm7, %ymm10, %ymm3
    vfmadd231ps %ymm8, %ymm10, %ymm4
    vfmadd231ps %ymm9, %ymm10, %ymm5

    inc %rax

    vbroadcastss (%rdi, %rax, 4), %ymm6 # pack_A[KC * i + k + 1]
    vbroadcastss (KC * 4 * 1)(%rdi, %rax, 4), %ymm7 # pack_A[KC * (i + 1) + k + 1]
    vbroadcastss (KC * 4 * 2)(%rdi, %rax, 4), %ymm8 # pack_A[KC * (i + 2) + k + 1]
    vbroadcastss (KC * 4 * 3)(%rdi, %rax, 4), %ymm9 # pack_A[KC * (i + 3) + k + 1]

    vmovups (NC * 4 * 1)(%rsi, %r10, 4), %ymm10 # pack_B[NC * (k + 1) + j : NC * (k + 1) + j + 8]
    vmulps %ymm0, %ymm10, %ymm10 # alpha * pack_B[...]

    vfmadd231ps %ymm6, %ymm10, %ymm2
    vfmadd231ps %ymm7, %ymm10, %ymm3
    vfmadd231ps %ymm8, %ymm10, %ymm4
    vfmadd231ps %ymm9, %ymm10, %ymm5

    inc %rax

    vbroadcastss (%rdi, %rax, 4), %ymm6
    vbroadcastss (KC * 4 * 1)(%rdi, %rax, 4), %ymm7
    vbroadcastss (KC * 4 * 2)(%rdi, %rax, 4), %ymm8
    vbroadcastss (KC * 4 * 3)(%rdi, %rax, 4), %ymm9

    vmovups (NC * 4 * 2)(%rsi, %r10, 4), %ymm10
    vmulps %ymm0, %ymm10, %ymm10

    vfmadd231ps %ymm6, %ymm10, %ymm2
    vfmadd231ps %ymm7, %ymm10, %ymm3
    vfmadd231ps %ymm8, %ymm10, %ymm4
    vfmadd231ps %ymm9, %ymm10, %ymm5

    inc %rax

    vbroadcastss (%rdi, %rax, 4), %ymm6
    vbroadcastss (KC * 4 * 1)(%rdi, %rax, 4), %ymm7
    vbroadcastss (KC * 4 * 2)(%rdi, %rax, 4), %ymm8
    vbroadcastss (KC * 4 * 3)(%rdi, %rax, 4), %ymm9

    vmovups (NC * 4 * 3)(%rsi, %r10, 4), %ymm10
    vmulps %ymm0, %ymm10, %ymm10

    vfmadd231ps %ymm6, %ymm10, %ymm2
    vfmadd231ps %ymm7, %ymm10, %ymm3
    vfmadd231ps %ymm8, %ymm10, %ymm4
    vfmadd231ps %ymm9, %ymm10, %ymm5

    inc %rax

    vbroadcastss (%rdi, %rax, 4), %ymm6
    vbroadcastss (KC * 4 * 1)(%rdi, %rax, 4), %ymm7
    vbroadcastss (KC * 4 * 2)(%rdi, %rax, 4), %ymm8
    vbroadcastss (KC * 4 * 3)(%rdi, %rax, 4), %ymm9

    vmovups (NC * 4 * 4)(%rsi, %r10, 4), %ymm10
    vmulps %ymm0, %ymm10, %ymm10

    vfmadd231ps %ymm6, %ymm10, %ymm2
    vfmadd231ps %ymm7, %ymm10, %ymm3
    vfmadd231ps %ymm8, %ymm10, %ymm4
    vfmadd231ps %ymm9, %ymm10, %ymm5

    inc %rax

    vbroadcastss (%rdi, %rax, 4), %ymm6
    vbroadcastss (KC * 4 * 1)(%rdi, %rax, 4), %ymm7
    vbroadcastss (KC * 4 * 2)(%rdi, %rax, 4), %ymm8
    vbroadcastss (KC * 4 * 3)(%rdi, %rax, 4), %ymm9

    vmovups (NC * 4 * 5)(%rsi, %r10, 4), %ymm10
    vmulps %ymm0, %ymm10, %ymm10

    vfmadd231ps %ymm6, %ymm10, %ymm2
    vfmadd231ps %ymm7, %ymm10, %ymm3
    vfmadd231ps %ymm8, %ymm10, %ymm4
    vfmadd231ps %ymm9, %ymm10, %ymm5

    inc %rax

    vbroadcastss (%rdi, %rax, 4), %ymm6
    vbroadcastss (KC * 4 * 1)(%rdi, %rax, 4), %ymm7
    vbroadcastss (KC * 4 * 2)(%rdi, %rax, 4), %ymm8
    vbroadcastss (KC * 4 * 3)(%rdi, %rax, 4), %ymm9

    vmovups (NC * 4 * 6)(%rsi, %r10, 4), %ymm10
    vmulps %ymm0, %ymm10, %ymm10

    vfmadd231ps %ymm6, %ymm10, %ymm2
    vfmadd231ps %ymm7, %ymm10, %ymm3
    vfmadd231ps %ymm8, %ymm10, %ymm4
    vfmadd231ps %ymm9, %ymm10, %ymm5

    inc %rax

    vbroadcastss (%rdi, %rax, 4), %ymm6
    vbroadcastss (KC * 4 * 1)(%rdi, %rax, 4), %ymm7
    vbroadcastss (KC * 4 * 2)(%rdi, %rax, 4), %ymm8
    vbroadcastss (KC * 4 * 3)(%rdi, %rax, 4), %ymm9

    vmovups (NC * 4 * 7)(%rsi, %r10, 4), %ymm10
    vmulps %ymm0, %ymm10, %ymm10

    vfmadd231ps %ymm6, %ymm10, %ymm2
    vfmadd231ps %ymm7, %ymm10, %ymm3
    vfmadd231ps %ymm8, %ymm10, %ymm4
    vfmadd231ps %ymm9, %ymm10, %ymm5

    add $NR, %rcx
    cmp $KC, %rcx
    jb .LKC_loop
.LKC_loop_end:

    mov (%rsp), %rax # ldc
    imul %r8, %rax # ldc * i
    add %r9, %rax # ldc * i + j
    lea (%rdx, %rax, 4), %r10 # C + ldc * i + j
    vmovups %ymm2, (%r10) # C[ldc * i + j : ldc * i + j + 8]
    add (%rsp), %rax # ldc * (i + 1) + j
    lea (%rdx, %rax, 4), %r10
    vmovups %ymm3, (%r10)
    add (%rsp), %rax # ldc * (i + 2) + j
    lea (%rdx, %rax, 4), %r10
    vmovups %ymm4, (%r10)
    add (%rsp), %rax # ldc * (i + 3) + j
    lea (%rdx, %rax, 4), %r10
    vmovups %ymm5, (%r10)

    add $NR, %r9
    cmp $NC, %r9
    jb .LNR_loop
.LNR_loop_end:
    add $MR, %r8
    cmp $MC, %r8
    jb .LMR_loop
.LMR_loop_end:
    add $8, %rsp
    vzeroupper
    ret
    .size sgemm_microkernel_avx2, . - sgemm_microkernel_avx2
