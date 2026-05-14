#include "gemm_config.h"

#define MR 4
#define NR 16

.macro load_broadcast_fma B_row
    vmovups (NC * 4 * \B_row)(%rsi, %r10, 4), %ymm9
    vmovups (NC * 4 * \B_row + 32)(%rsi, %r10, 4), %ymm10
    vmulps %ymm0, %ymm9, %ymm9
    vmulps %ymm0, %ymm10, %ymm10

    vbroadcastss (%rdi, %rax, 4), %ymm11
    vfmadd231ps %ymm11, %ymm9, %ymm1
    vfmadd231ps %ymm11, %ymm10, %ymm2

    vbroadcastss (KC * 4 * 1)(%rdi, %rax, 4), %ymm11
    vfmadd231ps %ymm11, %ymm9, %ymm3
    vfmadd231ps %ymm11, %ymm10, %ymm4

    vbroadcastss (KC * 4 * 2)(%rdi, %rax, 4), %ymm11
    vfmadd231ps %ymm11, %ymm9, %ymm5
    vfmadd231ps %ymm11, %ymm10, %ymm6

    vbroadcastss (KC * 4 * 3)(%rdi, %rax, 4), %ymm11
    vfmadd231ps %ymm11, %ymm9, %ymm7
    vfmadd231ps %ymm11, %ymm10, %ymm8
.endm

    .section .text
    .globl sgemm_microkernel4x16_avx2
    .type sgemm_microkernel4x16_avx2, @function
# const float* pack_A, const float* pack_B, float* C,
# const size_t ldc, float alpha,
# float beta
sgemm_microkernel4x16_avx2:
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
    vmovups (%r10), %ymm1 # C[ldc * i + j : ldc * i + j + 8]
    vmovups 0x20(%r10), %ymm2
    add (%rsp), %rax # ldc * (i + 1) + j
    lea (%rdx, %rax, 4), %r10
    vmovups (%r10), %ymm3
    vmovups 0x20(%r10), %ymm4
    add (%rsp), %rax # ldc * (i + 2) + j
    lea (%rdx, %rax, 4), %r10
    vmovups (%r10), %ymm5
    vmovups 0x20(%r10), %ymm6
    add (%rsp), %rax # ldc * (i + 3) + j
    lea (%rdx, %rax, 4), %r10
    vmovups (%r10), %ymm7
    vmovups 0x20(%r10), %ymm8

    xor %rcx, %rcx # pack_A col, pack_B row, k
.LKC_loop:
    mov $KC, %rax
    imul %r8, %rax
    add %rcx, %rax # KC * i + k

    mov $NC, %r10
    imul %rcx, %r10
    add %r9, %r10 # NC * k + j

    load_broadcast_fma 0

    inc %rax

    load_broadcast_fma 1

    inc %rax

    load_broadcast_fma 2

    inc %rax

    load_broadcast_fma 3

    inc %rax

    load_broadcast_fma 4

    inc %rax

    load_broadcast_fma 5

    inc %rax

    load_broadcast_fma 6

    inc %rax

    load_broadcast_fma 7

    inc %rax

    load_broadcast_fma 8

    inc %rax

    load_broadcast_fma 9

    inc %rax

    load_broadcast_fma 10

    inc %rax

    load_broadcast_fma 11

    inc %rax

    load_broadcast_fma 12

    inc %rax

    load_broadcast_fma 13

    inc %rax

    load_broadcast_fma 14

    inc %rax

    load_broadcast_fma 15

    add $NR, %rcx
    cmp $KC, %rcx
    jb .LKC_loop
.LKC_loop_end:

    mov (%rsp), %rax # ldc
    imul %r8, %rax # ldc * i
    add %r9, %rax # ldc * i + j
    lea (%rdx, %rax, 4), %r10 # C + ldc * i + j
    vmovups %ymm1, (%r10) # C[ldc * i + j : ldc * i + j + 8]
    vmovups %ymm2, 0x20(%r10)
    add (%rsp), %rax # ldc * (i + 1) + j
    lea (%rdx, %rax, 4), %r10
    vmovups %ymm3, (%r10)
    vmovups %ymm4, 0x20(%r10)
    add (%rsp), %rax # ldc * (i + 2) + j
    lea (%rdx, %rax, 4), %r10
    vmovups %ymm5, (%r10)
    vmovups %ymm6, 0x20(%r10)
    add (%rsp), %rax # ldc * (i + 3) + j
    lea (%rdx, %rax, 4), %r10
    vmovups %ymm7, (%r10)
    vmovups %ymm8, 0x20(%r10)

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
    .size sgemm_microkernel4x16_avx2, . - sgemm_microkernel4x16_avx2
