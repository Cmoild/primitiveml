#include "gemm_config.h"

#define MR 8
#define NR 8

.macro load_broadcast_fma k
    vmovaps (NC * 4 * \k)(%r10), %ymm9

    vbroadcastss (KC * 4 * 0 + 4 * \k)(%rax), %ymm11
    vbroadcastss (KC * 4 * 1 + 4 * \k)(%rax), %ymm0

    vfmadd231ps %ymm11, %ymm9, %ymm1

    vfmadd231ps %ymm0, %ymm9, %ymm2

    vbroadcastss (KC * 4 * 2 + 4 * \k)(%rax), %ymm11
    vbroadcastss (KC * 4 * 3 + 4 * \k)(%rax), %ymm0

    vfmadd231ps %ymm11, %ymm9, %ymm3

    vfmadd231ps %ymm0, %ymm9, %ymm4

    vbroadcastss (KC * 4 * 4 + 4 * \k)(%rax), %ymm11
    vbroadcastss (KC * 4 * 5 + 4 * \k)(%rax), %ymm0

    vfmadd231ps %ymm11, %ymm9, %ymm5

    vfmadd231ps %ymm0, %ymm9, %ymm6

    vbroadcastss (KC * 4 * 6 + 4 * \k)(%rax), %ymm11
    vbroadcastss (KC * 4 * 7 + 4 * \k)(%rax), %ymm0

    vfmadd231ps %ymm11, %ymm9, %ymm7

    vfmadd231ps %ymm0, %ymm9, %ymm8
.endm

    .section .text
    .globl sgemm_microkernel8x8_avx2
#ifdef __ELF__
    .type sgemm_microkernel8x8_avx2, @function
#endif
# const float* pack_A, const float* pack_B, float* C,
# const size_t ldc
sgemm_microkernel8x8_avx2:
    mov %rcx, %r11
    shl $2, %r11

    xor %r8, %r8 # pack_A, C row, i
.LMR_loop:

    xor %r9, %r9 # pack_B, C col, j
.LNR_loop:

    mov %r11, %rax # ldc
    shr $2, %rax
    imul %r8, %rax # ldc * i
    add %r9, %rax # ldc * i + j
    lea (%rdx, %rax, 4), %r10 # C + ldc * i + j
    vmovups (%r10), %ymm1 # C[ldc * i + j : ldc * i + j + 8]
    add %r11, %r10 # ldc * (i + 1) + j
    vmovups (%r10), %ymm2
    add %r11, %r10 # ldc * (i + 2) + j
    vmovups (%r10), %ymm3
    add %r11, %r10 # ldc * (i + 3) + j
    vmovups (%r10), %ymm4
    add %r11, %r10 # ldc * (i + 4) + j
    vmovups (%r10), %ymm5
    add %r11, %r10 # ldc * (i + 5) + j
    vmovups (%r10), %ymm6
    add %r11, %r10 # ldc * (i + 6) + j
    vmovups (%r10), %ymm7
    add %r11, %r10 # ldc * (i + 7) + j
    vmovups (%r10), %ymm8

    mov $KC, %rax
    imul %r8, %rax # KC * i
    lea (%rdi, %rax, 4), %rax

    xor %r10, %r10
    add %r9, %r10 # + j
    lea (%rsi, %r10, 4), %r10

    xor %rcx, %rcx # pack_A col, pack_B row, k
.LKC_loop:

    load_broadcast_fma 0

    load_broadcast_fma 1

    load_broadcast_fma 2

    load_broadcast_fma 3

    load_broadcast_fma 4

    load_broadcast_fma 5

    load_broadcast_fma 6

    load_broadcast_fma 7

    add $NR, %rcx
    add $(NR * 4), %rax # + k * 4
    # + NC * k * 4
    add $(NR * NC * 4), %r10
    cmp $KC, %rcx
    jb .LKC_loop
.LKC_loop_end:

    mov %r11, %rax # ldc
    shr $2, %rax
    imul %r8, %rax # ldc * i
    add %r9, %rax # ldc * i + j
    lea (%rdx, %rax, 4), %r10 # C + ldc * i + j
    vmovups %ymm1, (%r10) # C[ldc * i + j : ldc * i + j + 8]
    add %r11, %r10 # ldc * (i + 1) + j
    vmovups %ymm2, (%r10)
    add %r11, %r10 # ldc * (i + 2) + j
    vmovups %ymm3, (%r10)
    add %r11, %r10 # ldc * (i + 3) + j
    vmovups %ymm4, (%r10)
    add %r11, %r10 # ldc * (i + 4) + j
    vmovups %ymm5, (%r10)
    add %r11, %r10 # ldc * (i + 5) + j
    vmovups %ymm6, (%r10)
    add %r11, %r10 # ldc * (i + 6) + j
    vmovups %ymm7, (%r10)
    add %r11, %r10 # ldc * (i + 7) + j
    vmovups %ymm8, (%r10)

    add $NR, %r9
    cmp $NC, %r9
    jb .LNR_loop
.LNR_loop_end:
    add $MR, %r8
    cmp $MC, %r8
    jb .LMR_loop
.LMR_loop_end:
    vzeroupper
    ret
#ifdef __ELF__
    .size sgemm_microkernel8x8_avx2, . - sgemm_microkernel8x8_avx2
#endif
