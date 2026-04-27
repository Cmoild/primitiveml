    .section .rodata
.LC_log2e:
.float 1.4426950408889634
.LC_a0:
.float 1.0
.LC_a1:
.float 0.6931471805599453
.LC_a2:
.float 0.2402265069591007
.LC_a3:
.float 0.05550410866482158
.LC_a4:
.float 0.009618129107628477
.LC_a5:
.float 0.001333355814642844
.LC_IEEE754_bias:
.int 127
.LC_254:
.int 254

    .section .text
    .globl exp_avx2_aligned
    .type exp_avx2_aligned, @function
exp_avx2_aligned:
    xor %rax, %rax

    vbroadcastss .LC_log2e(%rip), %ymm5
    vbroadcastss .LC_a0(%rip), %ymm6
    vbroadcastss .LC_a1(%rip), %ymm7
    vbroadcastss .LC_a2(%rip), %ymm8
    vbroadcastss .LC_a3(%rip), %ymm9
    vbroadcastss .LC_a4(%rip), %ymm10
    vbroadcastss .LC_a5(%rip), %ymm11
    vpbroadcastd .LC_IEEE754_bias(%rip), %ymm12
    vpbroadcastd .LC_254(%rip), %ymm13
    vpxor %ymm14, %ymm14, %ymm14 # ymm14 = 0

    mov %rdx, %rcx
    and $-8, %rdx
    jz .Lsimd_loop_end
.Lsimd_loop:
    vmulps (%rdi, %rax, 4), %ymm5, %ymm0 # y = x * math.log(math.e)
    vroundps $1, %ymm0, %ymm1 # n = math.floor(y)
    vsubps %ymm1, %ymm0, %ymm0 # f = y - n

    vmovaps %ymm10, %ymm2
    vfmadd231ps %ymm11, %ymm0, %ymm2 # a4 + f * a5
    vmovaps %ymm9, %ymm3
    vfmadd231ps %ymm2, %ymm0, %ymm3 # a3 + f * (...)
    vmovaps %ymm8, %ymm2
    vfmadd231ps %ymm3, %ymm0, %ymm2 # a2 + f * (...)
    vmovaps %ymm7, %ymm3
    vfmadd231ps %ymm2, %ymm0, %ymm3 # a1 + f * (...)
    vmovaps %ymm6, %ymm2
    vfmadd231ps %ymm3, %ymm0, %ymm2 # a0 + f * (...)

    vcvtps2dq %ymm1, %ymm3 # int(n)
    vpaddd %ymm12, %ymm3, %ymm3 # n + 127

    # clamp exponent to [0, 254]
    vpmaxsd %ymm14, %ymm3, %ymm3

    vpminsd %ymm13, %ymm3, %ymm3

    vpslld $23, %ymm3, %ymm3 # (n + 127) << 23
    vmulps %ymm2, %ymm3, %ymm0 # 2 ** n * 2 ** f

    vmovaps %ymm0, (%rsi, %rax, 4)

    add $8, %rax
    cmp %rdx, %rax
    jb .Lsimd_loop
.Lsimd_loop_end:
    lea (%rdi, %rdx, 4), %rdi
    lea (%rsi, %rdx, 4), %rsi

    xor %rax, %rax
    and $7, %rcx
    jz .Ltail_loop_end
.Ltail_loop:
    vmulss (%rdi, %rax, 4), %xmm5, %xmm0 # y = x * math.log(math.e)
    vroundps $1, %xmm0, %xmm1 # n = math.floor(y)
    vsubss %xmm1, %xmm0, %xmm0 # f = y - n

    vmovaps %xmm10, %xmm2
    vfmadd231ss %xmm11, %xmm0, %xmm2 # a4 + f * a5
    vmovaps %xmm9, %xmm3
    vfmadd231ss %xmm2, %xmm0, %xmm3 # a3 + f * (...)
    vmovaps %xmm8, %xmm2
    vfmadd231ss %xmm3, %xmm0, %xmm2 # a2 + f * (...)
    vmovaps %xmm7, %xmm3
    vfmadd231ss %xmm2, %xmm0, %xmm3 # a1 + f * (...)
    vmovaps %xmm6, %xmm2
    vfmadd231ss %xmm3, %xmm0, %xmm2 # a0 + f * (...)

    vcvtps2dq %xmm1, %xmm3 # int(n)
    vpaddd %xmm12, %xmm3, %xmm3 # n + 127

    vpmaxsd %xmm14, %xmm3, %xmm3

    vpminsd %xmm13, %xmm3, %xmm3

    vpslld $23, %xmm3, %xmm3
    vmulss %xmm2, %xmm3, %xmm0

    vmovss %xmm0, (%rsi, %rax, 4)

    inc %rax
    cmp %rcx, %rax
    jb .Ltail_loop
.Ltail_loop_end:
    vzeroupper
    ret
    .size exp_avx2_aligned, . - exp_avx2_aligned
