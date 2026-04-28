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
    .globl exp_avx2
    .type exp_avx2, @function
    .p2align 4
exp_avx2:
    xor %rax, %rax

    mov $254, %ecx
    vpcmpeqd %ymm12, %ymm12, %ymm12
    vbroadcastss .LC_log2e(%rip), %ymm5
    vbroadcastss .LC_a0(%rip), %ymm6
    vmovd %ecx, %xmm13
    vbroadcastss .LC_a1(%rip), %ymm7
    vbroadcastss .LC_a2(%rip), %ymm8
    vpsrld $25, %ymm12, %ymm12
    vpxor %ymm14, %ymm14, %ymm14 # ymm14 = 0
    vpbroadcastd %xmm13, %ymm13
    vbroadcastss .LC_a3(%rip), %ymm9
    vbroadcastss .LC_a4(%rip), %ymm10
    vbroadcastss .LC_a5(%rip), %ymm11

    mov %rdx, %rcx
    and $-8, %rdx
    jz .Lsimd_loop_end
    .p2align 4
    .p2align 3
.Lsimd_loop:
    vmulps (%rdi, %rax, 4), %ymm5, %ymm0 # y = x * math.log(math.e)
    vmovaps %ymm11, %ymm2
    vroundps $9, %ymm0, %ymm1 # n = math.floor(y)
    vsubps %ymm1, %ymm0, %ymm0 # f = y - n
    vcvtps2dq %ymm1, %ymm3 # int(n)
    vpaddd %ymm12, %ymm3, %ymm3 # n + 127

    # clamp exponent to [0, 254]
    vpmaxsd %ymm14, %ymm3, %ymm3
    vpminsd %ymm13, %ymm3, %ymm3

    vfmadd132ps %ymm0, %ymm10, %ymm2   # a5 * f + a4
    vpslld $23, %ymm3, %ymm3 # (n + 127) << 23
    vfmadd132ps %ymm0, %ymm9,  %ymm2   # (...) * f + a3
    vfmadd132ps %ymm0, %ymm8,  %ymm2   # (...) * f + a2
    vfmadd132ps %ymm0, %ymm7,  %ymm2   # (...) * f + a1
    vfmadd132ps %ymm0, %ymm6,  %ymm2   # (...) * f + a0

    vmulps %ymm2, %ymm3, %ymm0 # 2 ** n * 2 ** f

    vmovups %ymm0, (%rsi, %rax, 4)

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
    vmovaps %xmm11, %xmm2
    vroundps $9, %xmm0, %xmm1 # n = math.floor(y)
    vsubss %xmm1, %xmm0, %xmm0 # f = y - n
    vcvtps2dq %xmm1, %xmm3 # int(n)
    vpaddd %xmm12, %xmm3, %xmm3 # n + 127

    vpmaxsd %xmm14, %xmm3, %xmm3
    vpminsd %xmm13, %xmm3, %xmm3

    vfmadd132ss %xmm0, %xmm10, %xmm2   # a5 * f + a4
    vpslld $23, %xmm3, %xmm3
    vfmadd132ss %xmm0, %xmm9,  %xmm2   # (...) * f + a3
    vfmadd132ss %xmm0, %xmm8,  %xmm2   # (...) * f + a2
    vfmadd132ss %xmm0, %xmm7,  %xmm2   # (...) * f + a1
    vfmadd132ss %xmm0, %xmm6,  %xmm2   # (...) * f + a0

    vmulss %xmm2, %xmm3, %xmm0

    vmovss %xmm0, (%rsi, %rax, 4)

    inc %rax
    cmp %rcx, %rax
    jb .Ltail_loop
.Ltail_loop_end:
    vzeroupper
    ret
    .size exp_avx2, . - exp_avx2
