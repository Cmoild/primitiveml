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

    mov %rdx, %rcx
    and $-8, %rdx
    jz .Lsimd_loop_end
.Lsimd_loop:
    vmulps (%rdi, %rax, 4), %ymm5, %ymm0 # y = x * math.log(math.e)
    vroundps $1, %ymm0, %ymm1 # n = math.floor(y)
    vsubps %ymm1, %ymm0, %ymm0 # f = y - n
    
    add $8, %rax
    cmp %rdx, %rax
    jb .Lsimd_loop
.Lsimd_loop_end:
    vzeroupper
    ret
    .size exp_avx2_aligned, . - exp_avx2_aligned
