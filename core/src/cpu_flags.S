    .section .text

    .globl avx2_fma_supported
    .type avx2_fma_supported, @function
avx2_fma_supported:
    push %rbx
    mov $0x01, %eax
    cpuid
    mov %ecx, %eax

    # check FMA flag
    shr $12, %eax
    and $0x01, %eax
    cmp $0x01, %eax
    jne .Lnot_supported

    mov %ecx, %eax

    # check AVX flag
    shr $28, %eax
    and $0x01, %eax
    cmp $0x01, %eax
    jne .Lnot_supported

    mov %ecx, %eax

    # check OSXSAVE flag
    shr $27, %eax
    and $0x01, %eax
    cmp $0x01, %eax
    jne .Lnot_supported

    mov $7, %eax
    xor %ecx, %ecx
    cpuid
    mov %ebx, %eax

    # check AVX2 flag
    shr $5, %eax
    and $0x01, %eax
    cmp $0x01, %eax
    jne .Lnot_supported

    # check OS has enabled both XMM and YMM state support
    xor %ecx, %ecx
    xgetbv
    and $0x06, %eax
    cmp $0x06, %eax
    jne .Lnot_supported

    pop %rbx
    mov $1, %rax
    ret
.Lnot_supported:
    pop %rbx
    mov $0, %rax
    ret
    .size avx2_fma_supported, . - avx2_fma_supported
