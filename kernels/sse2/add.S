/* ml_k_add_sse2(float *y, const float *x, int n): y[i] += x[i].
 * Hand-written sample for the ASM=1 override path. x86-64 System V ABI:
 * rdi = y, rsi = x, edx = n. */
#if defined(__x86_64__) && !defined(_WIN32)
        .text
        .globl  ml_k_add_sse2
        .type   ml_k_add_sse2, @function
ml_k_add_sse2:
        movslq  %edx, %rdx
        xorl    %eax, %eax              /* i = 0 */
        movq    %rdx, %rcx
        andq    $-4, %rcx               /* n rounded down to a multiple of 4 */
        jmp     2f
1:      movups  (%rdi,%rax,4), %xmm0    /* 4 floats per step */
        movups  (%rsi,%rax,4), %xmm1
        addps   %xmm1, %xmm0
        movups  %xmm0, (%rdi,%rax,4)
        addq    $4, %rax
2:      cmpq    %rcx, %rax
        jl      1b
        jmp     4f
3:      movss   (%rdi,%rax,4), %xmm0    /* scalar tail */
        addss   (%rsi,%rax,4), %xmm0
        movss   %xmm0, (%rdi,%rax,4)
        incq    %rax
4:      cmpq    %rdx, %rax
        jl      3b
        ret
        .size   ml_k_add_sse2, .-ml_k_add_sse2
        .section .note.GNU-stack,"",@progbits
#endif
