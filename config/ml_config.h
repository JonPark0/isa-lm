/* Compile-time kernel binding.
 *
 * The Makefile defines ML_ISA_<TARGET> and ML_ENABLE_<ISA> for every ISA in the
 * target's fallback chain. Each enabled ISA header defines ML_HAS_<OP>_<ISA>
 * for the kernels it implements; every generic name ml_k_<op> below resolves
 * to the highest available implementation, else to the scalar one.
 */
#ifndef ML_CONFIG_H
#define ML_CONFIG_H

#include "kernels/scalar/k_scalar.h"

#ifdef ML_ENABLE_NEON
#include "kernels/neon/k_neon.h"
#endif
#ifdef ML_ENABLE_NEON_DOTPROD
#include "kernels/neon_dotprod/k_neon_dotprod.h"
#endif
#ifdef ML_ENABLE_SVE
#include "kernels/sve/k_sve.h"
#endif
#ifdef ML_ENABLE_SSE2
#include "kernels/sse2/k_sse2.h"
#endif
#ifdef ML_ENABLE_SSE41
#include "kernels/sse41/k_sse41.h"
#endif
#ifdef ML_ENABLE_AVX2
#include "kernels/avx2/k_avx2.h"
#endif
#ifdef ML_ENABLE_AVXVNNI
#include "kernels/avxvnni/k_avxvnni.h"
#endif
#ifdef ML_ENABLE_AVX512VNNI
#include "kernels/avx512vnni/k_avx512vnni.h"
#endif

#if defined(ML_HAS_MATVEC_Q8_AVX512VNNI)
#define ml_k_matvec_q8 ml_k_matvec_q8_avx512vnni
#elif defined(ML_HAS_MATVEC_Q8_AVXVNNI)
#define ml_k_matvec_q8 ml_k_matvec_q8_avxvnni
#elif defined(ML_HAS_MATVEC_Q8_AVX2)
#define ml_k_matvec_q8 ml_k_matvec_q8_avx2
#elif defined(ML_HAS_MATVEC_Q8_SSE41)
#define ml_k_matvec_q8 ml_k_matvec_q8_sse41
#elif defined(ML_HAS_MATVEC_Q8_SSE2)
#define ml_k_matvec_q8 ml_k_matvec_q8_sse2
#elif defined(ML_HAS_MATVEC_Q8_SVE)
#define ml_k_matvec_q8 ml_k_matvec_q8_sve
#elif defined(ML_HAS_MATVEC_Q8_NEON_DOTPROD)
#define ml_k_matvec_q8 ml_k_matvec_q8_neon_dotprod
#elif defined(ML_HAS_MATVEC_Q8_NEON)
#define ml_k_matvec_q8 ml_k_matvec_q8_neon
#else
#define ml_k_matvec_q8 ml_k_matvec_q8_scalar
#endif

#if defined(ML_HAS_MATVEC_Q4_AVX512VNNI)
#define ml_k_matvec_q4 ml_k_matvec_q4_avx512vnni
#elif defined(ML_HAS_MATVEC_Q4_AVXVNNI)
#define ml_k_matvec_q4 ml_k_matvec_q4_avxvnni
#elif defined(ML_HAS_MATVEC_Q4_AVX2)
#define ml_k_matvec_q4 ml_k_matvec_q4_avx2
#elif defined(ML_HAS_MATVEC_Q4_SSE41)
#define ml_k_matvec_q4 ml_k_matvec_q4_sse41
#elif defined(ML_HAS_MATVEC_Q4_SSE2)
#define ml_k_matvec_q4 ml_k_matvec_q4_sse2
#elif defined(ML_HAS_MATVEC_Q4_SVE)
#define ml_k_matvec_q4 ml_k_matvec_q4_sve
#elif defined(ML_HAS_MATVEC_Q4_NEON_DOTPROD)
#define ml_k_matvec_q4 ml_k_matvec_q4_neon_dotprod
#elif defined(ML_HAS_MATVEC_Q4_NEON)
#define ml_k_matvec_q4 ml_k_matvec_q4_neon
#else
#define ml_k_matvec_q4 ml_k_matvec_q4_scalar
#endif

#if defined(ML_HAS_QUANTIZE_Q8_AVX512VNNI)
#define ml_k_quantize_q8 ml_k_quantize_q8_avx512vnni
#elif defined(ML_HAS_QUANTIZE_Q8_AVXVNNI)
#define ml_k_quantize_q8 ml_k_quantize_q8_avxvnni
#elif defined(ML_HAS_QUANTIZE_Q8_AVX2)
#define ml_k_quantize_q8 ml_k_quantize_q8_avx2
#elif defined(ML_HAS_QUANTIZE_Q8_SSE41)
#define ml_k_quantize_q8 ml_k_quantize_q8_sse41
#elif defined(ML_HAS_QUANTIZE_Q8_SSE2)
#define ml_k_quantize_q8 ml_k_quantize_q8_sse2
#elif defined(ML_HAS_QUANTIZE_Q8_SVE)
#define ml_k_quantize_q8 ml_k_quantize_q8_sve
#elif defined(ML_HAS_QUANTIZE_Q8_NEON_DOTPROD)
#define ml_k_quantize_q8 ml_k_quantize_q8_neon_dotprod
#elif defined(ML_HAS_QUANTIZE_Q8_NEON)
#define ml_k_quantize_q8 ml_k_quantize_q8_neon
#else
#define ml_k_quantize_q8 ml_k_quantize_q8_scalar
#endif

#if defined(ML_HAS_RMSNORM_AVX512VNNI)
#define ml_k_rmsnorm ml_k_rmsnorm_avx512vnni
#elif defined(ML_HAS_RMSNORM_AVXVNNI)
#define ml_k_rmsnorm ml_k_rmsnorm_avxvnni
#elif defined(ML_HAS_RMSNORM_AVX2)
#define ml_k_rmsnorm ml_k_rmsnorm_avx2
#elif defined(ML_HAS_RMSNORM_SSE41)
#define ml_k_rmsnorm ml_k_rmsnorm_sse41
#elif defined(ML_HAS_RMSNORM_SSE2)
#define ml_k_rmsnorm ml_k_rmsnorm_sse2
#elif defined(ML_HAS_RMSNORM_SVE)
#define ml_k_rmsnorm ml_k_rmsnorm_sve
#elif defined(ML_HAS_RMSNORM_NEON_DOTPROD)
#define ml_k_rmsnorm ml_k_rmsnorm_neon_dotprod
#elif defined(ML_HAS_RMSNORM_NEON)
#define ml_k_rmsnorm ml_k_rmsnorm_neon
#else
#define ml_k_rmsnorm ml_k_rmsnorm_scalar
#endif

#if defined(ML_HAS_SOFTMAX_AVX512VNNI)
#define ml_k_softmax ml_k_softmax_avx512vnni
#elif defined(ML_HAS_SOFTMAX_AVXVNNI)
#define ml_k_softmax ml_k_softmax_avxvnni
#elif defined(ML_HAS_SOFTMAX_AVX2)
#define ml_k_softmax ml_k_softmax_avx2
#elif defined(ML_HAS_SOFTMAX_SSE41)
#define ml_k_softmax ml_k_softmax_sse41
#elif defined(ML_HAS_SOFTMAX_SSE2)
#define ml_k_softmax ml_k_softmax_sse2
#elif defined(ML_HAS_SOFTMAX_SVE)
#define ml_k_softmax ml_k_softmax_sve
#elif defined(ML_HAS_SOFTMAX_NEON_DOTPROD)
#define ml_k_softmax ml_k_softmax_neon_dotprod
#elif defined(ML_HAS_SOFTMAX_NEON)
#define ml_k_softmax ml_k_softmax_neon
#else
#define ml_k_softmax ml_k_softmax_scalar
#endif

#if defined(ML_HAS_ROPE_AVX512VNNI)
#define ml_k_rope ml_k_rope_avx512vnni
#elif defined(ML_HAS_ROPE_AVXVNNI)
#define ml_k_rope ml_k_rope_avxvnni
#elif defined(ML_HAS_ROPE_AVX2)
#define ml_k_rope ml_k_rope_avx2
#elif defined(ML_HAS_ROPE_SSE41)
#define ml_k_rope ml_k_rope_sse41
#elif defined(ML_HAS_ROPE_SSE2)
#define ml_k_rope ml_k_rope_sse2
#elif defined(ML_HAS_ROPE_SVE)
#define ml_k_rope ml_k_rope_sve
#elif defined(ML_HAS_ROPE_NEON_DOTPROD)
#define ml_k_rope ml_k_rope_neon_dotprod
#elif defined(ML_HAS_ROPE_NEON)
#define ml_k_rope ml_k_rope_neon
#else
#define ml_k_rope ml_k_rope_scalar
#endif

#if defined(ML_HAS_SILU_MUL_AVX512VNNI)
#define ml_k_silu_mul ml_k_silu_mul_avx512vnni
#elif defined(ML_HAS_SILU_MUL_AVXVNNI)
#define ml_k_silu_mul ml_k_silu_mul_avxvnni
#elif defined(ML_HAS_SILU_MUL_AVX2)
#define ml_k_silu_mul ml_k_silu_mul_avx2
#elif defined(ML_HAS_SILU_MUL_SSE41)
#define ml_k_silu_mul ml_k_silu_mul_sse41
#elif defined(ML_HAS_SILU_MUL_SSE2)
#define ml_k_silu_mul ml_k_silu_mul_sse2
#elif defined(ML_HAS_SILU_MUL_SVE)
#define ml_k_silu_mul ml_k_silu_mul_sve
#elif defined(ML_HAS_SILU_MUL_NEON_DOTPROD)
#define ml_k_silu_mul ml_k_silu_mul_neon_dotprod
#elif defined(ML_HAS_SILU_MUL_NEON)
#define ml_k_silu_mul ml_k_silu_mul_neon
#else
#define ml_k_silu_mul ml_k_silu_mul_scalar
#endif

#if defined(ML_HAS_ADD_AVX512VNNI)
#define ml_k_add ml_k_add_avx512vnni
#elif defined(ML_HAS_ADD_AVXVNNI)
#define ml_k_add ml_k_add_avxvnni
#elif defined(ML_HAS_ADD_AVX2)
#define ml_k_add ml_k_add_avx2
#elif defined(ML_HAS_ADD_SSE41)
#define ml_k_add ml_k_add_sse41
#elif defined(ML_HAS_ADD_SSE2)
#define ml_k_add ml_k_add_sse2
#elif defined(ML_HAS_ADD_SVE)
#define ml_k_add ml_k_add_sve
#elif defined(ML_HAS_ADD_NEON_DOTPROD)
#define ml_k_add ml_k_add_neon_dotprod
#elif defined(ML_HAS_ADD_NEON)
#define ml_k_add ml_k_add_neon
#else
#define ml_k_add ml_k_add_scalar
#endif

#if defined(ML_HAS_DOT_F32_AVX512VNNI)
#define ml_k_dot_f32 ml_k_dot_f32_avx512vnni
#elif defined(ML_HAS_DOT_F32_AVXVNNI)
#define ml_k_dot_f32 ml_k_dot_f32_avxvnni
#elif defined(ML_HAS_DOT_F32_AVX2)
#define ml_k_dot_f32 ml_k_dot_f32_avx2
#elif defined(ML_HAS_DOT_F32_SSE41)
#define ml_k_dot_f32 ml_k_dot_f32_sse41
#elif defined(ML_HAS_DOT_F32_SSE2)
#define ml_k_dot_f32 ml_k_dot_f32_sse2
#elif defined(ML_HAS_DOT_F32_SVE)
#define ml_k_dot_f32 ml_k_dot_f32_sve
#elif defined(ML_HAS_DOT_F32_NEON_DOTPROD)
#define ml_k_dot_f32 ml_k_dot_f32_neon_dotprod
#elif defined(ML_HAS_DOT_F32_NEON)
#define ml_k_dot_f32 ml_k_dot_f32_neon
#else
#define ml_k_dot_f32 ml_k_dot_f32_scalar
#endif

#if defined(ML_HAS_AXPY_F32_AVX512VNNI)
#define ml_k_axpy_f32 ml_k_axpy_f32_avx512vnni
#elif defined(ML_HAS_AXPY_F32_AVXVNNI)
#define ml_k_axpy_f32 ml_k_axpy_f32_avxvnni
#elif defined(ML_HAS_AXPY_F32_AVX2)
#define ml_k_axpy_f32 ml_k_axpy_f32_avx2
#elif defined(ML_HAS_AXPY_F32_SSE41)
#define ml_k_axpy_f32 ml_k_axpy_f32_sse41
#elif defined(ML_HAS_AXPY_F32_SSE2)
#define ml_k_axpy_f32 ml_k_axpy_f32_sse2
#elif defined(ML_HAS_AXPY_F32_SVE)
#define ml_k_axpy_f32 ml_k_axpy_f32_sve
#elif defined(ML_HAS_AXPY_F32_NEON_DOTPROD)
#define ml_k_axpy_f32 ml_k_axpy_f32_neon_dotprod
#elif defined(ML_HAS_AXPY_F32_NEON)
#define ml_k_axpy_f32 ml_k_axpy_f32_neon
#else
#define ml_k_axpy_f32 ml_k_axpy_f32_scalar
#endif

/* Packed (row-interleaved) matvec kernels and the layout they expect. */
#if defined(ML_HAS_MATVEC_PK_AVX512VNNI)
#define ml_k_matvec_q8_pk ml_k_matvec_q8_pk_avx512vnni
#define ml_k_matvec_q4_pk ml_k_matvec_q4_pk_avx512vnni
#define ML_PACK_G 16
#define ML_PACK_Q8_OFFSET 128
#define ML_PACK_Q8 1
#elif defined(ML_HAS_MATVEC_PK_AVXVNNI)
#define ml_k_matvec_q8_pk ml_k_matvec_q8_pk_avxvnni
#define ml_k_matvec_q4_pk ml_k_matvec_q4_pk_avxvnni
#define ML_PACK_G 8
#define ML_PACK_Q8_OFFSET 128
#define ML_PACK_Q8 1
#elif defined(ML_HAS_MATVEC_PK_AVX2)
#define ml_k_matvec_q8_pk ml_k_matvec_q8_pk_avx2
#define ml_k_matvec_q4_pk ml_k_matvec_q4_pk_avx2
#define ML_PACK_G 8
#define ML_PACK_Q8_OFFSET 0
#define ML_PACK_Q8 1
#elif defined(ML_HAS_MATVEC_PK_SSE41)
#define ml_k_matvec_q8_pk ml_k_matvec_q8_pk_sse41
#define ml_k_matvec_q4_pk ml_k_matvec_q4_pk_sse41
#define ML_PACK_G 4
#define ML_PACK_Q8_OFFSET 0
#define ML_PACK_Q8 0   /* 4-row Q8 groups are no faster than row-major at 128 bits */
#elif defined(ML_HAS_MATVEC_PK_NEON_DOTPROD)
#define ml_k_matvec_q8_pk ml_k_matvec_q8_pk_neon_dotprod
#define ml_k_matvec_q4_pk ml_k_matvec_q4_pk_neon_dotprod
#define ML_PACK_G 4
#define ML_PACK_Q8_OFFSET 0
#define ML_PACK_Q8 1
#elif defined(ML_HAS_MATVEC_PK_NEON)
#define ml_k_matvec_q8_pk ml_k_matvec_q8_pk_neon
#define ml_k_matvec_q4_pk ml_k_matvec_q4_pk_neon
#define ML_PACK_G 4
#define ML_PACK_Q8_OFFSET 0
#define ML_PACK_Q8 1
#else
#define ML_PACK_G 0
#define ML_PACK_Q8_OFFSET 0
#define ML_PACK_Q8 0
#endif

/* Batched packed kernels: native where the packed ISA has one, else a loop
 * over the packed matvec. */
#include "kernels/pk_loop.h"
#if defined(ML_HAS_MATVEC_PK_AVX512VNNI) && defined(ML_HAS_MATMUL_PK_AVX512VNNI)
#define ml_k_matmul_q8_pk ml_k_matmul_q8_pk_avx512vnni
#define ml_k_matmul_q4_pk ml_k_matmul_q4_pk_avx512vnni
#elif !defined(ML_HAS_MATVEC_PK_AVX512VNNI) && defined(ML_HAS_MATMUL_PK_AVXVNNI)
#define ml_k_matmul_q8_pk ml_k_matmul_q8_pk_avxvnni
#define ml_k_matmul_q4_pk ml_k_matmul_q4_pk_avxvnni
#elif !defined(ML_HAS_MATVEC_PK_AVX512VNNI) && !defined(ML_HAS_MATVEC_PK_AVXVNNI) && defined(ML_HAS_MATMUL_PK_AVX2)
#define ml_k_matmul_q8_pk ml_k_matmul_q8_pk_avx2
#define ml_k_matmul_q4_pk ml_k_matmul_q4_pk_avx2
#elif !defined(ML_HAS_MATVEC_PK_AVX2) && defined(ML_HAS_MATMUL_PK_SSE41)
#define ml_k_matmul_q8_pk ml_k_matmul_q8_pk_sse41
#define ml_k_matmul_q4_pk ml_k_matmul_q4_pk_sse41
#elif defined(ML_HAS_MATMUL_PK_NEON_DOTPROD)
#define ml_k_matmul_q8_pk ml_k_matmul_q8_pk_neon_dotprod
#define ml_k_matmul_q4_pk ml_k_matmul_q4_pk_neon_dotprod
#elif defined(ML_HAS_MATMUL_PK_NEON)
#define ml_k_matmul_q8_pk ml_k_matmul_q8_pk_neon
#define ml_k_matmul_q4_pk ml_k_matmul_q4_pk_neon
#elif ML_PACK_G
#define ml_k_matmul_q8_pk ml_k_matmul_q8_pk_loop
#define ml_k_matmul_q4_pk ml_k_matmul_q4_pk_loop
#endif

#if defined(ML_ISA_SCALAR)
#define ML_ISA_NAME "scalar"
#elif defined(ML_ISA_AVX512VNNI)
#define ML_ISA_NAME "avx512vnni"
#elif defined(ML_ISA_AVXVNNI)
#define ML_ISA_NAME "avxvnni"
#elif defined(ML_ISA_AVX2)
#define ML_ISA_NAME "avx2"
#elif defined(ML_ISA_SSE41)
#define ML_ISA_NAME "sse41"
#elif defined(ML_ISA_SSE2)
#define ML_ISA_NAME "sse2"
#elif defined(ML_ISA_SVE)
#define ML_ISA_NAME "sve"
#elif defined(ML_ISA_NEON_DOTPROD)
#define ML_ISA_NAME "neon_dotprod"
#elif defined(ML_ISA_NEON)
#define ML_ISA_NAME "neon"
#else
#error "no ML_ISA_* defined; build through the Makefile"
#endif

#define ML_STR_(x) #x
#define ML_STR(x) ML_STR_(x)

#endif
