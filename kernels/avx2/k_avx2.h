/* AVX2 + FMA kernels (256-bit). */
#ifndef ML_K_AVX2_H
#define ML_K_AVX2_H
#include "core/ml.h"

#define ML_HAS_MATVEC_Q8_AVX2
#define ML_HAS_MATVEC_Q4_AVX2
#define ML_HAS_QUANTIZE_Q8_AVX2
#define ML_HAS_RMSNORM_AVX2
#define ML_HAS_SOFTMAX_AVX2
#define ML_HAS_ROPE_AVX2
#define ML_HAS_SILU_MUL_AVX2
#define ML_HAS_ADD_AVX2
#define ML_HAS_DOT_F32_AVX2
#define ML_HAS_AXPY_F32_AVX2
#define ML_HAS_DOT_Q8_AVX2
#define ML_HAS_AXPY_Q8_AVX2
#define ML_HAS_MATVEC_PK_AVX2   /* packed layout: 8-row groups, Q8 offset 0 */
#define ML_HAS_MATMUL_PK_AVX2

void ml_k_matvec_q8_avx2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_avx2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q8_pk_avx2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_pk_avx2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matmul_q8_pk_avx2(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1);
void ml_k_matmul_q4_pk_avx2(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1);
void ml_k_quantize_q8_avx2(ml_qact *out, const float *x, int k);
void ml_k_rmsnorm_avx2(float *o, const float *x, const float *w, int n, float eps);
void ml_k_softmax_avx2(float *x, int n);
void ml_k_rope_avx2(float *q, float *k, const float *cos_t, const float *sin_t,
                    int head_dim, int n_heads, int n_kv_heads);
void ml_k_silu_mul_avx2(float *g, const float *u, int n);
void ml_k_add_avx2(float *y, const float *x, int n);
void ml_k_dot_f32_avx2(float *s, const float *q, const float *k, size_t ldk, int rows, int n, float scale);
void ml_k_axpy_f32_avx2(float *y, const float *a, const float *x, size_t ldx, int rows, int n);
#define ML_HAS_DOT_F16_AVX2
#define ML_HAS_AXPY_F16_AVX2
void ml_k_dot_f16_avx2(float *s, const float *q, const uint16_t *k, size_t ldk, int rows, int n, float scale);
void ml_k_axpy_f16_avx2(float *y, const float *a, const uint16_t *x, size_t ldx, int rows, int n);
void ml_k_dot_q8_avx2(float *s, const int8_t *q, float dq, const int8_t *k, const float *dk, size_t ldk, size_t lddk, int rows, int n, float scale);
void ml_k_axpy_q8_avx2(float *y, const float *a, const int8_t *x, const float *dx, size_t ldx, size_t lddx, int rows, int n);

#endif
