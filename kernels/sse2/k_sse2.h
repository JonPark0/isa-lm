/* SSE2 kernels (x86-64 baseline). */
#ifndef ML_K_SSE2_H
#define ML_K_SSE2_H
#include "core/ml.h"

#define ML_HAS_MATVEC_Q8_SSE2
#define ML_HAS_MATVEC_Q4_SSE2
#define ML_HAS_QUANTIZE_Q8_SSE2
#define ML_HAS_RMSNORM_SSE2
#define ML_HAS_SOFTMAX_SSE2
#define ML_HAS_ROPE_SSE2
#define ML_HAS_SILU_MUL_SSE2
#define ML_HAS_ADD_SSE2
#define ML_HAS_DOT_F32_SSE2
#define ML_HAS_AXPY_F32_SSE2

void ml_k_matvec_q8_sse2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_sse2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_quantize_q8_sse2(ml_qact *out, const float *x, int k);
void ml_k_rmsnorm_sse2(float *o, const float *x, const float *w, int n, float eps);
void ml_k_softmax_sse2(float *x, int n);
void ml_k_rope_sse2(float *q, float *k, const float *cos_t, const float *sin_t,
                    int head_dim, int n_heads, int n_kv_heads);
void ml_k_silu_mul_sse2(float *g, const float *u, int n);
void ml_k_add_sse2(float *y, const float *x, int n);
float ml_k_dot_f32_sse2(const float *a, const float *b, int n);
void ml_k_axpy_f32_sse2(float *y, float a, const float *x, int n);

#endif
