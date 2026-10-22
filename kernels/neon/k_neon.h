/* NEON kernels for ARMv7 (with NEON) and AArch64. */
#ifndef ML_K_NEON_H
#define ML_K_NEON_H
#include "core/ml.h"

#define ML_HAS_MATVEC_Q8_NEON
#define ML_HAS_MATVEC_Q4_NEON
#define ML_HAS_QUANTIZE_Q8_NEON
#define ML_HAS_RMSNORM_NEON
#define ML_HAS_SOFTMAX_NEON
#define ML_HAS_ROPE_NEON
#define ML_HAS_SILU_MUL_NEON
#define ML_HAS_ADD_NEON
#define ML_HAS_DOT_F32_NEON
#define ML_HAS_AXPY_F32_NEON

void ml_k_matvec_q8_neon(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_neon(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_quantize_q8_neon(ml_qact *out, const float *x, int k);
void ml_k_rmsnorm_neon(float *o, const float *x, const float *w, int n, float eps);
void ml_k_softmax_neon(float *x, int n);
void ml_k_rope_neon(float *q, float *k, const float *cos_t, const float *sin_t,
                    int head_dim, int n_heads, int n_kv_heads);
void ml_k_silu_mul_neon(float *g, const float *u, int n);
void ml_k_add_neon(float *y, const float *x, int n);
float ml_k_dot_f32_neon(const float *a, const float *b, int n);
void ml_k_axpy_f32_neon(float *y, float a, const float *x, int n);

#endif
