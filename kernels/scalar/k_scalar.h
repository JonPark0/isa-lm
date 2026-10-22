/* Scalar reference kernels. Always compiled: fallback for every ISA and the
 * baseline every test compares against. */
#ifndef ML_K_SCALAR_H
#define ML_K_SCALAR_H
#include "core/ml.h"

void ml_k_matvec_q8_scalar(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_scalar(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_quantize_q8_scalar(ml_qact *out, const float *x, int k);
void ml_k_rmsnorm_scalar(float *o, const float *x, const float *w, int n, float eps);
void ml_k_softmax_scalar(float *x, int n);
void ml_k_rope_scalar(float *q, float *k, const float *cos_t, const float *sin_t,
                      int head_dim, int n_heads, int n_kv_heads);
void ml_k_silu_mul_scalar(float *g, const float *u, int n);
void ml_k_add_scalar(float *y, const float *x, int n);
float ml_k_dot_f32_scalar(const float *a, const float *b, int n);
void ml_k_axpy_f32_scalar(float *y, float a, const float *x, int n);

#endif
