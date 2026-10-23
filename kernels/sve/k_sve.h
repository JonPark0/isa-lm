/* SVE kernels, vector-length agnostic (128..2048-bit). Other ops fall back to
 * NEON dotprod / NEON. */
#ifndef ML_K_SVE_H
#define ML_K_SVE_H
#include "core/ml.h"

#define ML_HAS_MATVEC_Q8_SVE
#define ML_HAS_MATVEC_Q4_SVE
#define ML_HAS_QUANTIZE_Q8_SVE
#define ML_HAS_RMSNORM_SVE
#define ML_HAS_ADD_SVE
#define ML_HAS_DOT_F32_SVE
#define ML_HAS_AXPY_F32_SVE

void ml_k_matvec_q8_sve(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_sve(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_quantize_q8_sve(ml_qact *out, const float *x, int k);
void ml_k_rmsnorm_sve(float *o, const float *x, const float *w, int n, float eps);
void ml_k_add_sve(float *y, const float *x, int n);
void ml_k_dot_f32_sve(float *s, const float *q, const float *k, size_t ldk, int rows, int n, float scale);
void ml_k_axpy_f32_sve(float *y, const float *a, const float *x, size_t ldx, int rows, int n);

#endif
