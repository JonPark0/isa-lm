/* AVX-512 (F, BW, VNNI) kernels: 512-bit vpdpbusd matvecs (two blocks per op)
 * and 16-wide float ops. Others fall back to AVX2. */
#ifndef ML_K_AVX512VNNI_H
#define ML_K_AVX512VNNI_H
#include "core/ml.h"

#define ML_HAS_MATVEC_Q8_AVX512VNNI
#define ML_HAS_MATVEC_Q4_AVX512VNNI
#define ML_HAS_QUANTIZE_Q8_AVX512VNNI
#define ML_HAS_RMSNORM_AVX512VNNI
#define ML_HAS_ADD_AVX512VNNI
#define ML_HAS_DOT_F32_AVX512VNNI
#define ML_HAS_AXPY_F32_AVX512VNNI

void ml_k_matvec_q8_avx512vnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_avx512vnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_quantize_q8_avx512vnni(ml_qact *out, const float *x, int k);
void ml_k_rmsnorm_avx512vnni(float *o, const float *x, const float *w, int n, float eps);
void ml_k_add_avx512vnni(float *y, const float *x, int n);
float ml_k_dot_f32_avx512vnni(const float *a, const float *b, int n);
void ml_k_axpy_f32_avx512vnni(float *y, float a, const float *x, int n);

#endif
