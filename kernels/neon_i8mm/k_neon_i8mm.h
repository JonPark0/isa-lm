/* ARMv8.6 int8 matrix multiply (smmla) for batched prefill. Matvecs and all
 * other ops fall back to NEON dotprod, and the packed layout is the same. */
#ifndef ML_K_NEON_I8MM_H
#define ML_K_NEON_I8MM_H
#include "core/ml.h"

#define ML_HAS_MATMUL_PK_NEON_I8MM

void ml_k_matmul_q8_pk_neon_i8mm(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1);
void ml_k_matmul_q4_pk_neon_i8mm(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1);

#endif
