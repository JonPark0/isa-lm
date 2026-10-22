/* ARMv8.2 dot-product (sdot) matvecs. Other ops fall back to NEON. */
#ifndef ML_K_NEON_DOTPROD_H
#define ML_K_NEON_DOTPROD_H
#include "core/ml.h"

#define ML_HAS_MATVEC_Q8_NEON_DOTPROD
#define ML_HAS_MATVEC_Q4_NEON_DOTPROD
#define ML_HAS_MATVEC_PK_NEON_DOTPROD   /* packed layout: 4-row groups, Q8 offset 0 */
#define ML_HAS_MATMUL_PK_NEON_DOTPROD

void ml_k_matvec_q8_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q8_pk_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_pk_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matmul_q8_pk_neon_dotprod(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1);
void ml_k_matmul_q4_pk_neon_dotprod(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1);

#endif
