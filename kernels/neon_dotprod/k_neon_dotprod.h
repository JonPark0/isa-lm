/* ARMv8.2 dot-product (sdot) matvecs. Other ops fall back to NEON. */
#ifndef ML_K_NEON_DOTPROD_H
#define ML_K_NEON_DOTPROD_H
#include "core/ml.h"

#define ML_HAS_MATVEC_Q8_NEON_DOTPROD
#define ML_HAS_MATVEC_Q4_NEON_DOTPROD

void ml_k_matvec_q8_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);

#endif
