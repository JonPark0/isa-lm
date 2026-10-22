/* SSE4.1 (+SSSE3) kernels: integer dot products via pmaddubsw. Other ops fall
 * back to SSE2. */
#ifndef ML_K_SSE41_H
#define ML_K_SSE41_H
#include "core/ml.h"

#define ML_HAS_MATVEC_Q8_SSE41
#define ML_HAS_MATVEC_Q4_SSE41

void ml_k_matvec_q8_sse41(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_sse41(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);

#endif
