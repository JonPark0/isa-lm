/* Batched packed kernels for ISAs without a native one: the packed matvec per
 * token. */
#ifndef ML_PK_LOOP_H
#define ML_PK_LOOP_H
#include "core/ml.h"

void ml_k_matmul_q8_pk_loop(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1);
void ml_k_matmul_q4_pk_loop(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1);

#endif
