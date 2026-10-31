/* AMX (AMX-TILE, AMX-INT8) kernels for prompt prefill: the batched packed
 * matmuls run on tile dot products. Everything else, decoding included, is
 * the AVX-512 VNNI build's (the amx chain contains it). */
#ifndef ML_K_AMX_H
#define ML_K_AMX_H
#include "core/ml.h"

#define ML_HAS_MATMUL_PK_AMX   /* packed layout of avx512vnni: 16-row groups, Q8 offset 128 */

/* Y[t*ldy + row] = W x_t for every token of X; r0 a multiple of 16. */
void ml_k_matmul_q8_pk_amx(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1);
void ml_k_matmul_q4_pk_amx(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1);

#endif
