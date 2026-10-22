#include "config/ml_config.h"

#if ML_PACK_G
void ml_k_matmul_q8_pk_loop(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    for (int t = 0; t < X->n; t++) {
        const ml_qact xt = { X->q + (size_t)t * X->k, X->d + (size_t)t * (X->k / QK), X->k };
        ml_k_matvec_q8_pk(Y + (size_t)t * ldy, W, &xt, r0, r1);
    }
}

void ml_k_matmul_q4_pk_loop(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    for (int t = 0; t < X->n; t++) {
        const ml_qact xt = { X->q + (size_t)t * X->k, X->d + (size_t)t * (X->k / QK), X->k };
        ml_k_matvec_q4_pk(Y + (size_t)t * ldy, W, &xt, r0, r1);
    }
}
#else
typedef int ml_pk_loop_unused;   /* ISO C forbids an empty translation unit */
#endif
