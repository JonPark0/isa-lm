/* Shared pieces of the x86 packed matvec kernels (core/pack.h layout). */
#ifndef ML_X86_PACK_H
#define ML_X86_PACK_H
#include "core/ml.h"

#define PK_MAX_BLOCKS 2048   /* cols <= 65536, enforced by the loader */

/* xsum[b] = offset * sum of block b's activation quants. */
static inline void pk_block_sums(int *xsum, const ml_qact *x, int nb, int offset) {
    for (int b = 0; b < nb; b++) {
        int s = 0;
        for (int j = 0; j < QK; j++) s += x->q[b * QK + j];
        xsum[b] = offset * s;
    }
}

/* Writes the rows of a group that lie below r1. */
static inline void pk_store(float *y, const float *acc, int row, int G, int r1) {
    for (int i = 0; i < G && row + i < r1; i++) y[row + i] = acc[i];
}

#endif
