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

#define PK_TILE 4   /* tokens per pass in the batched kernels */

/* Per-token block sums for tokens [t0, t0+T) of X. */
static inline void pk_block_sums_batch(int (*xsum)[PK_MAX_BLOCKS], const ml_qbatch *X, int t0, int T, int nb, int offset) {
    for (int t = 0; t < T; t++) {
        const ml_qact xt = { X->q + (size_t)(t0 + t) * X->k, X->d + (size_t)(t0 + t) * nb, X->k };
        pk_block_sums(xsum[t], &xt, nb, offset);
    }
}

/* Rows of W per pass over the tokens: about PK_ROW_BYTES of packed weights,
 * so that a pass stays in L2 while every tile of tokens reads it. Without it
 * each tile would stream all of r0..r1 from memory again. */
#ifndef PK_ROW_BYTES
#define PK_ROW_BYTES (512 * 1024)
#endif
static inline int pk_row_block(const ml_qtensor *W, int group) {
    const size_t row = (W->type == ML_Q8_0 ? (size_t)W->cols : (size_t)W->cols / 2) + W->cols / QK * sizeof(float);
    const int rows = (int)(PK_ROW_BYTES / row) / group * group;
    return rows > group ? rows : group;
}

/* Runs TILE_FN over X's tokens in tiles of N (at most 4), passing the tile's
 * token count as a constant so that each instantiation's loops over tokens
 * unroll completely; rows go in blocks of pk_row_block (PK: the
 * kernel's row group). Every output is the
 * same whatever the blocking. */
#define PK_TILES(TILE_FN, N)                                                      \
    for (int rb_ = pk_row_block(W, PK), b0_ = r0; b0_ < r1; b0_ += rb_) {             \
        const int b1_ = r1 - b0_ < rb_ ? r1 : b0_ + rb_;                          \
        for (int t0 = 0; t0 < X->n; t0 += (N)) {                                  \
            switch (X->n - t0 < (N) ? X->n - t0 : (N)) {                          \
            case 4: TILE_FN(Y, ldy, W, X, b0_, b1_, t0, 4); break;                \
            case 3: TILE_FN(Y, ldy, W, X, b0_, b1_, t0, 3); break;                \
            case 2: TILE_FN(Y, ldy, W, X, b0_, b1_, t0, 2); break;                \
            default: TILE_FN(Y, ldy, W, X, b0_, b1_, t0, 1); break;               \
            }                                                                     \
        }                                                                         \
    }

/* Writes the rows of a group that lie below r1. */
static inline void pk_store(float *y, const float *acc, int row, int G, int r1) {
    for (int i = 0; i < G && row + i < r1; i++) y[row + i] = acc[i];
}

#endif
