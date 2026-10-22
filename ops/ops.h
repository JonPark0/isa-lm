/* Operators built on the bound kernels, split across the worker pool. */
#ifndef ML_OPS_H
#define ML_OPS_H
#include "core/threads.h"
#include "model/mlb.h"

/* y[0..W->rows) = W x; rows split across threads in multiples of 4. */
void ml_matvec(ml_pool *pool, float *y, const ml_qtensor *W, const ml_qact *x);

/* Y[t*ldy + row] = W x_t for the n tokens of X (prompt prefill). */
void ml_matmul(ml_pool *pool, float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X);

/* Causal GQA attention for one position. kcache/vcache: this layer's
 * [max_seq][kv_dim] caches; att_scratch: n_heads * max_seq floats. */
void ml_attention(ml_pool *pool, float *out, const float *q, const float *kcache,
                  const float *vcache, float *att_scratch, const ml_config *cfg, int pos);

#endif
