/* Operators built on the bound kernels, split across the worker pool. */
#ifndef ML_OPS_H
#define ML_OPS_H
#include "core/threads.h"
#include "model/mlb.h"

/* y[0..W->rows) = W x; rows split across threads in multiples of 4. */
void ml_matvec(ml_pool *pool, float *y, const ml_qtensor *W, const ml_qact *x);

/* Y[t*ldy + row] = W x_t for the n tokens of X (prompt prefill). */
void ml_matmul(ml_pool *pool, float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X);

/* One layer of the KV cache, rows [max_seq][kv_dim]. F32 when k is set;
 * otherwise Q8 with one scale per position and KV head: kq, vq hold the
 * quants and kd, vd the scales [max_seq][n_kv_heads]. */
typedef struct {
    const float *k, *v;
    const int8_t *kq, *vq;
    const float *kd, *vd;
    int window;   /* positions attended at most (sliding window); 0: all */
} ml_kv_layer;

/* Quantizes x[0, n) with a single scale, by the quantize_q8 rule. */
void ml_quantize_row(int8_t *q, float *d, const float *x, int n);

/* Causal GQA attention for one position. att_scratch: n_heads * max_seq
 * floats; qq, qd (Q8 cache only): dim bytes and n_heads floats for the
 * quantized query. */
void ml_attention(ml_pool *pool, float *out, const float *q, const ml_kv_layer *kv, float *att_scratch,
                  int8_t *qq, float *qd, const ml_config *cfg, int pos);
/* ml_attention for positions pos0..pos0+n-1 at once (prompt prefill), the
 * cache already holding their K and V; out, q: [n][dim]. att_scratch:
 * max(n_heads, ml_pool_size * ml_attention_batch_heads) * max_seq floats;
 * qq, qd (Q8 cache): n times ml_attention's. Each position's output is the
 * one ml_attention gives, bit for bit, with one pass over the pool for all. */
int ml_attention_batch_heads(const ml_config *cfg, int nthreads);
void ml_attention_batch(ml_pool *pool, float *out, const float *q, const ml_kv_layer *kv, float *att_scratch,
                        int8_t *qq, float *qd, const ml_config *cfg, int pos0, int n);

#endif
