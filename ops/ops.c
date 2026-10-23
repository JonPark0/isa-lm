#include <math.h>
#include <string.h>
#include "config/ml_config.h"
#include "ops/ops.h"

/* Below this many weights a matvec finishes faster than threads can be woken
 * and joined, so it runs on the caller alone. */
#define PARALLEL_MIN_WEIGHTS (1 << 18)

typedef struct {
    float *y;
    const ml_qtensor *W;
    const ml_qact *x;
} matvec_ctx;

static void matvec_task(void *p, int tid, int nt) {
    const matvec_ctx *c = p;
    const int rows = c->W->rows;
    /* chunks start on a packed group (or 4-row) boundary */
    const int align = c->W->pq ? ML_PACK_G : 4;
    int chunk = (rows + nt - 1) / nt;
    chunk = (chunk + align - 1) / align * align;
    const int r0 = tid * chunk;
    const int r1 = r0 + chunk < rows ? r0 + chunk : rows;
    if (r0 >= r1) return;
#if ML_PACK_G
    if (c->W->pq) {
        if (c->W->type == ML_Q8_0) ml_k_matvec_q8_pk(c->y, c->W, c->x, r0, r1);
        else ml_k_matvec_q4_pk(c->y, c->W, c->x, r0, r1);
        return;
    }
#endif
    if (c->W->type == ML_Q8_0) ml_k_matvec_q8(c->y, c->W, c->x, r0, r1);
    else ml_k_matvec_q4(c->y, c->W, c->x, r0, r1);
}


void ml_matvec(ml_pool *pool, float *y, const ml_qtensor *W, const ml_qact *x) {
    matvec_ctx c = { y, W, x };
    if ((size_t)W->rows * W->cols < PARALLEL_MIN_WEIGHTS) matvec_task(&c, 0, 1);
    else ml_pool_run(pool, matvec_task, &c);
}

typedef struct {
    float *Y;
    int ldy;
    const ml_qtensor *W;
    const ml_qbatch *X;
} matmul_ctx;

static void matmul_task(void *p, int tid, int nt) {
    const matmul_ctx *c = p;
    const int rows = c->W->rows;
    const int align = c->W->pq ? ML_PACK_G : 4;
    int chunk = (rows + nt - 1) / nt;
    chunk = (chunk + align - 1) / align * align;
    const int r0 = tid * chunk;
    const int r1 = r0 + chunk < rows ? r0 + chunk : rows;
    if (r0 >= r1) return;
#if ML_PACK_G
    if (c->W->pq) {
        if (c->W->type == ML_Q8_0) ml_k_matmul_q8_pk(c->Y, c->ldy, c->W, c->X, r0, r1);
        else ml_k_matmul_q4_pk(c->Y, c->ldy, c->W, c->X, r0, r1);
        return;
    }
#endif
    for (int t = 0; t < c->X->n; t++) {
        const ml_qact xt = { c->X->q + (size_t)t * c->X->k, c->X->d + (size_t)t * (c->X->k / QK), c->X->k };
        float *y = c->Y + (size_t)t * c->ldy;
        if (c->W->type == ML_Q8_0) ml_k_matvec_q8(y, c->W, &xt, r0, r1);
        else ml_k_matvec_q4(y, c->W, &xt, r0, r1);
    }
}

void ml_matmul(ml_pool *pool, float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X) {
    matmul_ctx c = { Y, ldy, W, X };
    if ((size_t)W->rows * W->cols * X->n < PARALLEL_MIN_WEIGHTS) matmul_task(&c, 0, 1);
    else ml_pool_run(pool, matmul_task, &c);
}

void ml_quantize_row(int8_t *q, float *d, const float *x, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) amax = fmaxf(amax, fabsf(x[i]));
    if (amax < ML_QUANT_TINY) {
        *d = 0.0f;
        memset(q, 0, (size_t)n);
        return;
    }
    const float id = 127.0f / amax;
    *d = amax / 127.0f;
    for (int i = 0; i < n; i++) {
        float v = truncf(x[i] * id + copysignf(0.5f, x[i]));
        v = fminf(fmaxf(v, -127.0f), 127.0f);
        q[i] = (int8_t)v;
    }
}

typedef struct {
    float *out;
    const float *q;
    const ml_kv_layer *kv;
    float *att;
    const int8_t *qq;
    const float *qd;
    const ml_config *cfg;
    int pos;
} attn_ctx;

/* Each thread takes a contiguous range of query heads. Heads that share a KV
 * head (GQA) walk the cache together in blocks of ATT_BLOCK positions, so each
 * block of K and V is loaded into L1 once and reused by every head of the
 * group. Per head the arithmetic is that of one call over all positions: scores
 * are independent per position, and the mix adds positions in order. */
#define ATT_BLOCK 64

static void attention_task(void *p, int tid, int nt) {
    const attn_ctx *c = p;
    const ml_config *cfg = c->cfg;
    const ml_kv_layer *kv = c->kv;
    const int hd = cfg->head_dim, group = cfg->n_heads / cfg->n_kv_heads, n = c->pos + 1;
    const size_t ld = (size_t)cfg->kv_dim, ldd = (size_t)cfg->n_kv_heads, ldatt = (size_t)cfg->max_seq;
    const float scale = 1.0f / sqrtf((float)hd);
    const int h0 = (int)((long long)cfg->n_heads * tid / nt), h1 = (int)((long long)cfg->n_heads * (tid + 1) / nt);
    for (int g0 = h0; g0 < h1;) {
        const int kvh = g0 / group, gend = (kvh + 1) * group, g1 = gend < h1 ? gend : h1;   /* heads [g0, g1) share KV head kvh */
        const size_t off = (size_t)kvh * hd;
        for (int t0 = 0; t0 < n; t0 += ATT_BLOCK) {
            const int m = n - t0 < ATT_BLOCK ? n - t0 : ATT_BLOCK;
            for (int h = g0; h < g1; h++) {
                float *s = c->att + h * ldatt + t0;
                if (kv->k) ml_k_dot_f32(s, c->q + h * hd, kv->k + t0 * ld + off, ld, m, hd, scale);
                else ml_k_dot_q8(s, c->qq + h * hd, c->qd[h], kv->kq + t0 * ld + off, kv->kd + t0 * ldd + kvh, ld, ldd, m, hd, scale);
            }
        }
        for (int h = g0; h < g1; h++) {
            ml_k_softmax(c->att + h * ldatt, n);
            memset(c->out + h * hd, 0, sizeof(float) * hd);
        }
        for (int t0 = 0; t0 < n; t0 += ATT_BLOCK) {
            const int m = n - t0 < ATT_BLOCK ? n - t0 : ATT_BLOCK;
            for (int h = g0; h < g1; h++) {
                const float *a = c->att + h * ldatt + t0;
                if (kv->v) ml_k_axpy_f32(c->out + h * hd, a, kv->v + t0 * ld + off, ld, m, hd);
                else ml_k_axpy_q8(c->out + h * hd, a, kv->vq + t0 * ld + off, kv->vd + t0 * ldd + kvh, ld, ldd, m, hd);
            }
        }
        g0 = g1;
    }
}

void ml_attention(ml_pool *pool, float *out, const float *q, const ml_kv_layer *kv, float *att_scratch,
                  int8_t *qq, float *qd, const ml_config *cfg, int pos) {
    if (!kv->k)   /* Q8 cache: quantize each query head once, before the split */
        for (int h = 0; h < cfg->n_heads; h++)
            ml_quantize_row(qq + h * cfg->head_dim, qd + h, q + h * cfg->head_dim, cfg->head_dim);
    attn_ctx c = { out, q, kv, att_scratch, qq, qd, cfg, pos };
    /* Same cut-off as matvec, in multiply-adds: n_heads * (pos+1) * head_dim * 2. */
    if ((size_t)cfg->n_heads * (size_t)(pos + 1) * (size_t)cfg->head_dim * 2 < PARALLEL_MIN_WEIGHTS)
        attention_task(&c, 0, 1);
    else
        ml_pool_run(pool, attention_task, &c);
}
