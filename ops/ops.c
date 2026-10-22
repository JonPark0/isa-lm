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

typedef struct {
    float *out;
    const float *q, *kc, *vc;
    float *att;
    const ml_config *cfg;
    int pos;
} attn_ctx;

static void attention_task(void *p, int tid, int nt) {
    const attn_ctx *c = p;
    const ml_config *cfg = c->cfg;
    const int hd = cfg->head_dim, group = cfg->n_heads / cfg->n_kv_heads;
    const float scale = 1.0f / sqrtf((float)hd);
    for (int h = tid; h < cfg->n_heads; h += nt) {
        const float *qh = c->q + h * hd;
        const int off = (h / group) * hd;
        float *att = c->att + (size_t)h * cfg->max_seq;
        for (int t = 0; t <= c->pos; t++)
            att[t] = ml_k_dot_f32(qh, c->kc + (size_t)t * cfg->kv_dim + off, hd) * scale;
        ml_k_softmax(att, c->pos + 1);
        float *oh = c->out + h * hd;
        memset(oh, 0, sizeof(float) * hd);
        for (int t = 0; t <= c->pos; t++)
            ml_k_axpy_f32(oh, att[t], c->vc + (size_t)t * cfg->kv_dim + off, hd);
    }
}

void ml_attention(ml_pool *pool, float *out, const float *q, const float *kcache,
                  const float *vcache, float *att_scratch, const ml_config *cfg, int pos) {
    attn_ctx c = { out, q, kcache, vcache, att_scratch, cfg, pos };
    /* Same cut-off as matvec, in multiply-adds: n_heads * (pos+1) * head_dim * 2. */
    if ((size_t)cfg->n_heads * (size_t)(pos + 1) * (size_t)cfg->head_dim * 2 < PARALLEL_MIN_WEIGHTS)
        attention_task(&c, 0, 1);
    else
        ml_pool_run(pool, attention_task, &c);
}
