#include <math.h>
#include <string.h>
#include "config/ml_config.h"
#include "core/quant_ref.h"
#include "model/transformer.h"
#include "ops/ops.h"

#define RUP(n) (((size_t)(n) + ML_ALIGN - 1) & ~(size_t)(ML_ALIGN - 1))

ml_status ml_state_init(ml_state *s, ml_model *m, int nthreads) {
    memset(s, 0, sizeof *s);
    s->m = m;
    const ml_config *c = &m->cfg;
    const size_t f = sizeof(float);
    size_t lkv, kv, half, need = 0;
    /* Sizes are products of header fields: check them, a 32-bit size_t wraps. */
    if (!ml_mul_size((size_t)c->n_layers, (size_t)c->kv_dim, &lkv) ||
        !ml_mul_size(lkv, (size_t)c->max_seq, &kv) || kv > (size_t)-1 / f ||
        !ml_mul_size((size_t)c->max_seq, (size_t)(c->head_dim / 2), &half) || half > (size_t)-1 / f)
        return ML_E_NOMEM;
    /* {count, bytes}: every runtime buffer; each entry is at most 4 arrays,
     * each padded to ML_ALIGN by the arena. */
    const size_t parts[][2] = {
        { 4, f * c->dim }, { 2, f * c->hidden_dim }, { 1, f * c->vocab },
        { (size_t)c->n_heads, f * c->max_seq }, { 2, f * kv }, { 2, f * half },
        { 1, (size_t)c->dim }, { 1, f * c->dim / QK }, { 1, (size_t)c->hidden_dim }, { 1, f * c->hidden_dim / QK },
    };
    for (size_t i = 0; i < sizeof parts / sizeof parts[0]; i++) {
        size_t bytes;
        if (!ml_mul_size(parts[i][0], parts[i][1], &bytes) || bytes > (size_t)-1 - need - 4 * ML_ALIGN)
            return ML_E_NOMEM;
        need += bytes + 4 * ML_ALIGN;
    }
    ml_status st = ml_arena_init(&s->arena, need);
    if (st != ML_OK) return st;
    ml_arena *a = &s->arena;
    s->x = ml_arena_alloc(a, f * c->dim);
    s->xb = ml_arena_alloc(a, f * c->dim);
    s->xb2 = ml_arena_alloc(a, f * c->dim);
    s->hb = ml_arena_alloc(a, f * c->hidden_dim);
    s->hb2 = ml_arena_alloc(a, f * c->hidden_dim);
    s->q = ml_arena_alloc(a, f * c->dim);
    s->att = ml_arena_alloc(a, f * c->n_heads * c->max_seq);
    s->logits = ml_arena_alloc(a, f * c->vocab);
    s->kcache = ml_arena_alloc(a, f * kv);
    s->vcache = ml_arena_alloc(a, f * kv);
    s->rope_cos = ml_arena_alloc(a, f * half);
    s->rope_sin = ml_arena_alloc(a, f * half);
    s->qa_dim = (ml_qact){ ml_arena_alloc(a, c->dim), ml_arena_alloc(a, f * c->dim / QK), c->dim };
    s->qa_hidden = (ml_qact){ ml_arena_alloc(a, c->hidden_dim),
                              ml_arena_alloc(a, f * c->hidden_dim / QK), c->hidden_dim };
    if (!s->qa_hidden.d) { ml_arena_free(a); return ML_E_NOMEM; }

    const int h2 = c->head_dim / 2;
    for (int p = 0; p < c->max_seq; p++)
        for (int i = 0; i < h2; i++) {
            const double ang = p * pow((double)c->rope_theta, -2.0 * i / c->head_dim);
            s->rope_cos[(size_t)p * h2 + i] = (float)cos(ang);
            s->rope_sin[(size_t)p * h2 + i] = (float)sin(ang);
        }
    st = ml_pool_create(&s->pool, nthreads);
    if (st != ML_OK) ml_arena_free(a);
    return st;
}

float *ml_forward(ml_state *s, int token, int pos) {
    const ml_model *m = s->m;
    const ml_config *c = &m->cfg;
    if (pos < 0 || pos >= c->max_seq || token < 0 || token >= c->vocab) return NULL;
    const int h2 = c->head_dim / 2;
    const float *cs = s->rope_cos + (size_t)pos * h2, *sn = s->rope_sin + (size_t)pos * h2;

    ml_dequant_row(s->x, &m->tok_emb, token);
    for (int l = 0; l < c->n_layers; l++) {
        const ml_layer *L = &m->layers[l];
        float *kc = s->kcache + (size_t)l * c->max_seq * c->kv_dim;
        float *vc = s->vcache + (size_t)l * c->max_seq * c->kv_dim;
        float *k = kc + (size_t)pos * c->kv_dim, *v = vc + (size_t)pos * c->kv_dim;

        ml_k_rmsnorm(s->xb, s->x, L->attn_norm, c->dim, c->norm_eps);
        ml_k_quantize_q8(&s->qa_dim, s->xb, c->dim);
        ml_matvec(s->pool, s->q, &L->wq, &s->qa_dim);
        ml_matvec(s->pool, k, &L->wk, &s->qa_dim);
        ml_matvec(s->pool, v, &L->wv, &s->qa_dim);
        ml_k_rope(s->q, k, cs, sn, c->head_dim, c->n_heads, c->n_kv_heads);

        ml_attention(s->pool, s->xb, s->q, kc, vc, s->att, c, pos);
        ml_k_quantize_q8(&s->qa_dim, s->xb, c->dim);
        ml_matvec(s->pool, s->xb2, &L->wo, &s->qa_dim);
        ml_k_add(s->x, s->xb2, c->dim);

        ml_k_rmsnorm(s->xb, s->x, L->ffn_norm, c->dim, c->norm_eps);
        ml_k_quantize_q8(&s->qa_dim, s->xb, c->dim);
        ml_matvec(s->pool, s->hb, &L->w1, &s->qa_dim);
        ml_matvec(s->pool, s->hb2, &L->w3, &s->qa_dim);
        ml_k_silu_mul(s->hb, s->hb2, c->hidden_dim);
        ml_k_quantize_q8(&s->qa_hidden, s->hb, c->hidden_dim);
        ml_matvec(s->pool, s->xb2, &L->w2, &s->qa_hidden);
        ml_k_add(s->x, s->xb2, c->dim);
    }
    ml_k_rmsnorm(s->xb, s->x, m->final_norm, c->dim, c->norm_eps);
    ml_k_quantize_q8(&s->qa_dim, s->xb, c->dim);
    ml_matvec(s->pool, s->logits, &m->wcls, &s->qa_dim);
    return s->logits;
}

void ml_state_free(ml_state *s) {
    if (s->pool) ml_pool_destroy(s->pool);
    ml_arena_free(&s->arena);
    s->pool = NULL;
}
