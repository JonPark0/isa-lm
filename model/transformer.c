#include <math.h>
#include <string.h>
#include "config/ml_config.h"
#include "core/clock.h"
#include "core/quant_ref.h"
#include "model/transformer.h"
#include "ops/ops.h"

/* Inverse frequency of rotary pair i, with llama3 scaling: low frequencies
 * (wavelength above orig/low) are divided by factor, high ones (below
 * orig/high) kept, and the band between interpolated. Matches transformers. */
static double rope_inv_freq(const ml_config *c, int i) {
    const double inv = pow((double)c->rope_theta, -2.0 * i / c->head_dim);
    if (c->rope_type != ML_ROPE_LLAMA3) return inv;
    const double f = c->rope_factor, lo = c->rope_low_freq_factor, hi = c->rope_high_freq_factor;
    const double orig = c->rope_orig_max_seq, wavelen = 2.0 * 3.14159265358979323846 / inv;
    if (wavelen < orig / hi) return inv;
    if (wavelen > orig / lo) return inv / f;
    const double smooth = (orig / wavelen - lo) / (hi - lo);
    return (1.0 - smooth) * inv / f + smooth * inv;
}

const char *const ml_prof_names[ML_PROF_COUNT] = { "qkv", "attention", "wo", "ffn", "classifier", "other" };

/* Runs stmt, adding its wall time to prof[cat] when profiling is on. */
#define PROF(cat, stmt) do {                                                  \
        if (s->profile) { const double t0_ = ml_now(); stmt; s->prof[cat] += ml_now() - t0_; } \
        else { stmt; } } while (0)

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
        { 4, f * ML_BATCH_MAX * c->dim }, { 2, f * ML_BATCH_MAX * c->hidden_dim },
        { 1, (size_t)ML_BATCH_MAX * c->dim }, { 1, f * ML_BATCH_MAX * (c->dim / QK) },
        { 1, (size_t)ML_BATCH_MAX * c->hidden_dim }, { 1, f * ML_BATCH_MAX * (c->hidden_dim / QK) },
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
    const size_t B = ML_BATCH_MAX;
    s->bx = ml_arena_alloc(a, f * B * c->dim);
    s->bxb = ml_arena_alloc(a, f * B * c->dim);
    s->bxb2 = ml_arena_alloc(a, f * B * c->dim);
    s->bq = ml_arena_alloc(a, f * B * c->dim);
    s->bhb = ml_arena_alloc(a, f * B * c->hidden_dim);
    s->bhb2 = ml_arena_alloc(a, f * B * c->hidden_dim);
    s->qb_dim = (ml_qbatch){ ml_arena_alloc(a, B * c->dim), ml_arena_alloc(a, f * B * (c->dim / QK)), 0, c->dim };
    s->qb_hidden = (ml_qbatch){ ml_arena_alloc(a, B * c->hidden_dim),
                                ml_arena_alloc(a, f * B * (c->hidden_dim / QK)), 0, c->hidden_dim };
    s->batch = ML_BATCH_MAX;
    if (!s->qa_hidden.d || !s->qb_hidden.d) { ml_arena_free(a); return ML_E_NOMEM; }

    const int h2 = c->head_dim / 2;
    for (int i = 0; i < h2; i++) {
        const double freq = rope_inv_freq(c, i);
        for (int p = 0; p < c->max_seq; p++) {
            const double ang = p * freq;
            s->rope_cos[(size_t)p * h2 + i] = (float)cos(ang);
            s->rope_sin[(size_t)p * h2 + i] = (float)sin(ang);
        }
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

    PROF(ML_PROF_OTHER, ml_dequant_row(s->x, &m->tok_emb, token));
    for (int l = 0; l < c->n_layers; l++) {
        const ml_layer *L = &m->layers[l];
        float *kc = s->kcache + (size_t)l * c->max_seq * c->kv_dim;
        float *vc = s->vcache + (size_t)l * c->max_seq * c->kv_dim;
        float *k = kc + (size_t)pos * c->kv_dim, *v = vc + (size_t)pos * c->kv_dim;

        PROF(ML_PROF_OTHER, ml_k_rmsnorm(s->xb, s->x, L->attn_norm, c->dim, c->norm_eps);
                            ml_k_quantize_q8(&s->qa_dim, s->xb, c->dim));
        PROF(ML_PROF_QKV, ml_matvec(s->pool, s->q, &L->wq, &s->qa_dim);
                          ml_matvec(s->pool, k, &L->wk, &s->qa_dim);
                          ml_matvec(s->pool, v, &L->wv, &s->qa_dim));
        PROF(ML_PROF_OTHER, ml_k_rope(s->q, k, cs, sn, c->head_dim, c->n_heads, c->n_kv_heads));

        PROF(ML_PROF_ATTN, ml_attention(s->pool, s->xb, s->q, kc, vc, s->att, c, pos));
        PROF(ML_PROF_OTHER, ml_k_quantize_q8(&s->qa_dim, s->xb, c->dim));
        PROF(ML_PROF_WO, ml_matvec(s->pool, s->xb2, &L->wo, &s->qa_dim));
        PROF(ML_PROF_OTHER, ml_k_add(s->x, s->xb2, c->dim);
                            ml_k_rmsnorm(s->xb, s->x, L->ffn_norm, c->dim, c->norm_eps);
                            ml_k_quantize_q8(&s->qa_dim, s->xb, c->dim));
        PROF(ML_PROF_FFN, ml_matvec(s->pool, s->hb, &L->w1, &s->qa_dim);
                          ml_matvec(s->pool, s->hb2, &L->w3, &s->qa_dim));
        PROF(ML_PROF_OTHER, ml_k_silu_mul(s->hb, s->hb2, c->hidden_dim);
                            ml_k_quantize_q8(&s->qa_hidden, s->hb, c->hidden_dim));
        PROF(ML_PROF_FFN, ml_matvec(s->pool, s->xb2, &L->w2, &s->qa_hidden));
        PROF(ML_PROF_OTHER, ml_k_add(s->x, s->xb2, c->dim));
    }
    PROF(ML_PROF_OTHER, ml_k_rmsnorm(s->xb, s->x, m->final_norm, c->dim, c->norm_eps);
                        ml_k_quantize_q8(&s->qa_dim, s->xb, c->dim));
    PROF(ML_PROF_CLS, ml_matvec(s->pool, s->logits, &m->wcls, &s->qa_dim));
    return s->logits;
}

/* Quantizes rows [0, n) of src (each k floats) into the batch q. */
static void quantize_rows(ml_qbatch *q, const float *src, int n) {
    q->n = n;
    for (int t = 0; t < n; t++) {
        ml_qact qt = { q->q + (size_t)t * q->k, q->d + (size_t)t * (q->k / QK), q->k };
        ml_k_quantize_q8(&qt, src + (size_t)t * q->k, q->k);
    }
}

/* One batched step. Every per-token operation is the one ml_forward does,
 * in the same order, so logits match token-by-token decoding bit for bit;
 * only the matrix products are shared across the n tokens. */
static void prefill_step(ml_state *s, const int *tok, int n, int pos0, float *logits) {
    const ml_model *m = s->m;
    const ml_config *c = &m->cfg;
    const int D = c->dim, H = c->hidden_dim, KV = c->kv_dim, h2 = c->head_dim / 2;
    PROF(ML_PROF_OTHER, for (int t = 0; t < n; t++) ml_dequant_row(s->bx + (size_t)t * D, &m->tok_emb, tok[t]));
    for (int l = 0; l < c->n_layers; l++) {
        const ml_layer *L = &m->layers[l];
        float *kc = s->kcache + (size_t)l * c->max_seq * KV;
        float *vc = s->vcache + (size_t)l * c->max_seq * KV;
        PROF(ML_PROF_OTHER, for (int t = 0; t < n; t++)
                                ml_k_rmsnorm(s->bxb + (size_t)t * D, s->bx + (size_t)t * D, L->attn_norm, D, c->norm_eps);
                            quantize_rows(&s->qb_dim, s->bxb, n));
        PROF(ML_PROF_QKV, ml_matmul(s->pool, s->bq, D, &L->wq, &s->qb_dim);
                          ml_matmul(s->pool, kc + (size_t)pos0 * KV, KV, &L->wk, &s->qb_dim);
                          ml_matmul(s->pool, vc + (size_t)pos0 * KV, KV, &L->wv, &s->qb_dim));
        PROF(ML_PROF_OTHER, for (int t = 0; t < n; t++)
                                ml_k_rope(s->bq + (size_t)t * D, kc + (size_t)(pos0 + t) * KV,
                                          s->rope_cos + (size_t)(pos0 + t) * h2, s->rope_sin + (size_t)(pos0 + t) * h2,
                                          c->head_dim, c->n_heads, c->n_kv_heads));
        PROF(ML_PROF_ATTN, for (int t = 0; t < n; t++)
                               ml_attention(s->pool, s->bxb + (size_t)t * D, s->bq + (size_t)t * D, kc, vc, s->att, c, pos0 + t));
        PROF(ML_PROF_OTHER, quantize_rows(&s->qb_dim, s->bxb, n));
        PROF(ML_PROF_WO, ml_matmul(s->pool, s->bxb2, D, &L->wo, &s->qb_dim));
        PROF(ML_PROF_OTHER, for (int t = 0; t < n; t++) {
                                ml_k_add(s->bx + (size_t)t * D, s->bxb2 + (size_t)t * D, D);
                                ml_k_rmsnorm(s->bxb + (size_t)t * D, s->bx + (size_t)t * D, L->ffn_norm, D, c->norm_eps);
                            }
                            quantize_rows(&s->qb_dim, s->bxb, n));
        PROF(ML_PROF_FFN, ml_matmul(s->pool, s->bhb, H, &L->w1, &s->qb_dim);
                          ml_matmul(s->pool, s->bhb2, H, &L->w3, &s->qb_dim));
        PROF(ML_PROF_OTHER, for (int t = 0; t < n; t++)
                                ml_k_silu_mul(s->bhb + (size_t)t * H, s->bhb2 + (size_t)t * H, H);
                            quantize_rows(&s->qb_hidden, s->bhb, n));
        PROF(ML_PROF_FFN, ml_matmul(s->pool, s->bxb2, D, &L->w2, &s->qb_hidden));
        PROF(ML_PROF_OTHER, for (int t = 0; t < n; t++) ml_k_add(s->bx + (size_t)t * D, s->bxb2 + (size_t)t * D, D));
    }
    if (!logits) return;
    PROF(ML_PROF_OTHER, for (int t = 0; t < n; t++)
                            ml_k_rmsnorm(s->bxb + (size_t)t * D, s->bx + (size_t)t * D, m->final_norm, D, c->norm_eps);
                        quantize_rows(&s->qb_dim, s->bxb, n));
    PROF(ML_PROF_CLS, ml_matmul(s->pool, logits, c->vocab, &m->wcls, &s->qb_dim));
}

ml_status ml_prefill(ml_state *s, const int *tokens, int n, int pos0, float *logits) {
    const ml_config *c = &s->m->cfg;
    if (n < 0 || pos0 < 0 || pos0 + n > c->max_seq) return ML_E_SHAPE;
    for (int i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= c->vocab) return ML_E_SHAPE;
    const int B = s->batch < 1 ? 1 : (s->batch > ML_BATCH_MAX ? ML_BATCH_MAX : s->batch);
    for (int i = 0; i < n; i += B) {
        const int k = n - i < B ? n - i : B;
        prefill_step(s, tokens + i, k, pos0 + i, logits ? logits + (size_t)i * c->vocab : NULL);
    }
    return ML_OK;
}

void ml_state_free(ml_state *s) {
    if (s->pool) ml_pool_destroy(s->pool);
    ml_arena_free(&s->arena);
    s->pool = NULL;
}
