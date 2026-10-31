#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "model/mlb.h"

#define MAX_DIM (1 << 16)    /* dim, hidden_dim, n_layers, heads (SVE kernels size per-row scratch by it) */
#define MAX_TOKENS (1 << 24) /* vocab, max_seq */

typedef struct {
    const uint8_t *base;
    size_t off, size;
    int ok;
} cursor;

static const void *take(cursor *c, size_t nbytes) {
    if (!c->ok || nbytes > c->size - c->off) { c->ok = 0; return NULL; }
    const void *p = c->base + c->off;
    c->off += nbytes;
    c->off = (c->off + ML_ALIGN - 1) & ~(size_t)(ML_ALIGN - 1);
    if (c->off > c->size) c->off = c->size;   /* tolerate missing final padding */
    return p;
}

static ml_qtensor take_mat(cursor *c, int rows, int cols, int type, int f16) {
    ml_qtensor t = { NULL, NULL, rows, cols, (uint8_t)type, NULL, NULL, NULL };
    size_t n, nd;
    if (!ml_mul_size((size_t)rows, (size_t)cols, &n) || !ml_mul_size(n / QK, f16 ? 2 : sizeof(float), &nd)) {
        c->ok = 0;   /* cannot fit in this address space, let alone this file */
        return t;
    }
    t.q = take(c, type == ML_Q4_0 ? n / 2 : n);
    if (f16) t.dh = take(c, nd);
    else t.d = take(c, nd);
    /* Q8 quants must be in [-127, 127]: kernels negate weights by the
     * activation's sign, and -128 has no int8 negation. */
    if (c->ok && type == ML_Q8_0 && memchr(t.q, 0x80, n)) c->ok = 0;
    return t;
}

/* f16-scale files: converts the scales of the matrices that need floats into
 * one buffer, the embedding table's always, the others' when all is set or
 * they have no packed copy (the rest get d = NULL). The classifier shares
 * the table's when it is the table. */
static ml_status expand_scales(ml_model *m, int all) {
    const int nl = m->cfg.n_layers, shared = m->wcls.dh == m->tok_emb.dh;
    ml_qtensor **ws = malloc(sizeof(ml_qtensor *) * ((size_t)nl * 7 + 2));
    if (!ws) return ML_E_NOMEM;
    int n = 0;
    ws[n++] = &m->tok_emb;
    for (int l = 0; l < nl; l++) {
        ml_layer *L = &m->layers[l];
        ml_qtensor *lw[7] = { &L->wq, &L->wk, &L->wv, &L->wo, &L->w1, &L->w2, &L->w3 };
        for (int k = 0; k < 7; k++) ws[n++] = lw[k];
    }
    if (!shared) ws[n++] = &m->wcls;
    size_t total = 0;
    for (int i = 0; i < n; i++)
        if (i == 0 || all || !ws[i]->pq) total += (size_t)ws[i]->rows * (ws[i]->cols / QK);
    float *buf = ml_aligned_alloc(total * sizeof(float));
    if (!buf) { free(ws); return ML_E_NOMEM; }
    size_t off = 0;
    for (int i = 0; i < n; i++) {
        ml_qtensor *W = ws[i];
        if (i && !all && W->pq) { W->d = NULL; continue; }
        const size_t nb = (size_t)W->rows * (W->cols / QK);
        for (size_t j = 0; j < nb; j++) buf[off + j] = ml_f16_to_f32(W->dh[j]);
        W->d = buf + off;
        off += nb;
    }
    if (shared) m->wcls.d = m->tok_emb.d;
    free(ws);
    ml_aligned_free(m->scales);
    m->scales = buf;
    return ML_OK;
}

static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static float rdf(const uint8_t *p) { float v; memcpy(&v, p, 4); return v; }

static ml_status parse_header(ml_config *cfg, int *shared, const uint8_t *h) {
    if (memcmp(h, "MLB1", 4) != 0 || rd32(h + 4) != 1) return ML_E_FORMAT;
    uint32_t v[7];
    for (int i = 0; i < 7; i++) v[i] = rd32(h + 8 + 4 * i);
    for (int i = 0; i < 7; i++) if (v[i] == 0 || v[i] > (i == 5 || i == 6 ? MAX_TOKENS : MAX_DIM)) return ML_E_SHAPE;
    if (v[5] < 3) return ML_E_SHAPE;   /* ids 1 and 2 are BOS / EOS */
    cfg->dim = (int)v[0]; cfg->n_layers = (int)v[1]; cfg->n_heads = (int)v[2];
    cfg->n_kv_heads = (int)v[3]; cfg->hidden_dim = (int)v[4]; cfg->vocab = (int)v[5];
    cfg->max_seq = (int)v[6];
    cfg->rope_theta = rdf(h + 36);
    cfg->norm_eps = rdf(h + 40);
    cfg->wtype = (int)rd32(h + 44);
    *shared = rd32(h + 48) != 0;
    if (cfg->wtype != ML_Q8_0 && cfg->wtype != ML_Q4_0) return ML_E_FORMAT;
    cfg->q8_mask = rd32(h + 72);
    if ((cfg->q8_mask & ~(unsigned)ML_Q8M_ALL) || (cfg->wtype == ML_Q8_0 && cfg->q8_mask)) return ML_E_FORMAT;
    const uint32_t bias = rd32(h + 76);
    if (bias > 1) return ML_E_FORMAT;
    cfg->qkv_bias = (int)bias;
    if (cfg->dim % QK || cfg->hidden_dim % QK) return ML_E_SHAPE;
    if (cfg->n_heads % cfg->n_kv_heads) return ML_E_SHAPE;
    const uint32_t hd = rd32(h + 80);   /* 0: dim / n_heads */
    if (hd > MAX_DIM || (!hd && cfg->dim % cfg->n_heads)) return ML_E_SHAPE;
    cfg->head_dim = hd ? (int)hd : cfg->dim / cfg->n_heads;
    if ((uint64_t)cfg->head_dim * (uint64_t)cfg->n_heads > MAX_DIM) return ML_E_SHAPE;
    cfg->q_dim = cfg->head_dim * cfg->n_heads;
    cfg->kv_dim = cfg->head_dim * cfg->n_kv_heads;
    if (cfg->head_dim % 2 || cfg->q_dim % QK) return ML_E_SHAPE;
    const uint32_t qkn = rd32(h + 84), win = rd32(h + 92);
    if (qkn > 1 || win > MAX_TOKENS) return ML_E_FORMAT;
    cfg->qk_norm = (int)qkn;
    cfg->sliding_window = (int)win;
    const uint32_t lflags = rd32(h + 96);
    if (lflags > 1) return ML_E_FORMAT;
    cfg->layer_flags = (int)lflags;
    cfg->rope_local_theta = rdf(h + 100);
    if (!(cfg->rope_local_theta >= 0.0f && cfg->rope_local_theta < 1e30f)) return ML_E_FORMAT;
    const uint32_t act = rd32(h + 104), pn = rd32(h + 108);
    if (act > ML_ACT_GELU_TANH || pn > 1) return ML_E_FORMAT;
    cfg->act = (int)act;
    cfg->post_norms = (int)pn;
    cfg->embed_scale = rdf(h + 112);
    cfg->attn_scale = rdf(h + 116);
    if (!(cfg->embed_scale >= 0.0f && cfg->embed_scale < 1e6f) || !(cfg->attn_scale >= 0.0f && cfg->attn_scale < 1e6f))
        return ML_E_FORMAT;
    if (cfg->embed_scale == 0.0f) cfg->embed_scale = 1.0f;
    if (cfg->attn_scale == 0.0f) cfg->attn_scale = 1.0f / sqrtf((float)cfg->head_dim);
    const uint32_t f16 = rd32(h + 120);
    if (f16 > 1) return ML_E_FORMAT;
    cfg->scale_f16 = (int)f16;
    if (!(cfg->rope_theta > 0.0f) || !(cfg->norm_eps >= 0.0f)) return ML_E_FORMAT;
    cfg->rope_type = (int)rd32(h + 52);
    cfg->rope_factor = rdf(h + 56);
    cfg->rope_low_freq_factor = rdf(h + 60);
    cfg->rope_high_freq_factor = rdf(h + 64);
    cfg->rope_orig_max_seq = (int)rd32(h + 68);
    cfg->rope_attn_factor = rdf(h + 88);
    if (cfg->rope_type != ML_ROPE_YARN && cfg->rope_attn_factor != 0.0f) return ML_E_FORMAT;
    if (cfg->rope_type == ML_ROPE_LLAMA3) {
        if (!(cfg->rope_factor > 0.0f && cfg->rope_factor < 1e6f) || !(cfg->rope_low_freq_factor > 0.0f) ||
            !(cfg->rope_high_freq_factor > cfg->rope_low_freq_factor && cfg->rope_high_freq_factor < 1e6f) ||
            cfg->rope_orig_max_seq <= 0)
            return ML_E_FORMAT;
    } else if (cfg->rope_type == ML_ROPE_LINEAR) {
        if (!(cfg->rope_factor > 0.0f && cfg->rope_factor < 1e6f)) return ML_E_FORMAT;
    } else if (cfg->rope_type == ML_ROPE_YARN) {
        /* beta_fast > beta_slow > 0 rotations within the original context */
        if (!(cfg->rope_factor > 0.0f && cfg->rope_factor < 1e6f) || !(cfg->rope_high_freq_factor > 0.0f) ||
            !(cfg->rope_low_freq_factor > cfg->rope_high_freq_factor && cfg->rope_low_freq_factor < 1e6f) ||
            cfg->rope_orig_max_seq <= 0 || !(cfg->rope_attn_factor > 0.0f && cfg->rope_attn_factor < 1e3f))
            return ML_E_FORMAT;
    } else if (cfg->rope_type != ML_ROPE_NONE) {
        return ML_E_FORMAT;
    }
    return ML_OK;
}

ml_status ml_model_load(ml_model *m, const char *path) {
    memset(m, 0, sizeof *m);
    ml_status s = ml_file_map(&m->file, path);
    if (s != ML_OK) return s;
    int shared = 0;
    if (m->file.size < ML_MLB_HEADER_SIZE) s = ML_E_FORMAT;
    else s = parse_header(&m->cfg, &shared, m->file.data);
    if (s != ML_OK) { ml_model_free(m); return s; }

    const ml_config *c = &m->cfg;
    /* storage type of a matrix class: Q8 if the model is, or if its bit is set */
#define WT(bits) (c->wtype == ML_Q8_0 || (c->q8_mask & (bits)) ? ML_Q8_0 : ML_Q4_0)
    m->layers = calloc((size_t)c->n_layers, sizeof(ml_layer));
    if (!m->layers) { ml_model_free(m); return ML_E_NOMEM; }

    cursor cur = { m->file.data, ML_MLB_HEADER_SIZE, m->file.size, 1 };
    const size_t normsz = (size_t)c->dim * sizeof(float);
    const uint8_t *lf = c->layer_flags ? take(&cur, (size_t)c->n_layers) : NULL;
    for (int l = 0; lf && cur.ok && l < c->n_layers; l++)
        if ((lf[l] & ~ML_LAYER_ALL) || ((lf[l] & ML_LAYER_LOCAL_ROPE) && !(c->rope_local_theta > 0.0f))) {
            ml_model_free(m);
            return ML_E_FORMAT;
        }
    m->tok_emb = take_mat(&cur, c->vocab, c->dim, WT(ML_Q8M_EMB | (shared ? ML_Q8M_CLS : 0)), c->scale_f16);
    for (int l = 0; l < c->n_layers; l++) {
        ml_layer *L = &m->layers[l];
        L->window = !lf || (lf[l] & ML_LAYER_SLIDING) ? c->sliding_window : 0;
        L->local_rope = lf && (lf[l] & ML_LAYER_LOCAL_ROPE);
        L->attn_norm = take(&cur, normsz);
        L->wq = take_mat(&cur, c->q_dim, c->dim, WT(ML_Q8M_QK), c->scale_f16);
        L->wk = take_mat(&cur, c->kv_dim, c->dim, WT(ML_Q8M_QK), c->scale_f16);
        L->wv = take_mat(&cur, c->kv_dim, c->dim, WT(ML_Q8M_V), c->scale_f16);
        L->wo = take_mat(&cur, c->dim, c->q_dim, WT(ML_Q8M_O), c->scale_f16);
        if (c->qkv_bias) {
            L->bq = take(&cur, (size_t)c->q_dim * sizeof(float));
            L->bk = take(&cur, (size_t)c->kv_dim * sizeof(float));
            L->bv = take(&cur, (size_t)c->kv_dim * sizeof(float));
        }
        if (c->qk_norm) {
            L->q_norm = take(&cur, (size_t)c->head_dim * sizeof(float));
            L->k_norm = take(&cur, (size_t)c->head_dim * sizeof(float));
        }
        L->ffn_norm = take(&cur, normsz);
        L->w1 = take_mat(&cur, c->hidden_dim, c->dim, WT(ML_Q8M_UP), c->scale_f16);
        L->w2 = take_mat(&cur, c->dim, c->hidden_dim, WT(ML_Q8M_DOWN), c->scale_f16);
        L->w3 = take_mat(&cur, c->hidden_dim, c->dim, WT(ML_Q8M_UP), c->scale_f16);
        if (c->post_norms) {
            L->post_attn_norm = take(&cur, normsz);
            L->post_ffn_norm = take(&cur, normsz);
        }
    }
    m->final_norm = take(&cur, normsz);
    m->wcls = shared ? m->tok_emb : take_mat(&cur, c->vocab, c->dim, WT(ML_Q8M_CLS), c->scale_f16);
#undef WT
    if (!cur.ok) { ml_model_free(m); return ML_E_FORMAT; }
    if (c->scale_f16 && (s = expand_scales(m, 1)) != ML_OK) { ml_model_free(m); return s; }
    return ML_OK;
}

ml_status ml_model_trim_scales(ml_model *m) {
    return m->cfg.scale_f16 ? expand_scales(m, 0) : ML_OK;
}

void ml_model_free(ml_model *m) {
    ml_aligned_free(m->pack_mem);
    m->pack_mem = NULL;
    ml_aligned_free(m->scales);
    m->scales = NULL;
    if (m->pack_file.data) ml_file_unmap(&m->pack_file);
    m->packed = 0;
    free(m->layers);
    m->layers = NULL;
    ml_file_unmap(&m->file);
}
