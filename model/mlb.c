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

static ml_qtensor take_mat(cursor *c, int rows, int cols, int type) {
    ml_qtensor t = { NULL, NULL, rows, cols, (uint8_t)type };
    size_t n, nd;
    if (!ml_mul_size((size_t)rows, (size_t)cols, &n) || !ml_mul_size(n / QK, sizeof(float), &nd)) {
        c->ok = 0;   /* cannot fit in this address space, let alone this file */
        return t;
    }
    t.q = take(c, type == ML_Q4_0 ? n / 2 : n);
    t.d = take(c, nd);
    /* Q8 quants must be in [-127, 127]: kernels negate weights by the
     * activation's sign, and -128 has no int8 negation. */
    if (c->ok && type == ML_Q8_0 && memchr(t.q, 0x80, n)) c->ok = 0;
    return t;
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
    if (cfg->dim % QK || cfg->hidden_dim % QK) return ML_E_SHAPE;
    if (cfg->dim % cfg->n_heads || cfg->n_heads % cfg->n_kv_heads) return ML_E_SHAPE;
    cfg->head_dim = cfg->dim / cfg->n_heads;
    cfg->kv_dim = cfg->head_dim * cfg->n_kv_heads;
    if (cfg->head_dim % 2) return ML_E_SHAPE;
    if (!(cfg->rope_theta > 0.0f) || !(cfg->norm_eps >= 0.0f)) return ML_E_FORMAT;
    cfg->rope_type = (int)rd32(h + 52);
    cfg->rope_factor = rdf(h + 56);
    cfg->rope_low_freq_factor = rdf(h + 60);
    cfg->rope_high_freq_factor = rdf(h + 64);
    cfg->rope_orig_max_seq = (int)rd32(h + 68);
    if (cfg->rope_type == ML_ROPE_LLAMA3) {
        if (!(cfg->rope_factor > 0.0f && cfg->rope_factor < 1e6f) || !(cfg->rope_low_freq_factor > 0.0f) ||
            !(cfg->rope_high_freq_factor > cfg->rope_low_freq_factor && cfg->rope_high_freq_factor < 1e6f) ||
            cfg->rope_orig_max_seq <= 0)
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
    const int wt = c->wtype;
    m->layers = calloc((size_t)c->n_layers, sizeof(ml_layer));
    if (!m->layers) { ml_model_free(m); return ML_E_NOMEM; }

    cursor cur = { m->file.data, ML_MLB_HEADER_SIZE, m->file.size, 1 };
    const size_t normsz = (size_t)c->dim * sizeof(float);
    m->tok_emb = take_mat(&cur, c->vocab, c->dim, wt);
    for (int l = 0; l < c->n_layers; l++) {
        ml_layer *L = &m->layers[l];
        L->attn_norm = take(&cur, normsz);
        L->wq = take_mat(&cur, c->dim, c->dim, wt);
        L->wk = take_mat(&cur, c->kv_dim, c->dim, wt);
        L->wv = take_mat(&cur, c->kv_dim, c->dim, wt);
        L->wo = take_mat(&cur, c->dim, c->dim, wt);
        L->ffn_norm = take(&cur, normsz);
        L->w1 = take_mat(&cur, c->hidden_dim, c->dim, wt);
        L->w2 = take_mat(&cur, c->dim, c->hidden_dim, wt);
        L->w3 = take_mat(&cur, c->hidden_dim, c->dim, wt);
    }
    m->final_norm = take(&cur, normsz);
    m->wcls = shared ? m->tok_emb : take_mat(&cur, c->vocab, c->dim, wt);
    if (!cur.ok) { ml_model_free(m); return ML_E_FORMAT; }
    return ML_OK;
}

void ml_model_free(ml_model *m) {
    free(m->layers);
    m->layers = NULL;
    ml_file_unmap(&m->file);
}
