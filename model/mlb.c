#include <stdlib.h>
#include <string.h>
#include "model/mlb.h"

#define MAX_DIM (1 << 16)   /* bound on every header dimension: keeps sizes far from overflow */

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
    size_t n = (size_t)rows * cols;
    t.q = take(c, type == ML_Q4_0 ? n / 2 : n);
    t.d = take(c, n / QK * sizeof(float));
    return t;
}

static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static float rdf(const uint8_t *p) { float v; memcpy(&v, p, 4); return v; }

static ml_status parse_header(ml_config *cfg, int *shared, const uint8_t *h) {
    if (memcmp(h, "MLB1", 4) != 0 || rd32(h + 4) != 1) return ML_E_FORMAT;
    uint32_t v[7];
    for (int i = 0; i < 7; i++) v[i] = rd32(h + 8 + 4 * i);
    for (int i = 0; i < 7; i++) if (v[i] == 0 || v[i] > MAX_DIM) return ML_E_SHAPE;
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
