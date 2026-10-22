/* .mlb model file: header + 64 B-aligned tensor arrays, mapped zero-copy. */
#ifndef ML_MLB_H
#define ML_MLB_H
#include "core/file.h"
#include "core/ml.h"

#define ML_MLB_HEADER_SIZE 256

typedef struct {
    int dim, n_layers, n_heads, n_kv_heads, hidden_dim, vocab, max_seq, head_dim, kv_dim;
    float rope_theta, norm_eps;
    int wtype;
} ml_config;

typedef struct {
    const float *attn_norm, *ffn_norm;
    ml_qtensor wq, wk, wv, wo, w1, w2, w3;
} ml_layer;

typedef struct {
    ml_config cfg;
    ml_file file;
    ml_qtensor tok_emb, wcls;
    const float *final_norm;
    ml_layer *layers;
} ml_model;

ml_status ml_model_load(ml_model *m, const char *path);
void ml_model_free(ml_model *m);

#endif
