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
    /* RoPE scaling (header offsets 52..71): ML_ROPE_NONE or ML_ROPE_LLAMA3 */
    int rope_type, rope_orig_max_seq;
    float rope_factor, rope_low_freq_factor, rope_high_freq_factor;
} ml_config;

enum { ML_ROPE_NONE = 0, ML_ROPE_LLAMA3 = 1 };

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
    void *pack_mem;   /* row-interleaved weight copies (ml_model_pack), or NULL */
} ml_model;

ml_status ml_model_load(ml_model *m, const char *path);
void ml_model_free(ml_model *m);

#endif
