/* Decoder forward pass. All runtime buffers come from one arena. */
#ifndef ML_TRANSFORMER_H
#define ML_TRANSFORMER_H
#include "core/arena.h"
#include "core/threads.h"
#include "model/mlb.h"

/* Stages timed by ml_state.profile (seconds accumulated in ml_state.prof). */
enum { ML_PROF_QKV, ML_PROF_ATTN, ML_PROF_WO, ML_PROF_FFN, ML_PROF_CLS, ML_PROF_OTHER, ML_PROF_COUNT };
extern const char *const ml_prof_names[ML_PROF_COUNT];

typedef struct {
    ml_model *m;
    ml_arena arena;
    ml_pool *pool;
    float *x, *xb, *xb2, *hb, *hb2, *q, *logits;
    float *att;                    /* scores [max(n_heads, threads * heads per KV head)][max_seq] */
    int kv_type;                   /* ML_F32 or ML_Q8_0 (see ml_state_init) */
    float *kcache, *vcache;        /* F32: [layer][max_seq][kv_dim] */
    int8_t *kq, *vq;               /* Q8: [layer][max_seq][kv_dim] */
    float *kd, *vd;                /* Q8: [layer][max_seq][n_kv_heads] scales */
    float *kbuf, *vbuf;            /* Q8: this step's K and V before quantization, [ML_BATCH_MAX][kv_dim] */
    int8_t *qq;                    /* Q8: quantized query heads [ML_BATCH_MAX][dim] */
    float *qd;                     /* Q8: their scales [ML_BATCH_MAX][n_heads] */
    float *rope_cos, *rope_sin;    /* [max_seq][head_dim/2] */
    ml_qact qa_dim, qa_hidden;
    /* prompt prefill: up to ML_BATCH_MAX tokens per batched step */
    int batch;                     /* tokens per step, 1..ML_BATCH_MAX (default max) */
    float *bx, *bxb, *bxb2, *bq, *bhb, *bhb2;   /* [ML_BATCH_MAX][dim or hidden] */
    ml_qbatch qb_dim, qb_hidden;
    int profile;                   /* set to time each stage of ml_forward */
    double prof[ML_PROF_COUNT];
    /* importance matrix (ml_imatrix_enable): per channel, the sum of squares
     * of each matrix input over the tokens run, see ML_IMAT_* */
    double *imat;
    unsigned long long imat_tokens, imat_cls_tokens;   /* tokens summed into the layers, the classifier */
} ml_state;

/* imat layout: per layer [dim] wq/wk/wv input, [dim] wo, [dim] w1/w3,
 * [hidden_dim] w2; then [dim] classifier input. */
enum { ML_IMAT_QKV, ML_IMAT_WO, ML_IMAT_FFN, ML_IMAT_DOWN };
size_t ml_imatrix_size(const ml_config *c);

#define ML_BATCH_MAX 32

/* kv_type: ML_F32 keeps the KV cache in float; ML_Q8_0 stores each position's
 * K and V per KV head as int8 with one scale (about 3.8x less memory, faster
 * attention on long contexts, slightly different logits). */
ml_status ml_state_init(ml_state *s, ml_model *m, int nthreads, int kv_type);
/* Runs n tokens at positions pos0.. through the model in batches, filling the
 * KV cache. If logits is non-NULL it receives [n][vocab] logits; otherwise
 * the classifier is skipped. ML_E_SHAPE if a position or token is out of range. */
ml_status ml_prefill(ml_state *s, const int *tokens, int n, int pos0, float *logits);
/* Logits for `token` at position `pos`; NULL if pos or token is out of range. */
float *ml_forward(ml_state *s, int token, int pos);
/* Starts summing the importance matrix (zeroed) from the next token on. */
ml_status ml_imatrix_enable(ml_state *s);
void ml_state_free(ml_state *s);

#endif
