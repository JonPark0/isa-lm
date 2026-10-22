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
    float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits;
    float *kcache, *vcache;        /* [layer][max_seq][kv_dim] */
    float *rope_cos, *rope_sin;    /* [max_seq][head_dim/2] */
    ml_qact qa_dim, qa_hidden;
    int profile;                   /* set to time each stage of ml_forward */
    double prof[ML_PROF_COUNT];
} ml_state;

ml_status ml_state_init(ml_state *s, ml_model *m, int nthreads);
/* Logits for `token` at position `pos`; NULL if pos or token is out of range. */
float *ml_forward(ml_state *s, int token, int pos);
void ml_state_free(ml_state *s);

#endif
