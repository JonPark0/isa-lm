/* Next-token sampling: repetition penalty, then greedy, or temperature with
 * top-k and top-p (nucleus). */
#ifndef ML_SAMPLER_H
#define ML_SAMPLER_H
#include "core/ml.h"

typedef struct { float p; int i; } ml_prob_index;

typedef struct {
    float temperature;     /* 0 = greedy */
    float topp;            /* 0 or 1 = off */
    int topk;              /* 0 = off */
    float repeat_penalty;  /* 1 = off; > 1 discourages tokens seen recently */
    int repeat_last_n;     /* how many recent tokens the penalty looks at */
    uint64_t seed;
} ml_sampler_opts;

typedef struct {
    ml_sampler_opts o;
    uint64_t rng;
    int vocab;
    float *probs;
    ml_prob_index *cand;
    int *recent, n_recent, recent_at;   /* ring of the last repeat_last_n tokens */
    int *mark;                          /* per-token stamp: penalize each token once */
    int stamp;
} ml_sampler;

ml_status ml_sampler_init(ml_sampler *s, int vocab, const ml_sampler_opts *o);
/* Records a token of the context (prompt or generated) for the penalty. */
void ml_sampler_accept(ml_sampler *s, int token);
/* Picks a token; may overwrite logits. Greedy ties go to the lowest index. */
int ml_sample(ml_sampler *s, float *logits, int n);
void ml_sampler_free(ml_sampler *s);

#endif
