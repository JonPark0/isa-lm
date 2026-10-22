/* Next-token sampling: greedy, temperature, top-p (nucleus). */
#ifndef ML_SAMPLER_H
#define ML_SAMPLER_H
#include "core/ml.h"

typedef struct { float p; int i; } ml_prob_index;

typedef struct {
    float temperature, topp;   /* temperature 0 = greedy */
    uint64_t rng;
    float *probs;
    ml_prob_index *cand;
} ml_sampler;

ml_status ml_sampler_init(ml_sampler *s, int vocab, float temperature, float topp, uint64_t seed);
/* Picks a token; may overwrite logits. Greedy ties go to the lowest index. */
int ml_sample(ml_sampler *s, float *logits, int n);
void ml_sampler_free(ml_sampler *s);

#endif
