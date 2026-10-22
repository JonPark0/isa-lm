#include <stdlib.h>
#include <string.h>
#include "kernels/scalar/k_scalar.h"
#include "model/sampler.h"

ml_status ml_sampler_init(ml_sampler *s, int vocab, float temperature, float topp, uint64_t seed) {
    s->temperature = temperature;
    s->topp = topp;
    s->rng = seed ? seed : 0x9E3779B97F4A7C15ull;
    s->probs = malloc(sizeof(float) * vocab);
    s->cand = malloc(sizeof(ml_prob_index) * vocab);
    if (!s->probs || !s->cand) { ml_sampler_free(s); return ML_E_NOMEM; }
    return ML_OK;
}

void ml_sampler_free(ml_sampler *s) {
    free(s->probs);
    free(s->cand);
    s->probs = NULL;
    s->cand = NULL;
}

static float random_f32(uint64_t *st) {   /* xorshift64*, uniform in [0, 1) */
    *st ^= *st >> 12;
    *st ^= *st << 25;
    *st ^= *st >> 27;
    return (float)((uint32_t)((*st * 0x2545F4914F6CDD1Dull) >> 32) >> 8) / 16777216.0f;
}

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

static int sample_mult(const float *p, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += p[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

static int cmp_desc(const void *a, const void *b) {
    const ml_prob_index *x = a, *y = b;
    if (x->p > y->p) return -1;
    if (x->p < y->p) return 1;
    return x->i - y->i;
}

static int sample_topp(const float *p, int n, float topp, ml_prob_index *cand, float coin) {
    /* Tokens below (1 - topp) / (n - 1) can never be in the nucleus. */
    const float cutoff = n > 1 ? (1.0f - topp) / (float)(n - 1) : 0.0f;
    int m = 0;
    for (int i = 0; i < n; i++)
        if (p[i] >= cutoff) cand[m++] = (ml_prob_index){ p[i], i };
    qsort(cand, (size_t)m, sizeof *cand, cmp_desc);
    float cum = 0.0f;
    int last = m - 1;
    for (int i = 0; i < m; i++) {
        cum += cand[i].p;
        if (cum > topp) { last = i; break; }
    }
    const float r = coin * cum;
    float cdf = 0.0f;
    for (int i = 0; i <= last; i++) {
        cdf += cand[i].p;
        if (r < cdf) return cand[i].i;
    }
    return cand[last].i;
}

int ml_sample(ml_sampler *s, float *logits, int n) {
    if (s->temperature == 0.0f) return argmax(logits, n);
    for (int i = 0; i < n; i++) s->probs[i] = logits[i] / s->temperature;
    ml_k_softmax_scalar(s->probs, n);
    const float coin = random_f32(&s->rng);
    if (s->topp <= 0.0f || s->topp >= 1.0f) return sample_mult(s->probs, n, coin);
    return sample_topp(s->probs, n, s->topp, s->cand, coin);
}
