#include <stdlib.h>
#include <string.h>
#include "kernels/scalar/k_scalar.h"
#include "model/sampler.h"

ml_status ml_sampler_init(ml_sampler *s, int vocab, const ml_sampler_opts *o) {
    memset(s, 0, sizeof *s);
    s->o = *o;
    s->vocab = vocab;
    s->rng = o->seed ? o->seed : 0x9E3779B97F4A7C15ull;
    if (s->o.repeat_last_n < 0) s->o.repeat_last_n = 0;
    s->probs = malloc(sizeof(float) * vocab);
    s->cand = malloc(sizeof(ml_prob_index) * vocab);
    s->recent = malloc(sizeof(int) * (size_t)(s->o.repeat_last_n > 0 ? s->o.repeat_last_n : 1));
    s->mark = calloc((size_t)vocab, sizeof(int));
    if (!s->probs || !s->cand || !s->recent || !s->mark) { ml_sampler_free(s); return ML_E_NOMEM; }
    return ML_OK;
}

void ml_sampler_free(ml_sampler *s) {
    free(s->probs);
    free(s->cand);
    free(s->recent);
    free(s->mark);
    s->probs = NULL;
    s->cand = NULL;
    s->recent = s->mark = NULL;
}

void ml_sampler_accept(ml_sampler *s, int token) {
    if (s->o.repeat_last_n <= 0 || token < 0 || token >= s->vocab) return;
    s->recent[s->recent_at] = token;
    s->recent_at = (s->recent_at + 1) % s->o.repeat_last_n;
    if (s->n_recent < s->o.repeat_last_n) s->n_recent++;
}

/* The CTRL / Hugging Face rule: divide a positive logit by the penalty,
 * multiply a negative one, once per distinct recent token. */
static void penalize(ml_sampler *s, float *logits, int n) {
    if (s->o.repeat_penalty == 1.0f || s->n_recent == 0) return;
    if (++s->stamp == 0) { memset(s->mark, 0, sizeof(int) * (size_t)s->vocab); s->stamp = 1; }
    for (int i = 0; i < s->n_recent; i++) {
        const int t = s->recent[i];
        if (t >= n || s->mark[t] == s->stamp) continue;
        s->mark[t] = s->stamp;
        logits[t] = logits[t] > 0.0f ? logits[t] / s->o.repeat_penalty : logits[t] * s->o.repeat_penalty;
    }
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

/* ranks before: higher probability, then lower id */
static int before(ml_prob_index a, ml_prob_index b) { return a.p > b.p || (a.p == b.p && a.i < b.i); }

/* The k most probable tokens into cand[0, k), sorted, through a min-heap of k
 * entries: O(n log k). */
static int top_k(const float *p, int n, int k, ml_prob_index *cand) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        const ml_prob_index e = { p[i], i };
        if (m < k) {
            int j = m++;
            while (j > 0 && before(cand[(j - 1) / 2], e)) { cand[j] = cand[(j - 1) / 2]; j = (j - 1) / 2; }
            cand[j] = e;
        } else if (before(e, cand[0])) {
            int j = 0;
            for (;;) {
                int c = 2 * j + 1;
                if (c >= m) break;
                if (c + 1 < m && before(cand[c], cand[c + 1])) c++;
                if (!before(e, cand[c])) break;
                cand[j] = cand[c];
                j = c;
            }
            cand[j] = e;
        }
    }
    qsort(cand, (size_t)m, sizeof *cand, cmp_desc);
    return m;
}

/* Samples from the sorted candidates cand[0, m), cut to the nucleus that
 * reaches topp of their mass when 0 < topp < 1. */
static int sample_sorted(const ml_prob_index *cand, int m, float topp, float coin) {
    float total = 0.0f;
    for (int i = 0; i < m; i++) total += cand[i].p;
    int last = m - 1;
    if (topp > 0.0f && topp < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < m; i++) {
            cum += cand[i].p;
            if (cum > topp * total) { last = i; break; }
        }
        total = cum;
    }
    const float r = coin * total;
    float cdf = 0.0f;
    for (int i = 0; i <= last; i++) {
        cdf += cand[i].p;
        if (r < cdf) return cand[i].i;
    }
    return cand[last].i;
}

static int sample_topp(const float *p, int n, float topp, ml_prob_index *cand, float coin) {
    /* Tokens below (1 - topp) / (n - 1) can never be in the nucleus. */
    const float cutoff = n > 1 ? (1.0f - topp) / (float)(n - 1) : 0.0f;
    int m = 0;
    for (int i = 0; i < n; i++)
        if (p[i] >= cutoff) cand[m++] = (ml_prob_index){ p[i], i };
    if (m == 0) return argmax(p, n);   /* nucleus smaller than any token */
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
    penalize(s, logits, n);
    if (s->o.temperature == 0.0f) return argmax(logits, n);
    for (int i = 0; i < n; i++) s->probs[i] = logits[i] / s->o.temperature;
    ml_k_softmax_scalar(s->probs, n);
    const float coin = random_f32(&s->rng);
    if (s->o.topk > 0 && s->o.topk < n)
        return sample_sorted(s->cand, top_k(s->probs, n, s->o.topk, s->cand), s->o.topp, coin);
    if (s->o.topp <= 0.0f || s->o.topp >= 1.0f) return sample_mult(s->probs, n, coin);
    return sample_topp(s->probs, n, s->o.topp, s->cand, coin);
}
