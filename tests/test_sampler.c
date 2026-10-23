#include <math.h>
#include "model/sampler.h"
#include "tests/tt.h"

#define N 96

/* Nucleus smaller than any single token (topp < 1/N on a flat distribution):
 * no token passes the cutoff, which must not index before the candidates. */
static void test_topp_empty_nucleus(void) {
    ml_sampler s;
    TT_EQ_INT(ml_sampler_init(&s, N, &(ml_sampler_opts){ 1.0f, 1e-6f, 0, 1.0f, 0, 7 }), ML_OK);
    for (int r = 0; r < 50; r++) {
        float logits[N];
        for (int i = 0; i < N; i++) logits[i] = 0.0f;
        const int t = ml_sample(&s, logits, N);
        TT_ASSERT(t >= 0 && t < N);
    }
    ml_sampler_free(&s);
}

static void test_topp_picks_from_nucleus(void) {
    ml_sampler s;
    TT_EQ_INT(ml_sampler_init(&s, N, &(ml_sampler_opts){ 1.0f, 0.5f, 0, 1.0f, 0, 3 }), ML_OK);
    for (int r = 0; r < 50; r++) {
        float logits[N];
        for (int i = 0; i < N; i++) logits[i] = 0.0f;
        logits[17] = 20.0f;   /* holds > 99% of the mass */
        TT_EQ_INT(ml_sample(&s, logits, N), 17);
    }
    ml_sampler_free(&s);
}

static void test_greedy_ties_lowest(void) {
    ml_sampler s;
    TT_EQ_INT(ml_sampler_init(&s, N, &(ml_sampler_opts){ 0.0f, 0.9f, 0, 1.0f, 0, 1 }), ML_OK);
    float logits[N] = { 0 };
    logits[40] = logits[9] = 3.0f;
    TT_EQ_INT(ml_sample(&s, logits, N), 9);
    ml_sampler_free(&s);
}

/* top-k: only the k most likely tokens are ever drawn. */
static void test_topk(void) {
    ml_sampler s;
    TT_EQ_INT(ml_sampler_init(&s, N, &(ml_sampler_opts){ 2.0f, 0.0f, 3, 1.0f, 0, 5 }), ML_OK);
    int seen[N] = { 0 };
    for (int r = 0; r < 400; r++) {
        float logits[N];
        for (int i = 0; i < N; i++) logits[i] = (float)(i % 7) * 0.1f;   /* flat-ish */
        logits[10] = 1.0f; logits[50] = 0.95f; logits[80] = 0.9f;
        seen[ml_sample(&s, logits, N)]++;
    }
    TT_EQ_INT(seen[10] + seen[50] + seen[80], 400);
    TT_ASSERT(seen[10] > 0 && seen[50] > 0 && seen[80] > 0);
    ml_sampler_free(&s);
    /* k = 1 is greedy, whatever the temperature */
    TT_EQ_INT(ml_sampler_init(&s, N, &(ml_sampler_opts){ 1.5f, 0.9f, 1, 1.0f, 0, 9 }), ML_OK);
    for (int r = 0; r < 50; r++) {
        float logits[N] = { 0 };
        logits[33] = 0.5f;
        TT_EQ_INT(ml_sample(&s, logits, N), 33);
    }
    ml_sampler_free(&s);
}

/* Repetition penalty: a recent token's positive logit is divided by the
 * penalty, a negative one multiplied, once per distinct token, and only for
 * the last repeat_last_n tokens. */
static void test_repeat_penalty(void) {
    ml_sampler s;
    TT_EQ_INT(ml_sampler_init(&s, N, &(ml_sampler_opts){ 0.0f, 0.0f, 0, 2.0f, 3, 1 }), ML_OK);
    float logits[N] = { 0 };
    logits[5] = 4.0f; logits[6] = 3.0f; logits[7] = -1.0f;
    TT_EQ_INT(ml_sample(&s, logits, N), 5);          /* nothing seen yet */
    ml_sampler_accept(&s, 5);
    ml_sampler_accept(&s, 5);                         /* penalized once, not twice */
    ml_sampler_accept(&s, 7);
    float l2[N] = { 0 };
    l2[5] = 4.0f; l2[6] = 3.0f; l2[7] = -1.0f;
    TT_EQ_INT(ml_sample(&s, l2, N), 6);               /* 4 / 2 = 2 < 3 */
    TT_ASSERT(l2[5] == 2.0f && l2[7] == -2.0f && l2[6] == 3.0f);
    ml_sampler_accept(&s, 1); ml_sampler_accept(&s, 2); ml_sampler_accept(&s, 3);   /* 5 leaves the window */
    float l3[N] = { 0 };
    l3[5] = 4.0f; l3[6] = 3.0f;
    TT_EQ_INT(ml_sample(&s, l3, N), 5);
    ml_sampler_free(&s);
}

int TT_ENTRY(int argc, char **argv) {
    (void)argc; (void)argv;
    TT_RUN(test_topp_empty_nucleus);
    TT_RUN(test_topp_picks_from_nucleus);
    TT_RUN(test_greedy_ties_lowest);
    TT_RUN(test_topk);
    TT_RUN(test_repeat_penalty);
    TT_MAIN_END("test_sampler");
}
