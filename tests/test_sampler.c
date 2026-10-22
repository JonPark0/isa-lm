#include <math.h>
#include "model/sampler.h"
#include "tests/tt.h"

#define N 96

/* Nucleus smaller than any single token (topp < 1/N on a flat distribution):
 * no token passes the cutoff, which must not index before the candidates. */
static void test_topp_empty_nucleus(void) {
    ml_sampler s;
    TT_EQ_INT(ml_sampler_init(&s, N, 1.0f, 1e-6f, 7), ML_OK);
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
    TT_EQ_INT(ml_sampler_init(&s, N, 1.0f, 0.5f, 3), ML_OK);
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
    TT_EQ_INT(ml_sampler_init(&s, N, 0.0f, 0.9f, 1), ML_OK);
    float logits[N] = { 0 };
    logits[40] = logits[9] = 3.0f;
    TT_EQ_INT(ml_sample(&s, logits, N), 9);
    ml_sampler_free(&s);
}

int TT_ENTRY(int argc, char **argv) {
    (void)argc; (void)argv;
    TT_RUN(test_topp_empty_nucleus);
    TT_RUN(test_topp_picks_from_nucleus);
    TT_RUN(test_greedy_ties_lowest);
    TT_MAIN_END("test_sampler");
}
