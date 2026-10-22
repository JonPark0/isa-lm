/* Pool contract: ml_pool_run runs fn once per thread id and returns only
 * after all of them finished, across many back-to-back runs. */
#include <string.h>
#include "core/threads.h"
#include "tests/tt.h"

typedef struct {
    int hits[64];
    int run;
    int bad;
} ctx_t;

static void task(void *p, int tid, int nt) {
    ctx_t *c = p;
    if (tid < 0 || tid >= nt || nt > 64) { c->bad = 1; return; }
    c->hits[tid] += 1;   /* each tid touches only its own slot */
}

static void check_pool(int n, int runs) {
    ml_pool *p;
    TT_EQ_INT(ml_pool_create(&p, n), ML_OK);
    const int nt = ml_pool_size(p);
    ctx_t c;
    memset(&c, 0, sizeof c);
    for (int r = 1; r <= runs; r++) {
        ml_pool_run(p, task, &c);
        for (int t = 0; t < nt; t++)
            if (c.hits[t] != r) { TT_EQ_INT(c.hits[t], r); ml_pool_destroy(p); return; }
    }
    TT_ASSERT(!c.bad);
    ml_pool_destroy(p);
}

static void test_pool_1(void) { check_pool(1, 1000); }
static void test_pool_4(void) { check_pool(4, 20000); }
static void test_pool_7(void) { check_pool(7, 5000); }
static void test_pool_create_destroy_loop(void) {
    for (int i = 0; i < 50; i++) check_pool(3, 10);
}

int TT_ENTRY(int argc, char **argv) {
    (void)argc; (void)argv;
    TT_RUN(test_pool_1);
    TT_RUN(test_pool_4);
    TT_RUN(test_pool_7);
    TT_RUN(test_pool_create_destroy_loop);
    TT_MAIN_END("test_threads");
}
