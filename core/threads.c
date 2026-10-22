#include <stdlib.h>
#include "core/threads.h"

struct ml_pool {
    int n;
};

ml_status ml_pool_create(ml_pool **p, int n) {
    (void)n;
    *p = malloc(sizeof **p);
    if (!*p) return ML_E_NOMEM;
    (*p)->n = 1;
    return ML_OK;
}

void ml_pool_run(ml_pool *p, ml_task_fn fn, void *ctx) {
    (void)p;
    fn(ctx, 0, 1);
}

int ml_pool_size(const ml_pool *p) { return p->n; }

void ml_pool_destroy(ml_pool *p) { free(p); }
