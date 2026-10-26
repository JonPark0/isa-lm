/* Worker pool. THREADS=1 builds run every task inline on the caller. */
#ifndef ML_THREADS_H
#define ML_THREADS_H
#include "core/ml.h"

typedef void (*ml_task_fn)(void *ctx, int tid, int nthreads);
typedef struct ml_pool ml_pool;

ml_status ml_pool_create(ml_pool **p, int n);
/* Runs fn(ctx, tid, n) for tid = 0..n-1 and returns when all have finished. */
void ml_pool_run(ml_pool *p, ml_task_fn fn, void *ctx);
int ml_pool_size(const ml_pool *p);
/* The size ml_pool_create(n) gives a pool. */
int ml_pool_threads(int n);
void ml_pool_destroy(ml_pool *p);

#endif
