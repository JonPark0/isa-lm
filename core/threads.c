#include <stdlib.h>
#include "core/threads.h"

#define ML_MAX_THREADS 64

#if defined(ML_THREADS_PTHREAD)

#include <pthread.h>

typedef struct {
    ml_pool *pool;
    int tid;
} worker_arg;

struct ml_pool {
    int n;
    pthread_t *threads;
    worker_arg *args;
    pthread_mutex_t mu;
    pthread_cond_t start, done;
    ml_task_fn fn;
    void *ctx;
    unsigned gen;     /* bumped once per ml_pool_run */
    int pending;      /* workers still running the current task */
    int quit;
};

static void *worker(void *p) {
    const worker_arg *a = p;
    ml_pool *pool = a->pool;
    unsigned seen = 0;
    for (;;) {
        pthread_mutex_lock(&pool->mu);
        while (pool->gen == seen && !pool->quit) pthread_cond_wait(&pool->start, &pool->mu);
        if (pool->quit) { pthread_mutex_unlock(&pool->mu); return NULL; }
        seen = pool->gen;
        const ml_task_fn fn = pool->fn;
        void *ctx = pool->ctx;
        pthread_mutex_unlock(&pool->mu);

        fn(ctx, a->tid, pool->n);

        pthread_mutex_lock(&pool->mu);
        if (--pool->pending == 0) pthread_cond_signal(&pool->done);
        pthread_mutex_unlock(&pool->mu);
    }
}

ml_status ml_pool_create(ml_pool **out, int n) {
    if (n < 1) n = 1;
    if (n > ML_MAX_THREADS) n = ML_MAX_THREADS;
    ml_pool *p = calloc(1, sizeof *p);
    if (!p) return ML_E_NOMEM;
    p->n = n;
    p->threads = calloc((size_t)n, sizeof *p->threads);
    p->args = calloc((size_t)n, sizeof *p->args);
    if (!p->threads || !p->args) { free(p->threads); free(p->args); free(p); return ML_E_NOMEM; }
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->start, NULL);
    pthread_cond_init(&p->done, NULL);
    for (int i = 1; i < n; i++) {   /* the caller is thread 0 */
        p->args[i] = (worker_arg){ p, i };
        if (pthread_create(&p->threads[i], NULL, worker, &p->args[i]) != 0) {
            p->n = i;   /* destroy joins only the threads that started */
            ml_pool_destroy(p);
            return ML_E_NOMEM;
        }
    }
    *out = p;
    return ML_OK;
}

void ml_pool_run(ml_pool *p, ml_task_fn fn, void *ctx) {
    if (p->n == 1) { fn(ctx, 0, 1); return; }
    pthread_mutex_lock(&p->mu);
    p->fn = fn;
    p->ctx = ctx;
    p->pending = p->n - 1;
    p->gen++;
    pthread_cond_broadcast(&p->start);
    pthread_mutex_unlock(&p->mu);

    fn(ctx, 0, p->n);

    pthread_mutex_lock(&p->mu);
    while (p->pending > 0) pthread_cond_wait(&p->done, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

void ml_pool_destroy(ml_pool *p) {
    pthread_mutex_lock(&p->mu);
    p->quit = 1;
    pthread_cond_broadcast(&p->start);
    pthread_mutex_unlock(&p->mu);
    for (int i = 1; i < p->n; i++) pthread_join(p->threads[i], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->start);
    pthread_cond_destroy(&p->done);
    free(p->threads);
    free(p->args);
    free(p);
}

#else

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

void ml_pool_destroy(ml_pool *p) { free(p); }

#endif

int ml_pool_size(const ml_pool *p) { return p->n; }
