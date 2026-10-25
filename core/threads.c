#include <stdlib.h>
#include "core/threads.h"

#define ML_MAX_THREADS 64

#if defined(ML_THREADS_PTHREAD)

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>

/* Waiters spin this many pause iterations (tens of microseconds) before
 * sleeping: back-to-back ops in a forward pass then cost no syscalls, while an
 * idle pool still sleeps. */
#define SPIN_ITERS 20000

static inline void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#endif
}

typedef struct {
    ml_pool *pool;
    int tid;
} worker_arg;

struct ml_pool {
    int n;
    pthread_t *threads;
    worker_arg *args;
    pthread_mutex_t mu;
    pthread_cond_t start;
    ml_task_fn fn;          /* published by the release increment of gen */
    void *ctx;
    atomic_uint gen;        /* bumped once per ml_pool_run */
    atomic_int pending;     /* workers still running the current task */
    atomic_int sleepers;    /* workers blocked (or about to block) on start */
    atomic_int quit;
};

static void *worker(void *p) {
    const worker_arg *a = p;
    ml_pool *pool = a->pool;
    unsigned seen = 0;
    for (;;) {
        for (int i = 0; i < SPIN_ITERS && atomic_load_explicit(&pool->gen, memory_order_acquire) == seen &&
                        !atomic_load_explicit(&pool->quit, memory_order_relaxed); i++)
            cpu_relax();
        if (atomic_load_explicit(&pool->gen, memory_order_acquire) == seen) {
            /* Sleep. sleepers++ and the run's gen++ are both seq_cst, so either
             * the run sees a sleeper and broadcasts, or we see the new gen. */
            pthread_mutex_lock(&pool->mu);
            atomic_fetch_add(&pool->sleepers, 1);
            while (atomic_load(&pool->gen) == seen && !atomic_load(&pool->quit))
                pthread_cond_wait(&pool->start, &pool->mu);
            atomic_fetch_sub(&pool->sleepers, 1);
            pthread_mutex_unlock(&pool->mu);
        }
        if (atomic_load(&pool->quit)) return NULL;
        seen = atomic_load_explicit(&pool->gen, memory_order_acquire);
        pool->fn(pool->ctx, a->tid, pool->n);
        atomic_fetch_sub_explicit(&pool->pending, 1, memory_order_release);
    }
}

int ml_pool_threads(int n) { return n < 1 ? 1 : n > ML_MAX_THREADS ? ML_MAX_THREADS : n; }

ml_status ml_pool_create(ml_pool **out, int n) {
    n = ml_pool_threads(n);
    ml_pool *p = calloc(1, sizeof *p);
    if (!p) return ML_E_NOMEM;
    p->n = n;
    p->threads = calloc((size_t)n, sizeof *p->threads);
    p->args = calloc((size_t)n, sizeof *p->args);
    if (!p->threads || !p->args) { free(p->threads); free(p->args); free(p); return ML_E_NOMEM; }
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->start, NULL);
    atomic_init(&p->gen, 0);
    atomic_init(&p->pending, 0);
    atomic_init(&p->sleepers, 0);
    atomic_init(&p->quit, 0);
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
    p->fn = fn;
    p->ctx = ctx;
    atomic_store_explicit(&p->pending, p->n - 1, memory_order_relaxed);
    atomic_fetch_add(&p->gen, 1);   /* seq_cst: publishes fn/ctx/pending */
    if (atomic_load(&p->sleepers) > 0) {
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->start);
        pthread_mutex_unlock(&p->mu);
    }

    fn(ctx, 0, p->n);

    for (int i = 0; atomic_load_explicit(&p->pending, memory_order_acquire) > 0; i++) {
        if (i < SPIN_ITERS) cpu_relax();
        else sched_yield();
    }
}

void ml_pool_destroy(ml_pool *p) {
    pthread_mutex_lock(&p->mu);
    atomic_store(&p->quit, 1);
    pthread_cond_broadcast(&p->start);
    pthread_mutex_unlock(&p->mu);
    for (int i = 1; i < p->n; i++) pthread_join(p->threads[i], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->start);
    free(p->threads);
    free(p->args);
    free(p);
}

#else

struct ml_pool {
    int n;
};

int ml_pool_threads(int n) { (void)n; return 1; }

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
