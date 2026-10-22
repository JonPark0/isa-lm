#include "core/arena.h"

ml_status ml_arena_init(ml_arena *a, size_t cap) {
    a->base = NULL;
    a->cap = a->off = 0;
    if (cap > (size_t)-1 - ML_ALIGN) return ML_E_NOMEM;
    cap = (cap + ML_ALIGN - 1) & ~(size_t)(ML_ALIGN - 1);
    a->base = ml_aligned_alloc(cap);
    a->cap = cap;
    a->off = 0;
    return a->base ? ML_OK : ML_E_NOMEM;
}

void *ml_arena_alloc(ml_arena *a, size_t n) {
    size_t need = (n + ML_ALIGN - 1) & ~(size_t)(ML_ALIGN - 1);
    if (need > a->cap - a->off) return NULL;
    void *p = a->base + a->off;
    a->off += need;
    return p;
}

void ml_arena_free(ml_arena *a) {
    ml_aligned_free(a->base);
    a->base = NULL;
    a->cap = a->off = 0;
}
