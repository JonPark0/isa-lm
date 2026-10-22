#include <stdlib.h>
#include "core/arena.h"

ml_status ml_arena_init(ml_arena *a, size_t cap) {
    cap = (cap + ML_ALIGN - 1) & ~(size_t)(ML_ALIGN - 1);
    a->base = aligned_alloc(ML_ALIGN, cap ? cap : ML_ALIGN);
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
    free(a->base);
    a->base = NULL;
    a->cap = a->off = 0;
}
