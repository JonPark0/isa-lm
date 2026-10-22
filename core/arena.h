/* Single allocation, 64 B-aligned bump allocator. */
#ifndef ML_ARENA_H
#define ML_ARENA_H
#include "core/ml.h"

typedef struct {
    uint8_t *base;
    size_t cap, off;
} ml_arena;

ml_status ml_arena_init(ml_arena *a, size_t cap);
void *ml_arena_alloc(ml_arena *a, size_t n);   /* NULL when full */
void ml_arena_free(ml_arena *a);

#endif
