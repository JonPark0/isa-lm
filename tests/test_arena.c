#include <stdint.h>
#include "core/arena.h"
#include "tests/tt.h"

static void test_arena_alignment(void) {
    ml_arena a;
    TT_EQ_INT(ml_arena_init(&a, 4096), ML_OK);
    void *p1 = ml_arena_alloc(&a, 1), *p2 = ml_arena_alloc(&a, 100), *p3 = ml_arena_alloc(&a, 7);
    TT_ASSERT(p1 && p2 && p3);
    TT_EQ_INT((uintptr_t)p1 % 64, 0);
    TT_EQ_INT((uintptr_t)p2 % 64, 0);
    TT_EQ_INT((uintptr_t)p3 % 64, 0);
    TT_ASSERT(p1 != p2 && p2 != p3);
    ml_arena_free(&a);
}

static void test_arena_full(void) {
    ml_arena a;
    TT_EQ_INT(ml_arena_init(&a, 128), ML_OK);
    TT_ASSERT(ml_arena_alloc(&a, 64) != NULL);
    TT_ASSERT(ml_arena_alloc(&a, 65) == NULL);
    ml_arena_free(&a);
}

int TT_ENTRY(int argc, char **argv) {
    (void)argc; (void)argv;
    TT_RUN(test_arena_alignment);
    TT_RUN(test_arena_full);
    TT_MAIN_END("test_arena");
}
