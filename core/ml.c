#include <stdlib.h>
#include "core/ml.h"
#ifdef _WIN32
#include <malloc.h>
#endif

const char *ml_status_str(ml_status s) {
    switch (s) {
    case ML_OK: return "ok";
    case ML_E_IO: return "i/o error";
    case ML_E_FORMAT: return "bad file format";
    case ML_E_SHAPE: return "bad tensor shape";
    case ML_E_NOMEM: return "out of memory";
    case ML_E_ISA: return "CPU lacks required ISA";
    }
    return "unknown";
}

void *ml_aligned_alloc(size_t n) {
    n = (n + ML_ALIGN - 1) & ~(size_t)(ML_ALIGN - 1);
    if (n == 0) n = ML_ALIGN;
#ifdef _WIN32
    return _aligned_malloc(n, ML_ALIGN);
#else
    return aligned_alloc(ML_ALIGN, n);
#endif
}

void ml_aligned_free(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}
