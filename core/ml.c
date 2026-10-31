#include <math.h>
#include <stdlib.h>
#include <string.h>
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

uint16_t ml_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint16_t sign = (uint16_t)((x >> 16) & 0x8000);
    x &= 0x7fffffff;
    if (x > 0x7f800000) return sign | 0x7e00;               /* NaN */
    if (x >= 0x477ff000) return sign | 0x7c00;              /* inf, or rounds to it */
    if (x < 0x38800000) {                                   /* below 2^-14: subnormal */
        float a;
        memcpy(&a, &x, 4);
        return sign | (uint16_t)lrintf(a * 16777216.0f);    /* units of 2^-24 */
    }
    x += 0xfff + ((x >> 13) & 1);
    return sign | (uint16_t)((x >> 13) - (112u << 10));
}

float ml_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) return (sign ? -1.0f : 1.0f) * (float)m * (1.0f / 16777216.0f);  /* subnormal, exact */
    const uint32_t x = sign | (e == 31 ? 0x7f800000u | (m << 13) : ((e + 112u) << 23) | (m << 13));
    float f;
    memcpy(&f, &x, 4);
    return f;
}
