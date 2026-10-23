/* Core types shared by every layer. */
#ifndef ML_H
#define ML_H
#include <stddef.h>
#include <stdint.h>

typedef enum {
    ML_OK = 0,
    ML_E_IO,
    ML_E_FORMAT,
    ML_E_SHAPE,
    ML_E_NOMEM,
    ML_E_ISA
} ml_status;

#define QK 32          /* quantization block size */
#define ML_ALIGN 64    /* alignment of every tensor array */
/* Blocks with max |x| below this quantize to zero: 127/amax would overflow to
 * inf (amax < ~3.7e-37), and such values carry no signal. */
#define ML_QUANT_TINY 1e-30f

/* For kernels: ML_INLINE forces inlining, and ML_UNROLL(n) asks for complete
 * unrolling of the loop that follows. Together they let a tile body be
 * instantiated for constant token counts with its per-token accumulators in
 * registers instead of on the stack. */
#if defined(_MSC_VER) && !defined(__clang__)
#define ML_INLINE static __forceinline
#define ML_UNROLL(n)
#else
#define ML_INLINE static inline __attribute__((always_inline))
#define ML_PRAGMA_(x) _Pragma(#x)
#define ML_UNROLL(n) ML_PRAGMA_(GCC unroll n)
#endif

/* Software prefetch for the packed matvecs' weight stream. Decoding reads
 * every weight exactly once, and x86 hardware prefetchers stop at 4 KB page
 * boundaries; requesting the bytes ML_PREFETCH_DIST ahead keeps one core's
 * stream going (single-thread decode 1.3-2.5x faster on a Xeon). On Arm
 * Neoverse it made Q8 decoding 26% slower and Q4 no faster, so it is on by
 * default for x86 only. -DML_USE_PREFETCH=0 or =1 overrides the default. */
#ifndef ML_PREFETCH_DIST
#define ML_PREFETCH_DIST 4096
#endif
#ifndef ML_USE_PREFETCH
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define ML_USE_PREFETCH 1
#else
#define ML_USE_PREFETCH 0
#endif
#endif
#if !ML_USE_PREFETCH || (defined(_MSC_VER) && !defined(__clang__))
#define ML_PREFETCH(p) ((void)(p))
#else
#define ML_PREFETCH(p) __builtin_prefetch((p), 0, 3)
#endif

/* Prefetches the cache lines ML_PREFETCH_DIST bytes past [p, p + bytes). */
static inline void ml_prefetch_ahead(const void *p, int bytes) {
    for (int l = 0; l < bytes; l += 64) ML_PREFETCH((const void *)((uintptr_t)p + ML_PREFETCH_DIST + (uintptr_t)l));
}

enum { ML_F32 = 0, ML_Q8_0 = 1, ML_Q4_0 = 2 };

/* Quantized weight matrix, SoA: q[rows*cols] (Q4: rows*cols/2 packed bytes),
 * d[rows*cols/QK] scales. */
typedef struct {
    const int8_t *q;
    const float *d;
    int rows, cols;
    uint8_t type;
    /* Row-interleaved copy for the packed kernels (core/pack.h), or NULL. */
    const uint8_t *pq;
    const float *pd;
} ml_qtensor;

/* Q8_0 quantized activation vector of length n. */
typedef struct {
    int8_t *q;
    float *d;
    int n;
} ml_qact;

/* n Q8_0 activation vectors of length k: token t is q + t*k, d + t*(k/QK). */
typedef struct {
    int8_t *q;
    float *d;
    int n, k;
} ml_qbatch;

const char *ml_status_str(ml_status s);

/* *r = a * b; returns 0 if that overflows size_t (32-bit targets!). */
static inline int ml_mul_size(size_t a, size_t b, size_t *r) {
    if (b && a > (size_t)-1 / b) return 0;
    *r = a * b;
    return 1;
}

/* 64 B-aligned allocation (C11 aligned_alloc is missing on MSVC/MinGW). */
void *ml_aligned_alloc(size_t n);
void ml_aligned_free(void *p);

#endif
