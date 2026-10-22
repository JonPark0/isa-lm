#include <smmintrin.h>
#include "kernels/sse41/k_sse41.h"
#include "kernels/x86_common.h"

/* sum(w * x) for 16 int8 pairs as 4 x int32: |x| (u8) times w carrying x's
 * sign (s8) through pmaddubsw. Quants are within [-127, 127], so the pairwise
 * int16 sums cannot saturate. */
static inline __m128i dot16_ssse3(__m128i w, __m128i x) {
    const __m128i p = _mm_maddubs_epi16(_mm_sign_epi8(x, x), _mm_sign_epi8(w, x));
    return _mm_madd_epi16(p, _mm_set1_epi16(1));
}

#define MATVEC_BODY(LOADW)                                                        \
    const int nb = W->cols / QK;                                                  \
    int r = r0;                                                                   \
    for (; r + 4 <= r1; r += 4) {                                                 \
        __m128 acc[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() }; \
        for (int b = 0; b < nb; b++) {                                            \
            const __m128i x0 = _mm_loadu_si128((const __m128i *)(x->q + b * QK));  \
            const __m128i x1 = _mm_loadu_si128((const __m128i *)(x->q + b * QK + 16)); \
            for (int i = 0; i < 4; i++) {                                         \
                __m128i w0, w1;                                                   \
                LOADW(r + i, b, w0, w1);                                          \
                const __m128i s = _mm_add_epi32(dot16_ssse3(w0, x0), dot16_ssse3(w1, x1)); \
                const float sc = W->d[(size_t)(r + i) * nb + b] * x->d[b];        \
                acc[i] = _mm_add_ps(acc[i], _mm_mul_ps(_mm_cvtepi32_ps(s), _mm_set1_ps(sc))); \
            }                                                                     \
        }                                                                         \
        for (int i = 0; i < 4; i++) y[r + i] = hsum_ps(acc[i]);                   \
    }                                                                             \
    for (; r < r1; r++) {                                                         \
        __m128 acc = _mm_setzero_ps();                                            \
        for (int b = 0; b < nb; b++) {                                            \
            __m128i w0, w1;                                                       \
            const __m128i x0 = _mm_loadu_si128((const __m128i *)(x->q + b * QK));  \
            const __m128i x1 = _mm_loadu_si128((const __m128i *)(x->q + b * QK + 16)); \
            LOADW(r, b, w0, w1);                                                  \
            const __m128i s = _mm_add_epi32(dot16_ssse3(w0, x0), dot16_ssse3(w1, x1)); \
            const float sc = W->d[(size_t)r * nb + b] * x->d[b];                  \
            acc = _mm_add_ps(acc, _mm_mul_ps(_mm_cvtepi32_ps(s), _mm_set1_ps(sc))); \
        }                                                                         \
        y[r] = hsum_ps(acc);                                                      \
    }

#define LOAD_Q8(row, b, w0, w1) do {                                              \
        const int8_t *p = W->q + (size_t)(row) * W->cols + (b) * QK;             \
        w0 = _mm_loadu_si128((const __m128i *)p);                                 \
        w1 = _mm_loadu_si128((const __m128i *)(p + 16)); } while (0)

#define LOAD_Q4(row, b, w0, w1)                                                   \
    unpack_q4_sse2((const uint8_t *)W->q + ((size_t)(row) * nb + (b)) * (QK / 2), &w0, &w1)

void ml_k_matvec_q8_sse41(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC_BODY(LOAD_Q8)
}

void ml_k_matvec_q4_sse41(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC_BODY(LOAD_Q4)
}
