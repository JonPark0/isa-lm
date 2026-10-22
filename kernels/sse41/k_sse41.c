#include <smmintrin.h>
#include "kernels/sse41/k_sse41.h"
#include "kernels/x86_common.h"
#include "kernels/x86_pack.h"

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

/* Packed kernels: 4 rows per group, same scheme as the AVX2 ones at 128 bits. */
#define PK 4

void ml_k_matvec_q8_pk_sse41(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    const __m128i ones = _mm_set1_epi16(1);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (8 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        __m128 acc = _mm_setzero_ps();
        for (int b = 0; b < nb; b++, p += 8 * PK * 4) {
            const int32_t *xb = (const int32_t *)(x->q + b * QK);
            __m128i s = _mm_setzero_si128();
            for (int k = 0; k < 8; k++) {
                const __m128i xv = _mm_set1_epi32(xb[k]);
                const __m128i w = _mm_loadu_si128((const __m128i *)(p + k * PK * 4));
                s = _mm_add_epi32(s, _mm_madd_epi16(_mm_maddubs_epi16(_mm_sign_epi8(xv, xv), _mm_sign_epi8(w, xv)), ones));
            }
            acc = _mm_add_ps(acc, _mm_mul_ps(_mm_cvtepi32_ps(s), _mm_mul_ps(_mm_loadu_ps(d + b * PK), _mm_set1_ps(x->d[b]))));
        }
        float t[PK];
        _mm_storeu_ps(t, acc);
        pk_store(y, t, row, PK, r1);
    }
}

void ml_k_matvec_q4_pk_sse41(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    int xsum[PK_MAX_BLOCKS];
    pk_block_sums(xsum, x, nb, 8);
    const __m128i m4 = _mm_set1_epi8(0x0F), ones = _mm_set1_epi16(1);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (4 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        __m128 acc = _mm_setzero_ps();
        for (int b = 0; b < nb; b++, p += 4 * PK * 4) {
            const int32_t *xb = (const int32_t *)(x->q + b * QK);
            __m128i s16 = _mm_setzero_si128();
            for (int v = 0; v < 4; v++) {
                const __m128i w = _mm_loadu_si128((const __m128i *)(p + v * PK * 4));
                s16 = _mm_add_epi16(s16, _mm_maddubs_epi16(_mm_and_si128(w, m4), _mm_set1_epi32(xb[v])));
                s16 = _mm_add_epi16(s16, _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(w, 4), m4), _mm_set1_epi32(xb[v + 4])));
            }
            const __m128i s = _mm_sub_epi32(_mm_madd_epi16(s16, ones), _mm_set1_epi32(xsum[b]));
            acc = _mm_add_ps(acc, _mm_mul_ps(_mm_cvtepi32_ps(s), _mm_mul_ps(_mm_loadu_ps(d + b * PK), _mm_set1_ps(x->d[b]))));
        }
        float t[PK];
        _mm_storeu_ps(t, acc);
        pk_store(y, t, row, PK, r1);
    }
}

/* Batched packed kernels: each weight vector feeds up to PK_TILE tokens;
 * per token the arithmetic is the matvec's, so results match it exactly. */
void ml_k_matmul_q8_pk_sse41(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    const int nb = W->cols / QK;
    const __m128i ones = _mm_set1_epi16(1);
    for (int t0 = 0; t0 < X->n; t0 += PK_TILE) {
        const int T = X->n - t0 < PK_TILE ? X->n - t0 : PK_TILE;
        for (int row = r0; row < r1; row += PK) {
            const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (8 * PK * 4);
            const float *d = W->pd + (size_t)(row / PK) * nb * PK;
            __m128 acc[PK_TILE];
            for (int t = 0; t < PK_TILE; t++) acc[t] = _mm_setzero_ps();
            for (int b = 0; b < nb; b++, p += 8 * PK * 4) {
                __m128i s[PK_TILE];
                for (int t = 0; t < PK_TILE; t++) s[t] = _mm_setzero_si128();
                for (int k = 0; k < 8; k++) {
                    const __m128i w = _mm_loadu_si128((const __m128i *)(p + k * PK * 4));
                    for (int t = 0; t < T; t++) {
                        const __m128i xv = _mm_set1_epi32(((const int32_t *)(X->q + (size_t)(t0 + t) * X->k + b * QK))[k]);
                        s[t] = _mm_add_epi32(s[t], _mm_madd_epi16(_mm_maddubs_epi16(_mm_sign_epi8(xv, xv), _mm_sign_epi8(w, xv)), ones));
                    }
                }
                const __m128 dv = _mm_loadu_ps(d + b * PK);
                for (int t = 0; t < T; t++) acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_cvtepi32_ps(s[t]), _mm_mul_ps(dv, _mm_set1_ps(X->d[(size_t)(t0 + t) * nb + b]))));
            }
            for (int t = 0; t < T; t++) {
                float o[PK];
                _mm_storeu_ps(o, acc[t]);
                pk_store(Y + (size_t)(t0 + t) * ldy, o, row, PK, r1);
            }
        }
    }
}

void ml_k_matmul_q4_pk_sse41(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    const int nb = W->cols / QK;
    static _Thread_local int xsum[PK_TILE][PK_MAX_BLOCKS];
    const __m128i m4 = _mm_set1_epi8(0x0F), ones = _mm_set1_epi16(1);
    for (int t0 = 0; t0 < X->n; t0 += PK_TILE) {
        const int T = X->n - t0 < PK_TILE ? X->n - t0 : PK_TILE;
        pk_block_sums_batch(xsum, X, t0, T, nb, 8);
        for (int row = r0; row < r1; row += PK) {
            const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (4 * PK * 4);
            const float *d = W->pd + (size_t)(row / PK) * nb * PK;
            __m128 acc[PK_TILE];
            for (int t = 0; t < PK_TILE; t++) acc[t] = _mm_setzero_ps();
            for (int b = 0; b < nb; b++, p += 4 * PK * 4) {
                __m128i s16[PK_TILE];
                for (int t = 0; t < PK_TILE; t++) s16[t] = _mm_setzero_si128();
                for (int v = 0; v < 4; v++) {
                    const __m128i w = _mm_loadu_si128((const __m128i *)(p + v * PK * 4));
                    const __m128i lo = _mm_and_si128(w, m4), hi = _mm_and_si128(_mm_srli_epi16(w, 4), m4);
                    for (int t = 0; t < T; t++) {
                        s16[t] = _mm_add_epi16(s16[t], _mm_maddubs_epi16(lo, _mm_set1_epi32(((const int32_t *)(X->q + (size_t)(t0 + t) * X->k + b * QK))[v])));
                        s16[t] = _mm_add_epi16(s16[t], _mm_maddubs_epi16(hi, _mm_set1_epi32(((const int32_t *)(X->q + (size_t)(t0 + t) * X->k + b * QK))[v + 4])));
                    }
                }
                const __m128 dv = _mm_loadu_ps(d + b * PK);
                for (int t = 0; t < T; t++) {
                    const __m128i s = _mm_sub_epi32(_mm_madd_epi16(s16[t], ones), _mm_set1_epi32(xsum[t][b]));
                    acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_cvtepi32_ps(s), _mm_mul_ps(dv, _mm_set1_ps(X->d[(size_t)(t0 + t) * nb + b]))));
                }
            }
            for (int t = 0; t < T; t++) {
                float o[PK];
                _mm_storeu_ps(o, acc[t]);
                pk_store(Y + (size_t)(t0 + t) * ldy, o, row, PK, r1);
            }
        }
    }
}
