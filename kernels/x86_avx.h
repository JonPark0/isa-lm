/* AVX2/FMA helpers shared by the 256-bit x86 kernel files. */
#ifndef ML_X86_AVX_H
#define ML_X86_AVX_H
#include <immintrin.h>

static inline float hsum256_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static inline float hmax256_ps(__m256 v) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

/* exp(x): same Cephes scheme as exp_ps, 8 lanes. */
static inline __m256 exp256_ps(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f)), _mm256_set1_ps(88.3762626647949f));
    const __m256 t = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f)));
    x = _mm256_fnmadd_ps(t, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(t, _mm256_set1_ps(-2.12194440e-4f), x);
    const __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_add_ps(_mm256_fmadd_ps(y, z, x), _mm256_set1_ps(1.0f));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(t), _mm256_set1_epi32(0x7f)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

/* Q4 block (16 bytes) -> 32 int8 in element order. */
static inline __m256i unpack_q4_avx2(const uint8_t *p) {
    const __m128i w = _mm_loadu_si128((const __m128i *)p);
    const __m128i m = _mm_set1_epi8(0x0F);
    const __m256i v = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(w, 4), m), _mm_and_si128(w, m));
    return _mm256_sub_epi8(v, _mm256_set1_epi8(8));
}

static inline __m256i load_q8_block(const int8_t *p) { return _mm256_loadu_si256((const __m256i *)p); }

/* Row loop shared by the 256-bit matvecs: 4 rows per pass reuse one activation
 * block. LOADW(row, b) -> __m256i of 32 int8 weights; DOT(w, x) -> 8 x int32. */
#define MATVEC256_BODY(LOADW, DOT)                                                \
    const int nb = W->cols / QK;                                                  \
    int r = r0;                                                                   \
    for (; r + 4 <= r1; r += 4) {                                                 \
        __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() }; \
        for (int b = 0; b < nb; b++) {                                            \
            const __m256i xv = _mm256_loadu_si256((const __m256i *)(x->q + b * QK)); \
            for (int i = 0; i < 4; i++) {                                         \
                const __m256i s = DOT(LOADW(r + i, b), xv);                       \
                const float sc = W->d[(size_t)(r + i) * nb + b] * x->d[b];        \
                acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(sc), acc[i]); \
            }                                                                     \
        }                                                                         \
        for (int i = 0; i < 4; i++) y[r + i] = hsum256_ps(acc[i]);                \
    }                                                                             \
    for (; r < r1; r++) {                                                         \
        __m256 acc = _mm256_setzero_ps();                                         \
        for (int b = 0; b < nb; b++) {                                            \
            const __m256i xv = _mm256_loadu_si256((const __m256i *)(x->q + b * QK)); \
            const __m256i s = DOT(LOADW(r, b), xv);                               \
            const float sc = W->d[(size_t)r * nb + b] * x->d[b];                  \
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(sc), acc); \
        }                                                                         \
        y[r] = hsum256_ps(acc);                                                   \
    }

#define LOADW_Q8_256(row, b) load_q8_block(W->q + (size_t)(row) * W->cols + (b) * QK)
#define LOADW_Q4_256(row, b) unpack_q4_avx2((const uint8_t *)W->q + ((size_t)(row) * nb + (b)) * (QK / 2))

#endif
