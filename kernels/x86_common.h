/* SSE2 helpers shared by the x86 kernel files. Include from .c files only. */
#ifndef ML_X86_COMMON_H
#define ML_X86_COMMON_H
#include <emmintrin.h>

static inline float hsum_ps(__m128 v) {
    __m128 s = _mm_add_ps(v, _mm_movehl_ps(v, v));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static inline float hmax_ps(__m128 v) {
    __m128 m = _mm_max_ps(v, _mm_movehl_ps(v, v));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

static inline __m128 abs_ps(__m128 v) { return _mm_andnot_ps(_mm_set1_ps(-0.0f), v); }

/* exp(x), Cephes expf polynomial with range reduction; rel. err ~1e-7 on
 * [-88, 88]. Inputs are clamped to that range. */
static inline __m128 exp_ps(__m128 x) {
    x = _mm_min_ps(_mm_max_ps(x, _mm_set1_ps(-88.3762626647949f)), _mm_set1_ps(88.3762626647949f));
    __m128 fx = _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(1.44269504088896341f)), _mm_set1_ps(0.5f));
    __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(fx));   /* floor(fx) */
    t = _mm_sub_ps(t, _mm_and_ps(_mm_cmpgt_ps(t, fx), _mm_set1_ps(1.0f)));
    x = _mm_sub_ps(x, _mm_mul_ps(t, _mm_set1_ps(0.693359375f)));
    x = _mm_sub_ps(x, _mm_mul_ps(t, _mm_set1_ps(-2.12194440e-4f)));
    const __m128 z = _mm_mul_ps(x, x);
    __m128 y = _mm_set1_ps(1.9875691500e-4f);
    y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(1.3981999507e-3f));
    y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(8.3334519073e-3f));
    y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(4.1665795894e-2f));
    y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(1.6666665459e-1f));
    y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(5.0000001201e-1f));
    y = _mm_add_ps(_mm_add_ps(_mm_mul_ps(y, z), x), _mm_set1_ps(1.0f));
    const __m128i e = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(t), _mm_set1_epi32(0x7f)), 23);
    return _mm_mul_ps(y, _mm_castsi128_ps(e));
}

/* Q4 block (16 bytes) -> two int8 vectors: elements 0..15 and 16..31. */
static inline void unpack_q4_sse2(const uint8_t *p, __m128i *lo, __m128i *hi) {
    const __m128i w = _mm_loadu_si128((const __m128i *)p);
    const __m128i m = _mm_set1_epi8(0x0F), eight = _mm_set1_epi8(8);
    *lo = _mm_sub_epi8(_mm_and_si128(w, m), eight);
    *hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(w, 4), m), eight);
}

#endif
