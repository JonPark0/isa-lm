#include <math.h>
#include <string.h>
#include "kernels/avx512vnni/k_avx512vnni.h"
#include "kernels/x86_avx.h"

/* 64 int8 pairs (two blocks) -> 16 x int32; lanes 0..7 = first block. There
 * is no 512-bit psignb: negate w where x < 0 instead. */
static inline __m512i dot64_vnni(__m512i w, __m512i x) {
    const __m512i ws = _mm512_mask_sub_epi8(w, _mm512_movepi8_mask(x), _mm512_setzero_si512(), w);
    return _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(x), ws);
}

static inline __m256i dot32_avx2(__m256i w, __m256i x) {
    const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(w, x));
    return _mm256_madd_epi16(p, _mm256_set1_epi16(1));
}

static inline __m512i join256(__m256i lo, __m256i hi) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

static inline __m512 scale2(float s0, float s1) {
    return _mm512_mask_blend_ps(0xFF00, _mm512_set1_ps(s0), _mm512_set1_ps(s1));
}

/* LOADW2(row, b) -> two blocks b, b+1 as __m512i; LOADW1(row, b) -> one block
 * as __m256i (odd tail). 4 rows per pass share each activation load. */
#define MATVEC512_BODY(LOADW2, LOADW1)                                            \
    const int nb = W->cols / QK;                                                  \
    for (int r = r0; r < r1; r += 4) {                                            \
        const int nr = r1 - r < 4 ? r1 - r : 4;                                   \
        __m512 acc[4];                                                            \
        __m256 tail[4];                                                           \
        for (int i = 0; i < 4; i++) { acc[i] = _mm512_setzero_ps(); tail[i] = _mm256_setzero_ps(); } \
        int b = 0;                                                                \
        for (; b + 2 <= nb; b += 2) {                                             \
            const __m512i xv = _mm512_loadu_si512((const void *)(x->q + b * QK)); \
            for (int i = 0; i < nr; i++) {                                        \
                const float *wd = W->d + (size_t)(r + i) * nb + b;                \
                const __m512i s = dot64_vnni(LOADW2(r + i, b), xv);               \
                acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s),                   \
                                         scale2(wd[0] * x->d[b], wd[1] * x->d[b + 1]), acc[i]); \
            }                                                                     \
        }                                                                         \
        if (b < nb) {                                                             \
            const __m256i xv = _mm256_loadu_si256((const __m256i *)(x->q + b * QK)); \
            for (int i = 0; i < nr; i++) {                                        \
                const __m256i s = dot32_avx2(LOADW1(r + i, b), xv);               \
                tail[i] = _mm256_mul_ps(_mm256_cvtepi32_ps(s),                    \
                                        _mm256_set1_ps(W->d[(size_t)(r + i) * nb + b] * x->d[b])); \
            }                                                                     \
        }                                                                         \
        for (int i = 0; i < nr; i++) y[r + i] = _mm512_reduce_add_ps(acc[i]) + hsum256_ps(tail[i]); \
    }

#define LOADW2_Q8(row, b) _mm512_loadu_si512((const void *)(W->q + (size_t)(row) * W->cols + (b) * QK))
#define Q4P(row, b) ((const uint8_t *)W->q + ((size_t)(row) * nb + (b)) * (QK / 2))
#define LOADW2_Q4(row, b) join256(unpack_q4_avx2(Q4P(row, b)), unpack_q4_avx2(Q4P(row, (b) + 1)))

void ml_k_matvec_q8_avx512vnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC512_BODY(LOADW2_Q8, LOADW_Q8_256)
}

void ml_k_matvec_q4_avx512vnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC512_BODY(LOADW2_Q4, LOADW_Q4_256)
}

void ml_k_quantize_q8_avx512vnni(ml_qact *out, const float *x, int k) {
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512i signbit = _mm512_set1_epi32((int)0x80000000u);
    const __m128i lim = _mm_set1_epi8(-127);
    for (int b = 0; b < k / QK; b++) {
        const float *xb = x + b * QK;
        const __m512 v0 = _mm512_loadu_ps(xb), v1 = _mm512_loadu_ps(xb + 16);
        const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
        if (amax < ML_QUANT_TINY) {
            out->d[b] = 0.0f;
            memset(out->q + b * QK, 0, QK);
            continue;
        }
        const float id = 127.0f / amax;
        out->d[b] = amax / 127.0f;
        const __m512 vid = _mm512_set1_ps(id);
        const __m512 v[2] = { v0, v1 };
        for (int j = 0; j < 2; j++) {
            /* copysign(0.5, v): or v's sign bit into 0.5 (no FMA: matches scalar rounding) */
            const __m512 h = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(half),
                                 _mm512_and_si512(_mm512_castps_si512(v[j]), signbit)));
            const __m512i i32 = _mm512_cvttps_epi32(_mm512_add_ps(_mm512_mul_ps(v[j], vid), h));
            _mm_storeu_si128((__m128i *)(out->q + b * QK + 16 * j), _mm_max_epi8(_mm512_cvtsepi32_epi8(i32), lim));
        }
    }
}

void ml_k_rmsnorm_avx512vnni(float *o, const float *x, const float *w, int n, float eps) {
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) {
        const __mmask16 m = n - i >= 16 ? 0xFFFF : (__mmask16)((1u << (n - i)) - 1);
        const __m512 v = _mm512_maskz_loadu_ps(m, x + i);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    const float inv = 1.0f / sqrtf(_mm512_reduce_add_ps(acc) / (float)n + eps);
    const __m512 vinv = _mm512_set1_ps(inv);
    for (int i = 0; i < n; i += 16) {
        const __mmask16 m = n - i >= 16 ? 0xFFFF : (__mmask16)((1u << (n - i)) - 1);
        const __m512 v = _mm512_mul_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(m, x + i), vinv),
                                       _mm512_maskz_loadu_ps(m, w + i));
        _mm512_mask_storeu_ps(o + i, m, v);
    }
}

void ml_k_add_avx512vnni(float *y, const float *x, int n) {
    for (int i = 0; i < n; i += 16) {
        const __mmask16 m = n - i >= 16 ? 0xFFFF : (__mmask16)((1u << (n - i)) - 1);
        _mm512_mask_storeu_ps(y + i, m, _mm512_add_ps(_mm512_maskz_loadu_ps(m, y + i), _mm512_maskz_loadu_ps(m, x + i)));
    }
}

float ml_k_dot_f32_avx512vnni(const float *a, const float *b, int n) {
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) {
        const __mmask16 m = n - i >= 16 ? 0xFFFF : (__mmask16)((1u << (n - i)) - 1);
        acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i), acc);
    }
    return _mm512_reduce_add_ps(acc);
}

void ml_k_axpy_f32_avx512vnni(float *y, float a, const float *x, int n) {
    const __m512 va = _mm512_set1_ps(a);
    for (int i = 0; i < n; i += 16) {
        const __mmask16 m = n - i >= 16 ? 0xFFFF : (__mmask16)((1u << (n - i)) - 1);
        _mm512_mask_storeu_ps(y + i, m, _mm512_fmadd_ps(va, _mm512_maskz_loadu_ps(m, x + i), _mm512_maskz_loadu_ps(m, y + i)));
    }
}
