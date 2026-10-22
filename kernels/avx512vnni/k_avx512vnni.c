#include <math.h>
#include <string.h>
#include "kernels/avx512vnni/k_avx512vnni.h"
#include "kernels/x86_avx.h"

static inline __m256i dot32_avx2(__m256i w, __m256i x) {
    const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(w, x));
    return _mm256_madd_epi16(p, _mm256_set1_epi16(1));
}

static inline __m512 scale2(float s0, float s1) {
    return _mm512_mask_blend_ps(0xFF00, _mm512_set1_ps(s0), _mm512_set1_ps(s1));
}

/* Q8: |x| and x's sign mask depend only on the activation block pair, so they
 * are computed once and shared by the 4 rows; each row only negates w where
 * x < 0 (there is no 512-bit psignb) before vpdpbusd. */
void ml_k_matvec_q8_avx512vnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    for (int r = r0; r < r1; r += 4) {
        const int nr = r1 - r < 4 ? r1 - r : 4;
        __m512 acc[4];
        __m256 tail[4];
        for (int i = 0; i < 4; i++) { acc[i] = _mm512_setzero_ps(); tail[i] = _mm256_setzero_ps(); }
        int b = 0;
        for (; b + 2 <= nb; b += 2) {
            const __m512i xv = _mm512_loadu_si512((const void *)(x->q + b * QK));
            const __m512i ax = _mm512_abs_epi8(xv);
            const __mmask64 neg = _mm512_movepi8_mask(xv);
            for (int i = 0; i < nr; i++) {
                const int8_t *wq = W->q + (size_t)(r + i) * W->cols + b * QK;
                const float *wd = W->d + (size_t)(r + i) * nb + b;
                const __m512i w = _mm512_loadu_si512((const void *)wq);
                const __m512i ws = _mm512_mask_sub_epi8(w, neg, _mm512_setzero_si512(), w);
                const __m512i s = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ax, ws);
                acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), scale2(wd[0] * x->d[b], wd[1] * x->d[b + 1]), acc[i]);
            }
        }
        if (b < nb) {
            const __m256i xv = _mm256_loadu_si256((const __m256i *)(x->q + b * QK));
            for (int i = 0; i < nr; i++) {
                const __m256i s = dot32_avx2(load_q8_block(W->q + (size_t)(r + i) * W->cols + b * QK), xv);
                tail[i] = _mm256_mul_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(W->d[(size_t)(r + i) * nb + b] * x->d[b]));
            }
        }
        for (int i = 0; i < nr; i++) y[r + i] = _mm512_reduce_add_ps(acc[i]) + hsum256_ps(tail[i]);
    }
}

/* Q4: the nibbles n = w + 8 are used directly as the unsigned operand of
 * vpdpbusd, so no sign handling is needed:
 *     sum(w * x) = sum(n * x) - 8 * sum(x)
 * The second term is applied once per row as -dot(d_w, xs) with
 * xs[b] = 8 * sum(x in block b) * d_x[b], computed once per call.
 *
 * Two blocks (32 packed bytes) per step. Broadcasting them to both 256-bit
 * halves and taking low nibbles in the first half and high nibbles in the
 * second gives 128-bit lanes [b0 e0-15 | b1 e0-15 | b0 e16-31 | b1 e16-31];
 * the activations are permuted to that lane order once per block pair. */
#define Q4_MAX_BLOCKS 2048   /* cols <= 65536, enforced by the loader */

void ml_k_matvec_q4_avx512vnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    float xs[Q4_MAX_BLOCKS];
    for (int b = 0; b < nb; b++) {
        int sum = 0;
        for (int j = 0; j < QK; j++) sum += x->q[b * QK + j];
        xs[b] = 8.0f * (float)sum * x->d[b];
    }
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    for (int r = r0; r < r1; r += 4) {
        const int nr = r1 - r < 4 ? r1 - r : 4;
        __m512 acc[4];
        __m256 tail[4];
        for (int i = 0; i < 4; i++) { acc[i] = _mm512_setzero_ps(); tail[i] = _mm256_setzero_ps(); }
        int b = 0;
        for (; b + 2 <= nb; b += 2) {
            const __m512i xv = _mm512_loadu_si512((const void *)(x->q + b * QK));
            const __m512i xp = _mm512_shuffle_i64x2(xv, xv, _MM_SHUFFLE(3, 1, 2, 0));
            for (int i = 0; i < nr; i++) {
                const uint8_t *wq = (const uint8_t *)W->q + ((size_t)(r + i) * nb + b) * (QK / 2);
                const float *wd = W->d + (size_t)(r + i) * nb + b;
                const __m512i v = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)wq));
                const __m512i n = _mm512_and_si512(_mm512_mask_mov_epi64(v, 0xF0, _mm512_srli_epi16(v, 4)), m4);
                const __m512i s = _mm512_dpbusd_epi32(_mm512_setzero_si512(), n, xp);
                const __m512 sc = _mm512_mask_blend_ps(0xF0F0, _mm512_set1_ps(wd[0] * x->d[b]),
                                                      _mm512_set1_ps(wd[1] * x->d[b + 1]));
                acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), sc, acc[i]);
            }
        }
        if (b < nb) {   /* odd block: nibbles as u8 through pmaddubsw */
            const __m256i xv = _mm256_loadu_si256((const __m256i *)(x->q + b * QK));
            for (int i = 0; i < nr; i++) {
                const __m128i p = _mm_loadu_si128((const __m128i *)((const uint8_t *)W->q + ((size_t)(r + i) * nb + b) * (QK / 2)));
                const __m128i m = _mm_set1_epi8(0x0F);
                const __m256i n = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(p, 4), m), _mm_and_si128(p, m));
                const __m256i s = _mm256_madd_epi16(_mm256_maddubs_epi16(n, xv), _mm256_set1_epi16(1));
                tail[i] = _mm256_mul_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(W->d[(size_t)(r + i) * nb + b] * x->d[b]));
            }
        }
        for (int i = 0; i < nr; i++) {
            /* correction: sum_b d_w[b] * xs[b], 16 blocks per step */
            const float *wd = W->d + (size_t)(r + i) * nb;
            __m512 c = _mm512_setzero_ps();
            int k = 0;
            for (; k + 16 <= nb; k += 16) c = _mm512_fmadd_ps(_mm512_loadu_ps(wd + k), _mm512_loadu_ps(xs + k), c);
            if (k < nb) {
                const __mmask16 m = (__mmask16)((1u << (nb - k)) - 1);
                c = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, wd + k), _mm512_maskz_loadu_ps(m, xs + k), c);
            }
            y[r + i] = _mm512_reduce_add_ps(acc[i]) + hsum256_ps(tail[i]) - _mm512_reduce_add_ps(c);
        }
    }
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
