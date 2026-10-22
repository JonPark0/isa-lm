#include <math.h>
#include <string.h>
#include "kernels/scalar/k_scalar.h"
#include "kernels/sse2/k_sse2.h"
#include "kernels/x86_common.h"

/* int8 x int8 -> 4 x int32 partial sums for 16 elements, given x already
 * sign-extended to int16 (xl = elements 0..7, xh = 8..15). */
static inline __m128i dot16_sse2(__m128i w, __m128i xl, __m128i xh) {
    const __m128i wl = _mm_srai_epi16(_mm_unpacklo_epi8(w, w), 8);
    const __m128i wh = _mm_srai_epi16(_mm_unpackhi_epi8(w, w), 8);
    return _mm_add_epi32(_mm_madd_epi16(wl, xl), _mm_madd_epi16(wh, xh));
}

static inline void sext16(__m128i v, __m128i *lo, __m128i *hi) {
    *lo = _mm_srai_epi16(_mm_unpacklo_epi8(v, v), 8);
    *hi = _mm_srai_epi16(_mm_unpackhi_epi8(v, v), 8);
}

/* Shared row loop: 4 rows per pass reuse one activation block load. LOADW
 * yields the two int8 vectors (elements 0..15, 16..31) of row r, block b. */
#define MATVEC_BODY(LOADW)                                                        \
    const int nb = W->cols / QK;                                                  \
    int r = r0;                                                                   \
    for (; r + 4 <= r1; r += 4) {                                                 \
        __m128 acc[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() }; \
        for (int b = 0; b < nb; b++) {                                            \
            __m128i x0l, x0h, x1l, x1h;                                           \
            sext16(_mm_loadu_si128((const __m128i *)(x->q + b * QK)), &x0l, &x0h); \
            sext16(_mm_loadu_si128((const __m128i *)(x->q + b * QK + 16)), &x1l, &x1h); \
            for (int i = 0; i < 4; i++) {                                         \
                __m128i w0, w1;                                                   \
                LOADW(r + i, b, w0, w1);                                          \
                const __m128i s = _mm_add_epi32(dot16_sse2(w0, x0l, x0h), dot16_sse2(w1, x1l, x1h)); \
                const float sc = W->d[(size_t)(r + i) * nb + b] * x->d[b];        \
                acc[i] = _mm_add_ps(acc[i], _mm_mul_ps(_mm_cvtepi32_ps(s), _mm_set1_ps(sc))); \
            }                                                                     \
        }                                                                         \
        for (int i = 0; i < 4; i++) y[r + i] = hsum_ps(acc[i]);                   \
    }                                                                             \
    for (; r < r1; r++) {                                                         \
        __m128 acc = _mm_setzero_ps();                                            \
        for (int b = 0; b < nb; b++) {                                            \
            __m128i x0l, x0h, x1l, x1h, w0, w1;                                   \
            sext16(_mm_loadu_si128((const __m128i *)(x->q + b * QK)), &x0l, &x0h); \
            sext16(_mm_loadu_si128((const __m128i *)(x->q + b * QK + 16)), &x1l, &x1h); \
            LOADW(r, b, w0, w1);                                                  \
            const __m128i s = _mm_add_epi32(dot16_sse2(w0, x0l, x0h), dot16_sse2(w1, x1l, x1h)); \
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

void ml_k_matvec_q8_sse2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC_BODY(LOAD_Q8)
}

void ml_k_matvec_q4_sse2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC_BODY(LOAD_Q4)
}

void ml_k_quantize_q8_sse2(ml_qact *out, const float *x, int k) {
    const __m128 sign = _mm_set1_ps(-0.0f), half = _mm_set1_ps(0.5f);
    for (int b = 0; b < k / QK; b++) {
        const float *xb = x + b * QK;
        __m128 v[8], m = _mm_setzero_ps();
        for (int j = 0; j < 8; j++) {
            v[j] = _mm_loadu_ps(xb + 4 * j);
            m = _mm_max_ps(m, abs_ps(v[j]));
        }
        const float amax = hmax_ps(m);
        if (amax < ML_QUANT_TINY) {
            out->d[b] = 0.0f;
            memset(out->q + b * QK, 0, QK);
            continue;
        }
        const float id = 127.0f / amax;
        out->d[b] = amax / 127.0f;
        const __m128 vid = _mm_set1_ps(id);
        __m128i i32[8];
        for (int j = 0; j < 8; j++) {
            const __m128 t = _mm_mul_ps(v[j], vid);
            i32[j] = _mm_cvttps_epi32(_mm_add_ps(t, _mm_or_ps(_mm_and_ps(v[j], sign), half)));
        }
        const __m128i lim = _mm_set1_epi16(-127);
        for (int j = 0; j < 2; j++) {
            const __m128i a = _mm_max_epi16(_mm_packs_epi32(i32[4 * j], i32[4 * j + 1]), lim);
            const __m128i c = _mm_max_epi16(_mm_packs_epi32(i32[4 * j + 2], i32[4 * j + 3]), lim);
            _mm_storeu_si128((__m128i *)(out->q + b * QK + 16 * j), _mm_packs_epi16(a, c));
        }
    }
}

void ml_k_rmsnorm_sse2(float *o, const float *x, const float *w, int n, float eps) {
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m128 v = _mm_loadu_ps(x + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(v, v));
    }
    float ss = hsum_ps(acc);
    for (; i < n; i++) ss += x[i] * x[i];
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    const __m128 vinv = _mm_set1_ps(inv);
    for (i = 0; i + 4 <= n; i += 4)
        _mm_storeu_ps(o + i, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(x + i), vinv), _mm_loadu_ps(w + i)));
    for (; i < n; i++) o[i] = x[i] * inv * w[i];
}

void ml_k_softmax_sse2(float *x, int n) {
    int i = 0;
    __m128 vm = _mm_set1_ps(x[0]);
    for (; i + 4 <= n; i += 4) vm = _mm_max_ps(vm, _mm_loadu_ps(x + i));
    float mx = hmax_ps(vm);
    for (; i < n; i++) mx = fmaxf(mx, x[i]);
    if (mx == -INFINITY) {   /* every entry masked: uniform */
        for (i = 0; i < n; i++) x[i] = 1.0f / (float)n;
        return;
    }
    const __m128 vmx = _mm_set1_ps(mx);
    __m128 vs = _mm_setzero_ps();
    for (i = 0; i + 4 <= n; i += 4) {
        const __m128 e = exp_ps(_mm_sub_ps(_mm_loadu_ps(x + i), vmx));
        _mm_storeu_ps(x + i, e);
        vs = _mm_add_ps(vs, e);
    }
    float sum = hsum_ps(vs);
    for (; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    const float inv = 1.0f / sum;
    const __m128 vinv = _mm_set1_ps(inv);
    for (i = 0; i + 4 <= n; i += 4) _mm_storeu_ps(x + i, _mm_mul_ps(_mm_loadu_ps(x + i), vinv));
    for (; i < n; i++) x[i] *= inv;
}

static void rope_heads_sse2(float *v, const float *c, const float *s, int hd, int nh) {
    const __m128 alt = _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f);   /* lanes 0..3: -,+,-,+ */
    for (int h = 0; h < nh; h++) {
        float *p = v + h * hd;
        int i = 0;   /* pair index */
        for (; 2 * i + 4 <= hd; i += 2) {
            const __m128 a = _mm_loadu_ps(p + 2 * i);
            const __m128 cc = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *)(c + i)));
            const __m128 ss = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *)(s + i)));
            const __m128 cd = _mm_unpacklo_ps(cc, cc), sd = _mm_unpacklo_ps(ss, ss);
            const __m128 sw = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
            _mm_storeu_ps(p + 2 * i, _mm_add_ps(_mm_mul_ps(a, cd), _mm_mul_ps(_mm_mul_ps(sw, sd), alt)));
        }
        for (; 2 * i < hd; i++) {
            const float a = p[2 * i], b = p[2 * i + 1];
            p[2 * i] = a * c[i] - b * s[i];
            p[2 * i + 1] = a * s[i] + b * c[i];
        }
    }
}

void ml_k_rope_sse2(float *q, float *k, const float *cos_t, const float *sin_t,
                    int head_dim, int n_heads, int n_kv_heads) {
    rope_heads_sse2(q, cos_t, sin_t, head_dim, n_heads);
    rope_heads_sse2(k, cos_t, sin_t, head_dim, n_kv_heads);
}

void ml_k_silu_mul_sse2(float *g, const float *u, int n) {
    const __m128 one = _mm_set1_ps(1.0f), zero = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m128 v = _mm_loadu_ps(g + i);
        const __m128 sig = _mm_div_ps(v, _mm_add_ps(one, exp_ps(_mm_sub_ps(zero, v))));
        _mm_storeu_ps(g + i, _mm_mul_ps(sig, _mm_loadu_ps(u + i)));
    }
    for (; i < n; i++) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}

#ifndef ML_ASM_ADD_SSE2   /* ASM=1: kernels/sse2/add.S */
void ml_k_add_sse2(float *y, const float *x, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) _mm_storeu_ps(y + i, _mm_add_ps(_mm_loadu_ps(y + i), _mm_loadu_ps(x + i)));
    for (; i < n; i++) y[i] += x[i];
}
#endif

float ml_k_dot_f32_sse2(const float *a, const float *b, int n) {
    __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
        a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4)));
    }
    for (; i + 4 <= n; i += 4) a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    float s = hsum_ps(_mm_add_ps(a0, a1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

void ml_k_axpy_f32_sse2(float *y, float a, const float *x, int n) {
    const __m128 va = _mm_set1_ps(a);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        _mm_storeu_ps(y + i, _mm_add_ps(_mm_loadu_ps(y + i), _mm_mul_ps(va, _mm_loadu_ps(x + i))));
    for (; i < n; i++) y[i] += a * x[i];
}
