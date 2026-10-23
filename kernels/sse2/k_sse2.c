#include <math.h>
#include <string.h>
#include "kernels/scalar/k_scalar.h"
#include "kernels/sse2/k_sse2.h"
#include "kernels/x86_common.h"
#include "kernels/x86_pack.h"

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

/* Scores of 4 rows per pass: each row has its own accumulator, so the
 * arithmetic per row is the same in the 4-row and the 1-row loop. */
void ml_k_dot_f32_sse2(float *s, const float *q, const float *k, size_t ldk, int rows, int n, float scale) {
    int t = 0;
    for (; t + 4 <= rows; t += 4) {
        const float *k0 = k + (size_t)t * ldk, *k1 = k0 + ldk, *k2 = k1 + ldk, *k3 = k2 + ldk;
        __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps(), a2 = _mm_setzero_ps(), a3 = _mm_setzero_ps();
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            const __m128 qv = _mm_loadu_ps(q + i);
            a0 = _mm_add_ps(a0, _mm_mul_ps(qv, _mm_loadu_ps(k0 + i)));
            a1 = _mm_add_ps(a1, _mm_mul_ps(qv, _mm_loadu_ps(k1 + i)));
            a2 = _mm_add_ps(a2, _mm_mul_ps(qv, _mm_loadu_ps(k2 + i)));
            a3 = _mm_add_ps(a3, _mm_mul_ps(qv, _mm_loadu_ps(k3 + i)));
        }
        float r0 = hsum_ps(a0), r1 = hsum_ps(a1), r2 = hsum_ps(a2), r3 = hsum_ps(a3);
        for (; i < n; i++) { r0 += q[i] * k0[i]; r1 += q[i] * k1[i]; r2 += q[i] * k2[i]; r3 += q[i] * k3[i]; }
        s[t] = r0 * scale; s[t + 1] = r1 * scale; s[t + 2] = r2 * scale; s[t + 3] = r3 * scale;
    }
    for (; t < rows; t++) {
        const float *k0 = k + (size_t)t * ldk;
        __m128 a0 = _mm_setzero_ps();
        int i = 0;
        for (; i + 4 <= n; i += 4) a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(q + i), _mm_loadu_ps(k0 + i)));
        float r0 = hsum_ps(a0);
        for (; i < n; i++) r0 += q[i] * k0[i];
        s[t] = r0 * scale;
    }
}

/* y += sum_t a[t] * x_t, 16 outputs at a time held in registers across all
 * rows. Each output adds its terms in row order, whatever the chunk. */
void ml_k_axpy_f32_sse2(float *y, const float *a, const float *x, size_t ldx, int rows, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128 y0 = _mm_loadu_ps(y + i), y1 = _mm_loadu_ps(y + i + 4), y2 = _mm_loadu_ps(y + i + 8), y3 = _mm_loadu_ps(y + i + 12);
        const float *xt = x + i;
        for (int t = 0; t < rows; t++, xt += ldx) {
            const __m128 av = _mm_set1_ps(a[t]);
            y0 = _mm_add_ps(y0, _mm_mul_ps(av, _mm_loadu_ps(xt)));
            y1 = _mm_add_ps(y1, _mm_mul_ps(av, _mm_loadu_ps(xt + 4)));
            y2 = _mm_add_ps(y2, _mm_mul_ps(av, _mm_loadu_ps(xt + 8)));
            y3 = _mm_add_ps(y3, _mm_mul_ps(av, _mm_loadu_ps(xt + 12)));
        }
        _mm_storeu_ps(y + i, y0); _mm_storeu_ps(y + i + 4, y1); _mm_storeu_ps(y + i + 8, y2); _mm_storeu_ps(y + i + 12, y3);
    }
    for (; i + 4 <= n; i += 4) {
        __m128 y0 = _mm_loadu_ps(y + i);
        const float *xt = x + i;
        for (int t = 0; t < rows; t++, xt += ldx) y0 = _mm_add_ps(y0, _mm_mul_ps(_mm_set1_ps(a[t]), _mm_loadu_ps(xt)));
        _mm_storeu_ps(y + i, y0);
    }
    for (; i < n; i++) {
        float v = y[i];
        for (int t = 0; t < rows; t++) v += a[t] * x[(size_t)t * ldx + i];
        y[i] = v;
    }
}

/* Q8 KV cache kernels. Integer dot products widen to int16 for pmaddwd; the
 * query's widened chunk is shared by 4 rows. */
static inline int32_t hsum_epi32_sse2(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xB1));
    return _mm_cvtsi128_si32(v);
}

void ml_k_dot_q8_sse2(float *s, const int8_t *q, float dq, const int8_t *k, const float *dk, size_t ldk, size_t lddk, int rows, int n, float scale) {
    const float c = dq * scale;
    int t = 0;
    for (; t + 4 <= rows; t += 4) {
        const int8_t *k0 = k + (size_t)t * ldk;
        __m128i a[4] = { _mm_setzero_si128(), _mm_setzero_si128(), _mm_setzero_si128(), _mm_setzero_si128() };
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i ql, qh;
            sext16(_mm_loadu_si128((const __m128i *)(q + i)), &ql, &qh);
            for (int j = 0; j < 4; j++) {
                __m128i kl, kh;
                sext16(_mm_loadu_si128((const __m128i *)(k0 + (size_t)j * ldk + i)), &kl, &kh);
                a[j] = _mm_add_epi32(a[j], _mm_add_epi32(_mm_madd_epi16(ql, kl), _mm_madd_epi16(qh, kh)));
            }
        }
        for (int j = 0; j < 4; j++) {
            const int8_t *kt = k0 + (size_t)j * ldk;
            int32_t sum = hsum_epi32_sse2(a[j]);
            for (int r = i; r < n; r++) sum += q[r] * kt[r];
            s[t + j] = (float)sum * c * dk[(size_t)(t + j) * lddk];
        }
    }
    for (; t < rows; t++) {
        const int8_t *kt = k + (size_t)t * ldk;
        __m128i a0 = _mm_setzero_si128();
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i ql, qh, kl, kh;
            sext16(_mm_loadu_si128((const __m128i *)(q + i)), &ql, &qh);
            sext16(_mm_loadu_si128((const __m128i *)(kt + i)), &kl, &kh);
            a0 = _mm_add_epi32(a0, _mm_add_epi32(_mm_madd_epi16(ql, kl), _mm_madd_epi16(qh, kh)));
        }
        int32_t sum = hsum_epi32_sse2(a0);
        for (; i < n; i++) sum += q[i] * kt[i];
        s[t] = (float)sum * c * dk[(size_t)t * lddk];
    }
}

/* 16 outputs at a time held in registers across all rows. */
void ml_k_axpy_q8_sse2(float *y, const float *a, const int8_t *x, const float *dx, size_t ldx, size_t lddx, int rows, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128 y0 = _mm_loadu_ps(y + i), y1 = _mm_loadu_ps(y + i + 4), y2 = _mm_loadu_ps(y + i + 8), y3 = _mm_loadu_ps(y + i + 12);
        for (int t = 0; t < rows; t++) {
            const __m128 w = _mm_set1_ps(a[t] * dx[(size_t)t * lddx]);
            __m128i lo, hi;
            sext16(_mm_loadu_si128((const __m128i *)(x + (size_t)t * ldx + i)), &lo, &hi);
            y0 = _mm_add_ps(y0, _mm_mul_ps(w, _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(lo, lo), 16))));
            y1 = _mm_add_ps(y1, _mm_mul_ps(w, _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(lo, lo), 16))));
            y2 = _mm_add_ps(y2, _mm_mul_ps(w, _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(hi, hi), 16))));
            y3 = _mm_add_ps(y3, _mm_mul_ps(w, _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(hi, hi), 16))));
        }
        _mm_storeu_ps(y + i, y0); _mm_storeu_ps(y + i + 4, y1); _mm_storeu_ps(y + i + 8, y2); _mm_storeu_ps(y + i + 12, y3);
    }
    for (; i < n; i++) {
        float v = y[i];
        for (int t = 0; t < rows; t++) v += (a[t] * dx[(size_t)t * lddx]) * (float)x[(size_t)t * ldx + i];
        y[i] = v;
    }
}

/* Packed kernels: 4-row groups (core/pack.h). SSE2 has no 8-bit multiply, so
 * each 16-bit lane of a weight step (two bytes of one row) is split into its
 * even byte (w & 0xFF) and odd byte (w >> 8) and multiplied with pmaddwd
 * against activations expanded to match: the pairs (x0, x2) and (x1, x3) of
 * the step, broadcast. Each pmaddwd then yields one partial sum per row, with
 * no shuffles in the inner loop. Q8 is stored +128 so both halves are plain
 * unsigned bytes; 128 * sum(x) is subtracted per block.
 *
 * The expansion is 16 vectors (256 bytes) per activation block, built for XC
 * blocks at a time; one chunk covers 8192 columns. Between chunks a group's float accumulators are parked in y, which
 * keeps the order of float operations, and so the results, independent of XC. */
#define PK 4
#define XC 256

/* Expanded activations, PK_TILE tokens of XC blocks each (256 KB). */
static _Thread_local __m128i pk_xe[PK_TILE][XC * 16];

static void pk_expand(__m128i *o, const int8_t *q, int nblocks) {
    for (int i = 0; i < nblocks * 8; i++, q += 4) {
        o[2 * i]     = _mm_set1_epi32((int)(uint16_t)q[0] | (int)((uint32_t)(uint16_t)q[2] << 16));
        o[2 * i + 1] = _mm_set1_epi32((int)(uint16_t)q[1] | (int)((uint32_t)(uint16_t)q[3] << 16));
    }
}

/* Exact integer sums of one Q8 block for the group's 4 rows. */
static inline __m128i pk_q8_block(const uint8_t *p, const __m128i *xv) {
    const __m128i m8 = _mm_set1_epi16(0xFF);
    __m128i s0 = _mm_setzero_si128(), s1 = _mm_setzero_si128();
    for (int k = 0; k < 8; k++, p += PK * 4, xv += 2) {
        const __m128i w = _mm_loadu_si128((const __m128i *)p);
        s0 = _mm_add_epi32(s0, _mm_madd_epi16(_mm_and_si128(w, m8), xv[0]));
        s1 = _mm_add_epi32(s1, _mm_madd_epi16(_mm_srli_epi16(w, 8), xv[1]));
    }
    return _mm_add_epi32(s0, s1);
}

/* Q4: vector v holds step v in its low nibbles and step v + 4 in its high ones. */
static inline __m128i pk_q4_block(const uint8_t *p, const __m128i *xv) {
    const __m128i m4 = _mm_set1_epi8(0x0F), m8 = _mm_set1_epi16(0xFF);
    __m128i s0 = _mm_setzero_si128(), s1 = _mm_setzero_si128();
    for (int v = 0; v < 4; v++, p += PK * 4) {
        const __m128i w = _mm_loadu_si128((const __m128i *)p);
        const __m128i a = _mm_and_si128(w, m4), c = _mm_and_si128(_mm_srli_epi16(w, 4), m4);
        s0 = _mm_add_epi32(s0, _mm_madd_epi16(_mm_and_si128(a, m8), xv[2 * v]));
        s1 = _mm_add_epi32(s1, _mm_madd_epi16(_mm_srli_epi16(a, 8), xv[2 * v + 1]));
        s0 = _mm_add_epi32(s0, _mm_madd_epi16(_mm_and_si128(c, m8), xv[2 * v + 8]));
        s1 = _mm_add_epi32(s1, _mm_madd_epi16(_mm_srli_epi16(c, 8), xv[2 * v + 9]));
    }
    return _mm_add_epi32(s0, s1);
}

/* A group's accumulators: in y while the group is whole, else in tail. */
static inline __m128 pk_acc_load(const float *y, const float *tail, int row, int r1) {
    return _mm_loadu_ps(row + PK <= r1 ? y + row : tail);
}

static inline void pk_acc_store(float *y, float *tail, int row, int r1, __m128 acc) {
    _mm_storeu_ps(row + PK <= r1 ? y + row : tail, acc);
}

#define PK_MATVEC(BLOCK, STEPS, OFFSET)                                           \
    const int nb = W->cols / QK;                                                  \
    int xsum[PK_MAX_BLOCKS];                                                      \
    __m128i *xe = pk_xe[0];                                                       \
    float tail[PK];                                                               \
    pk_block_sums(xsum, x, nb, OFFSET);                                           \
    for (int b0 = 0; b0 < nb; b0 += XC) {                                         \
        const int b1 = nb - b0 < XC ? nb : b0 + XC;                               \
        pk_expand(xe, x->q + b0 * QK, b1 - b0);                                   \
        for (int row = r0; row < r1; row += PK) {                                 \
            const uint8_t *p = W->pq + ((size_t)(row / PK) * nb + b0) * (STEPS * PK * 4); \
            const float *d = W->pd + (size_t)(row / PK) * nb * PK;                \
            __m128 acc = b0 ? pk_acc_load(y, tail, row, r1) : _mm_setzero_ps();   \
            for (int b = b0; b < b1; b++, p += STEPS * PK * 4) {                  \
                const __m128i s = _mm_sub_epi32(BLOCK(p, xe + (b - b0) * 16), _mm_set1_epi32(xsum[b])); \
                acc = _mm_add_ps(acc, _mm_mul_ps(_mm_cvtepi32_ps(s), _mm_mul_ps(_mm_loadu_ps(d + b * PK), _mm_set1_ps(x->d[b])))); \
            }                                                                     \
            pk_acc_store(y, tail, row, r1, acc);                                  \
        }                                                                         \
    }                                                                             \
    if ((r1 - r0) % PK) pk_store(y, tail, r1 - (r1 - r0) % PK, PK, r1);

void ml_k_matvec_q8_pk_sse2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    PK_MATVEC(pk_q8_block, 8, 128)
}

void ml_k_matvec_q4_pk_sse2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    PK_MATVEC(pk_q4_block, 4, 8)
}

/* Batched: each group's weights are read once per tile of PK_TILE tokens;
 * per token the arithmetic is the matvec's, so the results match it exactly. */
#define PK_MATMUL_TILE(BLOCK, STEPS, OFFSET)                                      \
    const int nb = W->cols / QK;                                                  \
    static _Thread_local int xsum[PK_TILE][PK_MAX_BLOCKS];                        \
    __m128i (*xe)[XC * 16] = pk_xe;                                               \
    float tail[PK_TILE][PK];                                                      \
    pk_block_sums_batch(xsum, X, t0, T, nb, OFFSET);                              \
    for (int b0 = 0; b0 < nb; b0 += XC) {                                         \
        const int b1 = nb - b0 < XC ? nb : b0 + XC;                               \
        for (int t = 0; t < T; t++)                                               \
            pk_expand(xe[t], X->q + (size_t)(t0 + t) * X->k + b0 * QK, b1 - b0);  \
        for (int row = r0; row < r1; row += PK) {                                 \
            const uint8_t *p = W->pq + ((size_t)(row / PK) * nb + b0) * (STEPS * PK * 4); \
            const float *d = W->pd + (size_t)(row / PK) * nb * PK;                \
            __m128 acc[PK_TILE];                                                  \
            ML_UNROLL(4) for (int t = 0; t < T; t++)                              \
                acc[t] = b0 ? pk_acc_load(Y + (size_t)(t0 + t) * ldy, tail[t], row, r1) : _mm_setzero_ps(); \
            for (int b = b0; b < b1; b++, p += STEPS * PK * 4) {                  \
                const __m128 dv = _mm_loadu_ps(d + b * PK);                       \
                ML_UNROLL(4) for (int t = 0; t < T; t++) {                        \
                    const __m128i s = _mm_sub_epi32(BLOCK(p, xe[t] + (b - b0) * 16), _mm_set1_epi32(xsum[t][b])); \
                    acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_cvtepi32_ps(s), _mm_mul_ps(dv, _mm_set1_ps(X->d[(size_t)(t0 + t) * nb + b])))); \
                }                                                                 \
            }                                                                     \
            ML_UNROLL(4) for (int t = 0; t < T; t++)                              \
                pk_acc_store(Y + (size_t)(t0 + t) * ldy, tail[t], row, r1, acc[t]); \
        }                                                                         \
    }                                                                             \
    if ((r1 - r0) % PK)                                                           \
        for (int t = 0; t < T; t++)                                               \
            pk_store(Y + (size_t)(t0 + t) * ldy, tail[t], r1 - (r1 - r0) % PK, PK, r1);

/* One tile of T tokens; T is a constant in every call (PK_TILES). */
ML_INLINE void matmul_q8_tile(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1, int t0, const int T) {
    PK_MATMUL_TILE(pk_q8_block, 8, 128)
}

ML_INLINE void matmul_q4_tile(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1, int t0, const int T) {
    PK_MATMUL_TILE(pk_q4_block, 4, 8)
}

void ml_k_matmul_q8_pk_sse2(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    PK_TILES(matmul_q8_tile, PK_TILE)
}

void ml_k_matmul_q4_pk_sse2(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    PK_TILES(matmul_q4_tile, PK_TILE)
}
