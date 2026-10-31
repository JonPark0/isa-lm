#include <math.h>
#include <string.h>
#include "kernels/avx2/k_avx2.h"
#include "kernels/x86_avx.h"
#include "kernels/x86_pack.h"

/* 32 int8 pairs -> 8 x int32 via |x| (u8) * w-with-x's-sign (s8). Quants are
 * within [-127, 127], so pmaddubsw's int16 pair sums cannot saturate. */
static inline __m256i dot32_avx2(__m256i w, __m256i x) {
    const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(w, x));
    return _mm256_madd_epi16(p, _mm256_set1_epi16(1));
}

void ml_k_matvec_q8_avx2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC256_BODY(LOADW_Q8_256, dot32_avx2)
}

void ml_k_matvec_q4_avx2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC256_BODY(LOADW_Q4_256, dot32_avx2)
}

/* Packed kernels: 8 rows per group, one 32-bit lane per row.
 * Q8 (signed weights): |x| of the 4-byte broadcast is shared by all 8 rows;
 * each step is psignb + pmaddubsw + pmaddwd.
 * Q4: the raw nibbles are pmaddubsw's unsigned operand. A nibble pair sum is
 * at most 2*15*127 = 3810, so all 8 pmaddubsw results of a block add up in
 * int16 (<= 30480) before a single pmaddwd; then 8 * sum(x) is subtracted. */
#define PK 8

ML_INLINE void matvec_q8_pk_impl(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1, const int f16) {
    const int nb = W->cols / QK;
    const __m256i ones = _mm256_set1_epi16(1);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (8 * PK * 4);
        const void *d = pk_d(W, (size_t)(row / PK) * nb * PK, f16);
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++, p += 8 * PK * 4) {
            ml_prefetch_ahead(p, 8 * PK * 4);
            const int32_t *xb = (const int32_t *)(x->q + b * QK);
            __m256i s = _mm256_setzero_si256();
            for (int k = 0; k < 8; k++) {
                const __m256i xv = _mm256_set1_epi32(xb[k]);
                const __m256i w = _mm256_loadu_si256((const __m256i *)(p + k * PK * 4));
                const __m256i prod = _mm256_maddubs_epi16(_mm256_sign_epi8(xv, xv), _mm256_sign_epi8(w, xv));
                s = _mm256_add_epi32(s, _mm256_madd_epi16(prod, ones));
            }
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(pk_d8(d, b * PK, f16), _mm256_set1_ps(x->d[b])), acc);
        }
        float t[PK];
        _mm256_storeu_ps(t, acc);
        pk_store(y, t, row, PK, r1);
    }
}

/* f16 is a constant in each instance, so the scale load is chosen at compile time. */
void ml_k_matvec_q8_pk_avx2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    if (W->pd16) matvec_q8_pk_impl(y, W, x, r0, r1, 1);
    else matvec_q8_pk_impl(y, W, x, r0, r1, 0);
}

ML_INLINE void matvec_q4_pk_impl(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1, const int f16) {
    const int nb = W->cols / QK;
    int xsum[PK_MAX_BLOCKS];
    pk_block_sums(xsum, x, nb, 8);
    const __m256i m4 = _mm256_set1_epi8(0x0F), ones = _mm256_set1_epi16(1);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (4 * PK * 4);
        const void *d = pk_d(W, (size_t)(row / PK) * nb * PK, f16);
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++, p += 4 * PK * 4) {
            ml_prefetch_ahead(p, 4 * PK * 4);
            const int32_t *xb = (const int32_t *)(x->q + b * QK);
            __m256i s16 = _mm256_setzero_si256();
            for (int v = 0; v < 4; v++) {
                const __m256i w = _mm256_loadu_si256((const __m256i *)(p + v * PK * 4));
                s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(_mm256_and_si256(w, m4), _mm256_set1_epi32(xb[v])));
                s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(w, 4), m4),
                                                                 _mm256_set1_epi32(xb[v + 4])));
            }
            const __m256i s = _mm256_sub_epi32(_mm256_madd_epi16(s16, ones), _mm256_set1_epi32(xsum[b]));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(pk_d8(d, b * PK, f16), _mm256_set1_ps(x->d[b])), acc);
        }
        float t[PK];
        _mm256_storeu_ps(t, acc);
        pk_store(y, t, row, PK, r1);
    }
}

/* f16 is a constant in each instance, so the scale load is chosen at compile time. */
void ml_k_matvec_q4_pk_avx2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    if (W->pd16) matvec_q4_pk_impl(y, W, x, r0, r1, 1);
    else matvec_q4_pk_impl(y, W, x, r0, r1, 0);
}

void ml_k_quantize_q8_avx2(ml_qact *out, const float *x, int k) {
    const __m256 sign = _mm256_set1_ps(-0.0f), half = _mm256_set1_ps(0.5f);
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int b = 0; b < k / QK; b++) {
        const float *xb = x + b * QK;
        __m256 v[4], m = _mm256_setzero_ps();
        for (int j = 0; j < 4; j++) {
            v[j] = _mm256_loadu_ps(xb + 8 * j);
            m = _mm256_max_ps(m, _mm256_andnot_ps(sign, v[j]));
        }
        const float amax = hmax256_ps(m);
        if (amax < ML_QUANT_TINY) {
            out->d[b] = 0.0f;
            memset(out->q + b * QK, 0, QK);
            continue;
        }
        const float id = 127.0f / amax;
        out->d[b] = amax / 127.0f;
        const __m256 vid = _mm256_set1_ps(id);
        __m256i i32[4];
        for (int j = 0; j < 4; j++) {
            const __m256 t = _mm256_mul_ps(v[j], vid);   /* no FMA: must match scalar rounding */
            i32[j] = _mm256_cvttps_epi32(_mm256_add_ps(t, _mm256_or_ps(_mm256_and_ps(v[j], sign), half)));
        }
        __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(i32[0], i32[1]), _mm256_packs_epi32(i32[2], i32[3]));
        p = _mm256_max_epi8(_mm256_permutevar8x32_epi32(p, order), _mm256_set1_epi8(-127));
        _mm256_storeu_si256((__m256i *)(out->q + b * QK), p);
    }
}

void ml_k_rmsnorm_avx2(float *o, const float *x, const float *w, int n, float eps) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    float ss = hsum256_ps(acc);
    for (; i < n; i++) ss += x[i] * x[i];
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    const __m256 vinv = _mm256_set1_ps(inv);
    for (i = 0; i + 8 <= n; i += 8)
        _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), vinv), _mm256_loadu_ps(w + i)));
    for (; i < n; i++) o[i] = x[i] * inv * w[i];
}

void ml_k_softmax_avx2(float *x, int n) {
    int i = 0;
    __m256 vm = _mm256_set1_ps(x[0]);
    for (; i + 8 <= n; i += 8) vm = _mm256_max_ps(vm, _mm256_loadu_ps(x + i));
    float mx = hmax256_ps(vm);
    for (; i < n; i++) mx = fmaxf(mx, x[i]);
    if (mx == -INFINITY) {   /* every entry masked: uniform */
        for (i = 0; i < n; i++) x[i] = 1.0f / (float)n;
        return;
    }
    const __m256 vmx = _mm256_set1_ps(mx);
    __m256 vs = _mm256_setzero_ps();
    for (i = 0; i + 8 <= n; i += 8) {
        const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), vmx));
        _mm256_storeu_ps(x + i, e);
        vs = _mm256_add_ps(vs, e);
    }
    float sum = hsum256_ps(vs);
    for (; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    const float inv = 1.0f / sum;
    const __m256 vinv = _mm256_set1_ps(inv);
    for (i = 0; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
    for (; i < n; i++) x[i] *= inv;
}

static void rope_heads_avx2(float *v, const float *c, const float *s, int hd, int nh) {
    for (int h = 0; h < nh; h++) {
        float *p = v + h * hd;
        int i = 0;   /* pair index */
        for (; 2 * i + 8 <= hd; i += 4) {
            const __m256 a = _mm256_loadu_ps(p + 2 * i);
            const __m128 c4 = _mm_loadu_ps(c + i), s4 = _mm_loadu_ps(s + i);
            const __m256 cd = _mm256_set_m128(_mm_unpackhi_ps(c4, c4), _mm_unpacklo_ps(c4, c4));
            const __m256 sd = _mm256_set_m128(_mm_unpackhi_ps(s4, s4), _mm_unpacklo_ps(s4, s4));
            const __m256 sw = _mm256_permute_ps(a, 0xB1);   /* swap within pairs */
            /* even lanes: a*c - b*s, odd lanes: b*c + a*s */
            _mm256_storeu_ps(p + 2 * i, _mm256_fmaddsub_ps(a, cd, _mm256_mul_ps(sw, sd)));
        }
        for (; 2 * i < hd; i++) {
            const float a = p[2 * i], b = p[2 * i + 1];
            p[2 * i] = a * c[i] - b * s[i];
            p[2 * i + 1] = a * s[i] + b * c[i];
        }
    }
}

void ml_k_rope_avx2(float *q, float *k, const float *cos_t, const float *sin_t,
                    int head_dim, int n_heads, int n_kv_heads) {
    rope_heads_avx2(q, cos_t, sin_t, head_dim, n_heads);
    rope_heads_avx2(k, cos_t, sin_t, head_dim, n_kv_heads);
}

void ml_k_silu_mul_avx2(float *g, const float *u, int n) {
    const __m256 one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(g + i);
        const __m256 sig = _mm256_div_ps(v, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(zero, v))));
        _mm256_storeu_ps(g + i, _mm256_mul_ps(sig, _mm256_loadu_ps(u + i)));
    }
    for (; i < n; i++) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}

void ml_k_add_avx2(float *y, const float *x, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(y + i), _mm256_loadu_ps(x + i)));
    for (; i < n; i++) y[i] += x[i];
}

/* Scores of 4 rows per pass: each row has its own accumulator, so the
 * arithmetic per row is the same in the 4-row and the 1-row loop. */
void ml_k_dot_f32_avx2(float *s, const float *q, const float *k, size_t ldk, int rows, int n, float scale) {
    int t = 0;
    for (; t + 4 <= rows; t += 4) {
        const float *k0 = k + (size_t)t * ldk, *k1 = k0 + ldk, *k2 = k1 + ldk, *k3 = k2 + ldk;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            const __m256 qv = _mm256_loadu_ps(q + i);
            a0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0 + i), a0);
            a1 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k1 + i), a1);
            a2 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k2 + i), a2);
            a3 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k3 + i), a3);
        }
        float r0 = hsum256_ps(a0), r1 = hsum256_ps(a1), r2 = hsum256_ps(a2), r3 = hsum256_ps(a3);
        for (; i < n; i++) { r0 += q[i] * k0[i]; r1 += q[i] * k1[i]; r2 += q[i] * k2[i]; r3 += q[i] * k3[i]; }
        s[t] = r0 * scale; s[t + 1] = r1 * scale; s[t + 2] = r2 * scale; s[t + 3] = r3 * scale;
    }
    for (; t < rows; t++) {
        const float *k0 = k + (size_t)t * ldk;
        __m256 a0 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(k0 + i), a0);
        float r0 = hsum256_ps(a0);
        for (; i < n; i++) r0 += q[i] * k0[i];
        s[t] = r0 * scale;
    }
}

/* y += sum_t a[t] * x_t, 32 outputs at a time held in registers across all
 * rows. Each output adds its terms in row order, whatever the chunk. */
void ml_k_axpy_f32_avx2(float *y, const float *a, const float *x, size_t ldx, int rows, int n) {
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 y0 = _mm256_loadu_ps(y + i), y1 = _mm256_loadu_ps(y + i + 8), y2 = _mm256_loadu_ps(y + i + 16), y3 = _mm256_loadu_ps(y + i + 24);
        const float *xt = x + i;
        for (int t = 0; t < rows; t++, xt += ldx) {
            const __m256 av = _mm256_set1_ps(a[t]);
            y0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(xt), y0);
            y1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(xt + 8), y1);
            y2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(xt + 16), y2);
            y3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(xt + 24), y3);
        }
        _mm256_storeu_ps(y + i, y0); _mm256_storeu_ps(y + i + 8, y1); _mm256_storeu_ps(y + i + 16, y2); _mm256_storeu_ps(y + i + 24, y3);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 y0 = _mm256_loadu_ps(y + i);
        const float *xt = x + i;
        for (int t = 0; t < rows; t++, xt += ldx) y0 = _mm256_fmadd_ps(_mm256_set1_ps(a[t]), _mm256_loadu_ps(xt), y0);
        _mm256_storeu_ps(y + i, y0);
    }
    for (; i < n; i++) {
        float v = y[i];
        for (int t = 0; t < rows; t++) v += a[t] * x[(size_t)t * ldx + i];
        y[i] = v;
    }
}

/* Q8 KV cache kernels: 16 quants per step widened with vpmovsxbw for
 * vpmaddwd; the query's widened chunk is shared by 4 rows. */
static inline int32_t hsum256_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

void ml_k_dot_q8_avx2(float *s, const int8_t *q, float dq, const int8_t *k, const float *dk, size_t ldk, size_t lddk, int rows, int n, float scale) {
    const float c = dq * scale;
    int t = 0;
    for (; t + 4 <= rows; t += 4) {
        const int8_t *k0 = k + (size_t)t * ldk, *k1 = k0 + ldk, *k2 = k1 + ldk, *k3 = k2 + ldk;
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            const __m256i qv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(q + i)));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(qv, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(k0 + i)))));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(qv, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(k1 + i)))));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(qv, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(k2 + i)))));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(qv, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(k3 + i)))));
        }
        int32_t r0 = hsum256_epi32(a0), r1 = hsum256_epi32(a1), r2 = hsum256_epi32(a2), r3 = hsum256_epi32(a3);
        for (; i < n; i++) { r0 += q[i] * k0[i]; r1 += q[i] * k1[i]; r2 += q[i] * k2[i]; r3 += q[i] * k3[i]; }
        s[t] = (float)r0 * c * dk[(size_t)t * lddk];
        s[t + 1] = (float)r1 * c * dk[(size_t)(t + 1) * lddk];
        s[t + 2] = (float)r2 * c * dk[(size_t)(t + 2) * lddk];
        s[t + 3] = (float)r3 * c * dk[(size_t)(t + 3) * lddk];
    }
    for (; t < rows; t++) {
        const int8_t *kt = k + (size_t)t * ldk;
        __m256i a0 = _mm256_setzero_si256();
        int i = 0;
        for (; i + 16 <= n; i += 16)
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(q + i))),
                                                        _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(kt + i)))));
        int32_t sum = hsum256_epi32(a0);
        for (; i < n; i++) sum += q[i] * kt[i];
        s[t] = (float)sum * c * dk[(size_t)t * lddk];
    }
}

/* 32 outputs at a time held in registers across all rows. */
void ml_k_axpy_q8_avx2(float *y, const float *a, const int8_t *x, const float *dx, size_t ldx, size_t lddx, int rows, int n) {
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 y0 = _mm256_loadu_ps(y + i), y1 = _mm256_loadu_ps(y + i + 8), y2 = _mm256_loadu_ps(y + i + 16), y3 = _mm256_loadu_ps(y + i + 24);
        for (int t = 0; t < rows; t++) {
            const int8_t *xt = x + (size_t)t * ldx + i;
            const __m256 w = _mm256_set1_ps(a[t] * dx[(size_t)t * lddx]);
            y0 = _mm256_fmadd_ps(w, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)xt))), y0);
            y1 = _mm256_fmadd_ps(w, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(xt + 8)))), y1);
            y2 = _mm256_fmadd_ps(w, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(xt + 16)))), y2);
            y3 = _mm256_fmadd_ps(w, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(xt + 24)))), y3);
        }
        _mm256_storeu_ps(y + i, y0); _mm256_storeu_ps(y + i + 8, y1); _mm256_storeu_ps(y + i + 16, y2); _mm256_storeu_ps(y + i + 24, y3);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 y0 = _mm256_loadu_ps(y + i);
        for (int t = 0; t < rows; t++)
            y0 = _mm256_fmadd_ps(_mm256_set1_ps(a[t] * dx[(size_t)t * lddx]),
                                 _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(x + (size_t)t * ldx + i)))), y0);
        _mm256_storeu_ps(y + i, y0);
    }
    for (; i < n; i++) {
        float v = y[i];
        for (int t = 0; t < rows; t++) v += (a[t] * dx[(size_t)t * lddx]) * (float)x[(size_t)t * ldx + i];
        y[i] = v;
    }
}

/* Batched packed kernels: each weight vector feeds up to PK_TILE tokens;
 * per token the arithmetic is the matvec's, so results match it exactly. */
/* |x| of the tile's activations, made once per tile: the products take |x|
 * (unsigned) times w with x's sign, so this saves a psignb per token and step. */
static ML_THREAD_LOCAL int8_t xabs[PK_TILE][PK_MAX_BLOCKS * QK];

ML_INLINE void matmul_q8_tile(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1, int t0, const int T, const int f16) {
    const int nb = W->cols / QK;
    const __m256i ones = _mm256_set1_epi16(1);
    ML_UNROLL(4) for (int t = 0; t < T; t++)
        for (int i = 0; i < X->k; i += 32)
            _mm256_storeu_si256((__m256i *)(xabs[t] + i),
                                _mm256_abs_epi8(_mm256_loadu_si256((const __m256i *)(X->q + (size_t)(t0 + t) * X->k + i))));
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (8 * PK * 4);
    const void *d = pk_d(W, (size_t)(row / PK) * nb * PK, f16);
        __m256 acc[PK_TILE];
        ML_UNROLL(4) for (int t = 0; t < PK_TILE; t++) acc[t] = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++, p += 8 * PK * 4) {
            __m256i s[PK_TILE];
            ML_UNROLL(4) for (int t = 0; t < PK_TILE; t++) s[t] = _mm256_setzero_si256();
            for (int k = 0; k < 8; k++) {
                const __m256i w = _mm256_loadu_si256((const __m256i *)(p + k * PK * 4));
                ML_UNROLL(4) for (int t = 0; t < T; t++) {
                    const __m256i xv = _mm256_set1_epi32(((const int32_t *)(X->q + (size_t)(t0 + t) * X->k + b * QK))[k]);
                    const __m256i av = _mm256_set1_epi32(((const int32_t *)(xabs[t] + b * QK))[k]);
                    s[t] = _mm256_add_epi32(s[t], _mm256_madd_epi16(_mm256_maddubs_epi16(av, _mm256_sign_epi8(w, xv)), ones));
                }
            }
            const __m256 dv = pk_d8(d, b * PK, f16);
            ML_UNROLL(4) for (int t = 0; t < T; t++) acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s[t]), _mm256_mul_ps(dv, _mm256_set1_ps(X->d[(size_t)(t0 + t) * nb + b])), acc[t]);
        }
        ML_UNROLL(4) for (int t = 0; t < T; t++) {
            float o[PK];
            _mm256_storeu_ps(o, acc[t]);
            pk_store(Y + (size_t)(t0 + t) * ldy, o, row, PK, r1);
        }
    }
}

/* the scale type as a constant per instance (pk_d8 / pk_d16) */
#define matmul_q8_tile_h(Y, ldy, W, X, r0, r1, t0, T) matmul_q8_tile(Y, ldy, W, X, r0, r1, t0, T, 1)
#define matmul_q8_tile_f(Y, ldy, W, X, r0, r1, t0, T) matmul_q8_tile(Y, ldy, W, X, r0, r1, t0, T, 0)

void ml_k_matmul_q8_pk_avx2(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    if (W->pd16) { PK_TILES(matmul_q8_tile_h, PK_TILE) }
    else { PK_TILES(matmul_q8_tile_f, PK_TILE) }
}

ML_INLINE void matmul_q4_tile(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1, int t0, const int T, const int f16) {
    const int nb = W->cols / QK;
    static ML_THREAD_LOCAL int xsum[PK_TILE][PK_MAX_BLOCKS];
    const __m256i m4 = _mm256_set1_epi8(0x0F), ones = _mm256_set1_epi16(1);
    pk_block_sums_batch(xsum, X, t0, T, nb, 8);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (4 * PK * 4);
    const void *d = pk_d(W, (size_t)(row / PK) * nb * PK, f16);
        __m256 acc[PK_TILE];
        ML_UNROLL(4) for (int t = 0; t < PK_TILE; t++) acc[t] = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++, p += 4 * PK * 4) {
            __m256i s16[PK_TILE];
            ML_UNROLL(4) for (int t = 0; t < PK_TILE; t++) s16[t] = _mm256_setzero_si256();
            for (int v = 0; v < 4; v++) {
                const __m256i w = _mm256_loadu_si256((const __m256i *)(p + v * PK * 4));
                const __m256i lo = _mm256_and_si256(w, m4), hi = _mm256_and_si256(_mm256_srli_epi16(w, 4), m4);
                ML_UNROLL(4) for (int t = 0; t < T; t++) {
                    s16[t] = _mm256_add_epi16(s16[t], _mm256_maddubs_epi16(lo, _mm256_set1_epi32(((const int32_t *)(X->q + (size_t)(t0 + t) * X->k + b * QK))[v])));
                    s16[t] = _mm256_add_epi16(s16[t], _mm256_maddubs_epi16(hi, _mm256_set1_epi32(((const int32_t *)(X->q + (size_t)(t0 + t) * X->k + b * QK))[v + 4])));
                }
            }
            const __m256 dv = pk_d8(d, b * PK, f16);
            ML_UNROLL(4) for (int t = 0; t < T; t++) {
                const __m256i s = _mm256_sub_epi32(_mm256_madd_epi16(s16[t], ones), _mm256_set1_epi32(xsum[t][b]));
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(dv, _mm256_set1_ps(X->d[(size_t)(t0 + t) * nb + b])), acc[t]);
            }
        }
        ML_UNROLL(4) for (int t = 0; t < T; t++) {
            float o[PK];
            _mm256_storeu_ps(o, acc[t]);
            pk_store(Y + (size_t)(t0 + t) * ldy, o, row, PK, r1);
        }
    }
}

/* the scale type as a constant per instance (pk_d8 / pk_d16) */
#define matmul_q4_tile_h(Y, ldy, W, X, r0, r1, t0, T) matmul_q4_tile(Y, ldy, W, X, r0, r1, t0, T, 1)
#define matmul_q4_tile_f(Y, ldy, W, X, r0, r1, t0, T) matmul_q4_tile(Y, ldy, W, X, r0, r1, t0, T, 0)

void ml_k_matmul_q4_pk_avx2(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    if (W->pd16) { PK_TILES(matmul_q4_tile_h, PK_TILE) }
    else { PK_TILES(matmul_q4_tile_f, PK_TILE) }
}

/* F16 KV cache (F16C): 8 halves widen to floats per load. */
#define LDH8(p) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(p)))

void ml_k_dot_f16_avx2(float *s, const float *q, const uint16_t *k, size_t ldk, int rows, int n, float scale) {
    int t = 0;
    for (; t + 4 <= rows; t += 4) {
        const uint16_t *k0 = k + (size_t)t * ldk, *k1 = k0 + ldk, *k2 = k1 + ldk, *k3 = k2 + ldk;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            const __m256 qv = _mm256_loadu_ps(q + i);
            a0 = _mm256_fmadd_ps(qv, LDH8(k0 + i), a0);
            a1 = _mm256_fmadd_ps(qv, LDH8(k1 + i), a1);
            a2 = _mm256_fmadd_ps(qv, LDH8(k2 + i), a2);
            a3 = _mm256_fmadd_ps(qv, LDH8(k3 + i), a3);
        }
        float r0 = hsum256_ps(a0), r1 = hsum256_ps(a1), r2 = hsum256_ps(a2), r3 = hsum256_ps(a3);
        for (; i < n; i++) {
            r0 += q[i] * ml_f16_to_f32(k0[i]); r1 += q[i] * ml_f16_to_f32(k1[i]);
            r2 += q[i] * ml_f16_to_f32(k2[i]); r3 += q[i] * ml_f16_to_f32(k3[i]);
        }
        s[t] = r0 * scale; s[t + 1] = r1 * scale; s[t + 2] = r2 * scale; s[t + 3] = r3 * scale;
    }
    for (; t < rows; t++) {
        const uint16_t *k0 = k + (size_t)t * ldk;
        __m256 a0 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), LDH8(k0 + i), a0);
        float r0 = hsum256_ps(a0);
        for (; i < n; i++) r0 += q[i] * ml_f16_to_f32(k0[i]);
        s[t] = r0 * scale;
    }
}

void ml_k_axpy_f16_avx2(float *y, const float *a, const uint16_t *x, size_t ldx, int rows, int n) {
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 y0 = _mm256_loadu_ps(y + i), y1 = _mm256_loadu_ps(y + i + 8), y2 = _mm256_loadu_ps(y + i + 16), y3 = _mm256_loadu_ps(y + i + 24);
        const uint16_t *xt = x + i;
        for (int t = 0; t < rows; t++, xt += ldx) {
            const __m256 av = _mm256_set1_ps(a[t]);
            y0 = _mm256_fmadd_ps(av, LDH8(xt), y0);
            y1 = _mm256_fmadd_ps(av, LDH8(xt + 8), y1);
            y2 = _mm256_fmadd_ps(av, LDH8(xt + 16), y2);
            y3 = _mm256_fmadd_ps(av, LDH8(xt + 24), y3);
        }
        _mm256_storeu_ps(y + i, y0); _mm256_storeu_ps(y + i + 8, y1); _mm256_storeu_ps(y + i + 16, y2); _mm256_storeu_ps(y + i + 24, y3);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 y0 = _mm256_loadu_ps(y + i);
        const uint16_t *xt = x + i;
        for (int t = 0; t < rows; t++, xt += ldx) y0 = _mm256_fmadd_ps(_mm256_set1_ps(a[t]), LDH8(xt), y0);
        _mm256_storeu_ps(y + i, y0);
    }
    for (; i < n; i++) {
        float v = y[i];
        for (int t = 0; t < rows; t++) v += a[t] * ml_f16_to_f32(x[(size_t)t * ldx + i]);
        y[i] = v;
    }
}
