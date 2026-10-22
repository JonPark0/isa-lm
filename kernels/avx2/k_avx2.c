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

void ml_k_matvec_q8_pk_avx2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    const __m256i ones = _mm256_set1_epi16(1);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (8 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++, p += 8 * PK * 4) {
            const int32_t *xb = (const int32_t *)(x->q + b * QK);
            __m256i s = _mm256_setzero_si256();
            for (int k = 0; k < 8; k++) {
                const __m256i xv = _mm256_set1_epi32(xb[k]);
                const __m256i w = _mm256_loadu_si256((const __m256i *)(p + k * PK * 4));
                const __m256i prod = _mm256_maddubs_epi16(_mm256_sign_epi8(xv, xv), _mm256_sign_epi8(w, xv));
                s = _mm256_add_epi32(s, _mm256_madd_epi16(prod, ones));
            }
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(_mm256_loadu_ps(d + b * PK), _mm256_set1_ps(x->d[b])), acc);
        }
        float t[PK];
        _mm256_storeu_ps(t, acc);
        pk_store(y, t, row, PK, r1);
    }
}

void ml_k_matvec_q4_pk_avx2(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    int xsum[PK_MAX_BLOCKS];
    pk_block_sums(xsum, x, nb, 8);
    const __m256i m4 = _mm256_set1_epi8(0x0F), ones = _mm256_set1_epi16(1);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (4 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++, p += 4 * PK * 4) {
            const int32_t *xb = (const int32_t *)(x->q + b * QK);
            __m256i s16 = _mm256_setzero_si256();
            for (int v = 0; v < 4; v++) {
                const __m256i w = _mm256_loadu_si256((const __m256i *)(p + v * PK * 4));
                s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(_mm256_and_si256(w, m4), _mm256_set1_epi32(xb[v])));
                s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(w, 4), m4),
                                                                 _mm256_set1_epi32(xb[v + 4])));
            }
            const __m256i s = _mm256_sub_epi32(_mm256_madd_epi16(s16, ones), _mm256_set1_epi32(xsum[b]));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(_mm256_loadu_ps(d + b * PK), _mm256_set1_ps(x->d[b])), acc);
        }
        float t[PK];
        _mm256_storeu_ps(t, acc);
        pk_store(y, t, row, PK, r1);
    }
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

float ml_k_dot_f32_avx2(const float *a, const float *b, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
    }
    for (; i + 8 <= n; i += 8) a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    float s = hsum256_ps(_mm256_add_ps(a0, a1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

void ml_k_axpy_f32_avx2(float *y, float a, const float *x, int n) {
    const __m256 va = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * x[i];
}
