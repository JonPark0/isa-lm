#include <math.h>
#include <string.h>
#include "kernels/avx512vnni/k_avx512vnni.h"
#include "kernels/x86_avx.h"
#include "kernels/x86_pack.h"

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
void ml_k_matvec_q4_avx512vnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    float xs[PK_MAX_BLOCKS];
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

/* Packed kernels: one 32-bit lane per row of a 16-row group. Each step is one
 * 64-byte load (4 bytes x 16 rows) and one vpdpbusd against a 4-byte
 * activation broadcast, so a block costs 8 (Q8) or 4 x 2 (Q4) dot
 * instructions for 16 rows, and its scales are applied as one vector op.
 * Weights are unsigned (Q8 stored + 128, Q4 raw nibbles); the matching
 * offset * sum(x) is subtracted per block in int32, exactly. */
#define PK 16

static inline void store_group(float *y, __m512 acc, int row, int r1) {
    if (row + PK <= r1) { _mm512_storeu_ps(y + row, acc); return; }
    _mm512_mask_storeu_ps(y + row, (__mmask16)((1u << (r1 - row)) - 1), acc);
}

ML_INLINE void matvec_q8_pk_impl(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1, const int f16) {
    const int nb = W->cols / QK;
    int xsum[PK_MAX_BLOCKS];
    pk_block_sums(xsum, x, nb, 128);
    for (int row = r0; row < r1; row += PK) {
        const int g = row / PK;
        const uint8_t *p = W->pq + (size_t)g * nb * (8 * PK * 4);
        const void *d = pk_d(W, (size_t)g * nb * PK, f16);
        __m512 acc = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++, p += 8 * PK * 4) {
            ml_prefetch_ahead(p, 8 * PK * 4);
            const int32_t *xb = (const int32_t *)(x->q + b * QK);
            __m512i s = _mm512_setzero_si512();
            for (int k = 0; k < 8; k++)
                s = _mm512_dpbusd_epi32(s, _mm512_loadu_si512((const void *)(p + k * PK * 4)), _mm512_set1_epi32(xb[k]));
            s = _mm512_sub_epi32(s, _mm512_set1_epi32(xsum[b]));
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_mul_ps(pk_d16(d, b * PK, f16), _mm512_set1_ps(x->d[b])), acc);
        }
        store_group(y, acc, row, r1);
    }
}

/* f16 is a constant in each instance, so the scale load is chosen at compile time. */
void ml_k_matvec_q8_pk_avx512vnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    if (W->pd16) matvec_q8_pk_impl(y, W, x, r0, r1, 1);
    else matvec_q8_pk_impl(y, W, x, r0, r1, 0);
}

ML_INLINE void matvec_q4_pk_impl(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1, const int f16) {
    const int nb = W->cols / QK;
    int xsum[PK_MAX_BLOCKS];
    pk_block_sums(xsum, x, nb, 8);
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    for (int row = r0; row < r1; row += PK) {
        const int g = row / PK;
        const uint8_t *p = W->pq + (size_t)g * nb * (4 * PK * 4);
        const void *d = pk_d(W, (size_t)g * nb * PK, f16);
        __m512 acc = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++, p += 4 * PK * 4) {
            ml_prefetch_ahead(p, 4 * PK * 4);
            const int32_t *xb = (const int32_t *)(x->q + b * QK);
            __m512i s = _mm512_setzero_si512();
            for (int v = 0; v < 4; v++) {
                const __m512i w = _mm512_loadu_si512((const void *)(p + v * PK * 4));
                s = _mm512_dpbusd_epi32(s, _mm512_and_si512(w, m4), _mm512_set1_epi32(xb[v]));
                s = _mm512_dpbusd_epi32(s, _mm512_and_si512(_mm512_srli_epi16(w, 4), m4), _mm512_set1_epi32(xb[v + 4]));
            }
            s = _mm512_sub_epi32(s, _mm512_set1_epi32(xsum[b]));
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_mul_ps(pk_d16(d, b * PK, f16), _mm512_set1_ps(x->d[b])), acc);
        }
        store_group(y, acc, row, r1);
    }
}

/* f16 is a constant in each instance, so the scale load is chosen at compile time. */
void ml_k_matvec_q4_pk_avx512vnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    if (W->pd16) matvec_q4_pk_impl(y, W, x, r0, r1, 1);
    else matvec_q4_pk_impl(y, W, x, r0, r1, 0);
}

/* Batched packed kernels: each weight vector loaded once feeds up to 4
 * tokens, which turns the memory-bound matvec into a compute-bound matmul.
 * Per token the arithmetic is exactly the matvec's, so results match it
 * bit for bit. */
#define TILE 4

static void block_sums_batch(int (*xsum)[PK_MAX_BLOCKS], const ml_qbatch *X, int t0, int T, int nb, int offset) {
    for (int t = 0; t < T; t++) {
        const ml_qact xt = { X->q + (size_t)(t0 + t) * X->k, X->d + (size_t)(t0 + t) * nb, X->k };
        pk_block_sums(xsum[t], &xt, nb, offset);
    }
}

static inline void store_rows(float *Y, int ldy, int t0, int T, const __m512 *acc, int row, int r1) {
    for (int t = 0; t < T; t++) store_group(Y + (size_t)(t0 + t) * ldy, acc[t], row, r1);
}

/* NG row groups (1 or 2) of a tile: each activation broadcast feeds NG dot
 * products, and the NG x T accumulator chains hide vpdpbusd's latency. */
ML_INLINE void matmul_q8_groups(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int row, int r1, int t0,
                                const int T, const int NG, int (*xsum)[PK_MAX_BLOCKS], const int f16) {
    const int nb = W->cols / QK;
    const size_t gq = (size_t)nb * (8 * PK * 4), gd = (size_t)nb * PK;
    const uint8_t *p = W->pq + (size_t)(row / PK) * gq;
    const void *d = pk_d(W, (size_t)(row / PK) * gd, f16);
    __m512 acc[2][TILE];
    ML_UNROLL(2) for (int g = 0; g < 2; g++) ML_UNROLL(4) for (int t = 0; t < TILE; t++) acc[g][t] = _mm512_setzero_ps();
    for (int b = 0; b < nb; b++, p += 8 * PK * 4) {
        __m512i s[2][TILE];
        ML_UNROLL(2) for (int g = 0; g < 2; g++) ML_UNROLL(4) for (int t = 0; t < TILE; t++) s[g][t] = _mm512_setzero_si512();
        /* unrolled: in a loop GCC copies each vpdpbusd accumulator to and from
         * another register, two moves per product (no zmm move elimination) */
        ML_UNROLL(8) for (int k = 0; k < 8; k++) {
            __m512i w[2];
            ML_UNROLL(2) for (int g = 0; g < NG; g++) w[g] = _mm512_loadu_si512((const void *)(p + g * gq + k * PK * 4));
            ML_UNROLL(4) for (int t = 0; t < T; t++) {
                const __m512i xv = _mm512_set1_epi32(((const int32_t *)(X->q + (size_t)(t0 + t) * X->k + b * QK))[k]);
                ML_UNROLL(2) for (int g = 0; g < NG; g++) s[g][t] = _mm512_dpbusd_epi32(s[g][t], w[g], xv);
            }
        }
        ML_UNROLL(2) for (int g = 0; g < NG; g++) {
            const __m512 dv = pk_d16(d, g * gd + b * PK, f16);
            ML_UNROLL(4) for (int t = 0; t < T; t++) {
                const __m512i st = _mm512_sub_epi32(s[g][t], _mm512_set1_epi32(xsum[t][b]));
                acc[g][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(st),
                                            _mm512_mul_ps(dv, _mm512_set1_ps(X->d[(size_t)(t0 + t) * nb + b])), acc[g][t]);
            }
        }
    }
    ML_UNROLL(2) for (int g = 0; g < NG; g++) store_rows(Y, ldy, t0, T, acc[g], row + g * PK, r1);
}

ML_INLINE void matmul_q8_tile(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1, int t0, const int T, const int f16) {
    static ML_THREAD_LOCAL int xsum[TILE][PK_MAX_BLOCKS];
    block_sums_batch(xsum, X, t0, T, W->cols / QK, 128);
    for (int row = r0; row < r1; row += 2 * PK) {
        if (r1 - row > PK) matmul_q8_groups(Y, ldy, W, X, row, r1, t0, T, 2, xsum, f16);
        else matmul_q8_groups(Y, ldy, W, X, row, r1, t0, T, 1, xsum, f16);
    }
}

/* the scale type as a constant per instance (pk_d8 / pk_d16) */
#define matmul_q8_tile_h(Y, ldy, W, X, r0, r1, t0, T) matmul_q8_tile(Y, ldy, W, X, r0, r1, t0, T, 1)
#define matmul_q8_tile_f(Y, ldy, W, X, r0, r1, t0, T) matmul_q8_tile(Y, ldy, W, X, r0, r1, t0, T, 0)

void ml_k_matmul_q8_pk_avx512vnni(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    if (W->pd16) { PK_TILES(matmul_q8_tile_h, TILE) }
    else { PK_TILES(matmul_q8_tile_f, TILE) }
}

ML_INLINE void matmul_q4_groups(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int row, int r1, int t0,
                                const int T, const int NG, int (*xsum)[PK_MAX_BLOCKS], const int f16) {
    const int nb = W->cols / QK;
    const size_t gq = (size_t)nb * (4 * PK * 4), gd = (size_t)nb * PK;
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const uint8_t *p = W->pq + (size_t)(row / PK) * gq;
    const void *d = pk_d(W, (size_t)(row / PK) * gd, f16);
    __m512 acc[2][TILE];
    ML_UNROLL(2) for (int g = 0; g < 2; g++) ML_UNROLL(4) for (int t = 0; t < TILE; t++) acc[g][t] = _mm512_setzero_ps();
    for (int b = 0; b < nb; b++, p += 4 * PK * 4) {
        __m512i s[2][TILE];
        ML_UNROLL(2) for (int g = 0; g < 2; g++) ML_UNROLL(4) for (int t = 0; t < TILE; t++) s[g][t] = _mm512_setzero_si512();
        ML_UNROLL(4) for (int v = 0; v < 4; v++) {
            __m512i lo[2], hi[2];
            ML_UNROLL(2) for (int g = 0; g < NG; g++) {
                const __m512i w = _mm512_loadu_si512((const void *)(p + g * gq + v * PK * 4));
                lo[g] = _mm512_and_si512(w, m4);
                hi[g] = _mm512_and_si512(_mm512_srli_epi16(w, 4), m4);
            }
            ML_UNROLL(4) for (int t = 0; t < T; t++) {
                const int32_t *xb = (const int32_t *)(X->q + (size_t)(t0 + t) * X->k + b * QK);
                const __m512i xl = _mm512_set1_epi32(xb[v]), xh = _mm512_set1_epi32(xb[v + 4]);
                ML_UNROLL(2) for (int g = 0; g < NG; g++) {
                    s[g][t] = _mm512_dpbusd_epi32(s[g][t], lo[g], xl);
                    s[g][t] = _mm512_dpbusd_epi32(s[g][t], hi[g], xh);
                }
            }
        }
        ML_UNROLL(2) for (int g = 0; g < NG; g++) {
            const __m512 dv = pk_d16(d, g * gd + b * PK, f16);
            ML_UNROLL(4) for (int t = 0; t < T; t++) {
                const __m512i st = _mm512_sub_epi32(s[g][t], _mm512_set1_epi32(xsum[t][b]));
                acc[g][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(st),
                                            _mm512_mul_ps(dv, _mm512_set1_ps(X->d[(size_t)(t0 + t) * nb + b])), acc[g][t]);
            }
        }
    }
    ML_UNROLL(2) for (int g = 0; g < NG; g++) store_rows(Y, ldy, t0, T, acc[g], row + g * PK, r1);
}

ML_INLINE void matmul_q4_tile(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1, int t0, const int T, const int f16) {
    static ML_THREAD_LOCAL int xsum[TILE][PK_MAX_BLOCKS];
    block_sums_batch(xsum, X, t0, T, W->cols / QK, 8);
    for (int row = r0; row < r1; row += 2 * PK) {
        if (r1 - row > PK) matmul_q4_groups(Y, ldy, W, X, row, r1, t0, T, 2, xsum, f16);
        else matmul_q4_groups(Y, ldy, W, X, row, r1, t0, T, 1, xsum, f16);
    }
}

/* the scale type as a constant per instance (pk_d8 / pk_d16) */
#define matmul_q4_tile_h(Y, ldy, W, X, r0, r1, t0, T) matmul_q4_tile(Y, ldy, W, X, r0, r1, t0, T, 1)
#define matmul_q4_tile_f(Y, ldy, W, X, r0, r1, t0, T) matmul_q4_tile(Y, ldy, W, X, r0, r1, t0, T, 0)

void ml_k_matmul_q4_pk_avx512vnni(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    if (W->pd16) { PK_TILES(matmul_q4_tile_h, TILE) }
    else { PK_TILES(matmul_q4_tile_f, TILE) }
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

/* Scores of 4 rows per pass: each row has its own accumulator, so the
 * arithmetic per row is the same in the 4-row and the 1-row loop. */
void ml_k_dot_f32_avx512vnni(float *s, const float *q, const float *k, size_t ldk, int rows, int n, float scale) {
    int t = 0;
    for (; t + 4 <= rows; t += 4) {
        const float *k0 = k + (size_t)t * ldk, *k1 = k0 + ldk, *k2 = k1 + ldk, *k3 = k2 + ldk;
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            const __m512 qv = _mm512_loadu_ps(q + i);
            a0 = _mm512_fmadd_ps(qv, _mm512_loadu_ps(k0 + i), a0);
            a1 = _mm512_fmadd_ps(qv, _mm512_loadu_ps(k1 + i), a1);
            a2 = _mm512_fmadd_ps(qv, _mm512_loadu_ps(k2 + i), a2);
            a3 = _mm512_fmadd_ps(qv, _mm512_loadu_ps(k3 + i), a3);
        }
        float r0 = _mm512_reduce_add_ps(a0), r1 = _mm512_reduce_add_ps(a1), r2 = _mm512_reduce_add_ps(a2), r3 = _mm512_reduce_add_ps(a3);
        for (; i < n; i++) { r0 += q[i] * k0[i]; r1 += q[i] * k1[i]; r2 += q[i] * k2[i]; r3 += q[i] * k3[i]; }
        s[t] = r0 * scale; s[t + 1] = r1 * scale; s[t + 2] = r2 * scale; s[t + 3] = r3 * scale;
    }
    for (; t < rows; t++) {
        const float *k0 = k + (size_t)t * ldk;
        __m512 a0 = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= n; i += 16) a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q + i), _mm512_loadu_ps(k0 + i), a0);
        float r0 = _mm512_reduce_add_ps(a0);
        for (; i < n; i++) r0 += q[i] * k0[i];
        s[t] = r0 * scale;
    }
}

/* y += sum_t a[t] * x_t, 64 outputs at a time held in registers across all
 * rows. Each output adds its terms in row order, whatever the chunk. */
void ml_k_axpy_f32_avx512vnni(float *y, const float *a, const float *x, size_t ldx, int rows, int n) {
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512 y0 = _mm512_loadu_ps(y + i), y1 = _mm512_loadu_ps(y + i + 16), y2 = _mm512_loadu_ps(y + i + 32), y3 = _mm512_loadu_ps(y + i + 48);
        const float *xt = x + i;
        for (int t = 0; t < rows; t++, xt += ldx) {
            const __m512 av = _mm512_set1_ps(a[t]);
            y0 = _mm512_fmadd_ps(av, _mm512_loadu_ps(xt), y0);
            y1 = _mm512_fmadd_ps(av, _mm512_loadu_ps(xt + 16), y1);
            y2 = _mm512_fmadd_ps(av, _mm512_loadu_ps(xt + 32), y2);
            y3 = _mm512_fmadd_ps(av, _mm512_loadu_ps(xt + 48), y3);
        }
        _mm512_storeu_ps(y + i, y0); _mm512_storeu_ps(y + i + 16, y1); _mm512_storeu_ps(y + i + 32, y2); _mm512_storeu_ps(y + i + 48, y3);
    }
    for (; i + 16 <= n; i += 16) {
        __m512 y0 = _mm512_loadu_ps(y + i);
        const float *xt = x + i;
        for (int t = 0; t < rows; t++, xt += ldx) y0 = _mm512_fmadd_ps(_mm512_set1_ps(a[t]), _mm512_loadu_ps(xt), y0);
        _mm512_storeu_ps(y + i, y0);
    }
    for (; i < n; i++) {
        float v = y[i];
        for (int t = 0; t < rows; t++) v += a[t] * x[(size_t)t * ldx + i];
        y[i] = v;
    }
}
