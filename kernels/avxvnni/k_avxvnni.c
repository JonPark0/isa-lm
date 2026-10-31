#include "kernels/avxvnni/k_avxvnni.h"
#include "kernels/x86_avx.h"
#include "kernels/x86_pack.h"

/* vpdpbusd: u8 x s8, 4 products summed per int32 lane, no saturation. */
static inline __m256i dot32_vnni(__m256i w, __m256i x) {
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(x, x), _mm256_sign_epi8(w, x));
}

void ml_k_matvec_q8_avxvnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC256_BODY(LOADW_Q8_256, dot32_vnni)
}

void ml_k_matvec_q4_avxvnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC256_BODY(LOADW_Q4_256, dot32_vnni)
}

/* Packed kernels: 8 rows per group, one 32-bit lane per row (see the
 * avx512vnni version for the scheme). */
#define PK 8

void ml_k_matvec_q8_pk_avxvnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    int xsum[PK_MAX_BLOCKS];
    pk_block_sums(xsum, x, nb, 128);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (8 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++, p += 8 * PK * 4) {
            ml_prefetch_ahead(p, 8 * PK * 4);
            const int32_t *xb = (const int32_t *)(x->q + b * QK);
            __m256i s = _mm256_setzero_si256();
            for (int k = 0; k < 8; k++)
                s = _mm256_dpbusd_avx_epi32(s, _mm256_loadu_si256((const __m256i *)(p + k * PK * 4)), _mm256_set1_epi32(xb[k]));
            s = _mm256_sub_epi32(s, _mm256_set1_epi32(xsum[b]));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(_mm256_loadu_ps(d + b * PK), _mm256_set1_ps(x->d[b])), acc);
        }
        float t[PK];
        _mm256_storeu_ps(t, acc);
        pk_store(y, t, row, PK, r1);
    }
}

void ml_k_matvec_q4_pk_avxvnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    int xsum[PK_MAX_BLOCKS];
    pk_block_sums(xsum, x, nb, 8);
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (4 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++, p += 4 * PK * 4) {
            ml_prefetch_ahead(p, 4 * PK * 4);
            const int32_t *xb = (const int32_t *)(x->q + b * QK);
            __m256i s = _mm256_setzero_si256();
            for (int v = 0; v < 4; v++) {
                const __m256i w = _mm256_loadu_si256((const __m256i *)(p + v * PK * 4));
                s = _mm256_dpbusd_avx_epi32(s, _mm256_and_si256(w, m4), _mm256_set1_epi32(xb[v]));
                s = _mm256_dpbusd_avx_epi32(s, _mm256_and_si256(_mm256_srli_epi16(w, 4), m4), _mm256_set1_epi32(xb[v + 4]));
            }
            s = _mm256_sub_epi32(s, _mm256_set1_epi32(xsum[b]));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(_mm256_loadu_ps(d + b * PK), _mm256_set1_ps(x->d[b])), acc);
        }
        float t[PK];
        _mm256_storeu_ps(t, acc);
        pk_store(y, t, row, PK, r1);
    }
}

/* Batched packed kernels: each weight vector feeds up to PK_TILE tokens;
 * per token the arithmetic is the matvec's, so results match it exactly. */
ML_INLINE void matmul_q8_tile(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1, int t0, const int T) {
    const int nb = W->cols / QK;
    static ML_THREAD_LOCAL int xsum[PK_TILE][PK_MAX_BLOCKS];
    pk_block_sums_batch(xsum, X, t0, T, nb, 128);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (8 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        __m256 acc[PK_TILE];
        ML_UNROLL(4) for (int t = 0; t < PK_TILE; t++) acc[t] = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++, p += 8 * PK * 4) {
            __m256i s[PK_TILE];
            ML_UNROLL(4) for (int t = 0; t < PK_TILE; t++) s[t] = _mm256_setzero_si256();
            /* unrolled: in a loop GCC copies each vpdpbusd accumulator to and from
             * another register, two moves per product (no zmm move elimination) */
            ML_UNROLL(8) for (int k = 0; k < 8; k++) {
                const __m256i w = _mm256_loadu_si256((const __m256i *)(p + k * PK * 4));
                ML_UNROLL(4) for (int t = 0; t < T; t++) s[t] = _mm256_dpbusd_avx_epi32(s[t], w, _mm256_set1_epi32(((const int32_t *)(X->q + (size_t)(t0 + t) * X->k + b * QK))[k]));
            }
            const __m256 dv = _mm256_loadu_ps(d + b * PK);
            ML_UNROLL(4) for (int t = 0; t < T; t++) {
                s[t] = _mm256_sub_epi32(s[t], _mm256_set1_epi32(xsum[t][b]));
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s[t]), _mm256_mul_ps(dv, _mm256_set1_ps(X->d[(size_t)(t0 + t) * nb + b])), acc[t]);
            }
        }
        ML_UNROLL(4) for (int t = 0; t < T; t++) {
            float o[PK];
            _mm256_storeu_ps(o, acc[t]);
            pk_store(Y + (size_t)(t0 + t) * ldy, o, row, PK, r1);
        }
    }
}

void ml_k_matmul_q8_pk_avxvnni(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    PK_TILES(matmul_q8_tile, PK_TILE)
}

ML_INLINE void matmul_q4_tile(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1, int t0, const int T) {
    const int nb = W->cols / QK;
    static ML_THREAD_LOCAL int xsum[PK_TILE][PK_MAX_BLOCKS];
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    pk_block_sums_batch(xsum, X, t0, T, nb, 8);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (4 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        __m256 acc[PK_TILE];
        ML_UNROLL(4) for (int t = 0; t < PK_TILE; t++) acc[t] = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++, p += 4 * PK * 4) {
            __m256i s[PK_TILE];
            ML_UNROLL(4) for (int t = 0; t < PK_TILE; t++) s[t] = _mm256_setzero_si256();
            ML_UNROLL(4) for (int v = 0; v < 4; v++) {
                const __m256i w = _mm256_loadu_si256((const __m256i *)(p + v * PK * 4));
                const __m256i lo = _mm256_and_si256(w, m4), hi = _mm256_and_si256(_mm256_srli_epi16(w, 4), m4);
                ML_UNROLL(4) for (int t = 0; t < T; t++) {
                    s[t] = _mm256_dpbusd_avx_epi32(s[t], lo, _mm256_set1_epi32(((const int32_t *)(X->q + (size_t)(t0 + t) * X->k + b * QK))[v]));
                    s[t] = _mm256_dpbusd_avx_epi32(s[t], hi, _mm256_set1_epi32(((const int32_t *)(X->q + (size_t)(t0 + t) * X->k + b * QK))[v + 4]));
                }
            }
            const __m256 dv = _mm256_loadu_ps(d + b * PK);
            ML_UNROLL(4) for (int t = 0; t < T; t++) {
                s[t] = _mm256_sub_epi32(s[t], _mm256_set1_epi32(xsum[t][b]));
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s[t]), _mm256_mul_ps(dv, _mm256_set1_ps(X->d[(size_t)(t0 + t) * nb + b])), acc[t]);
            }
        }
        ML_UNROLL(4) for (int t = 0; t < T; t++) {
            float o[PK];
            _mm256_storeu_ps(o, acc[t]);
            pk_store(Y + (size_t)(t0 + t) * ldy, o, row, PK, r1);
        }
    }
}

void ml_k_matmul_q4_pk_avxvnni(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    PK_TILES(matmul_q4_tile, PK_TILE)
}
