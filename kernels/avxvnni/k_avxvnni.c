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
