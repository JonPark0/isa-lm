#include "kernels/arm_common.h"
#include "kernels/neon_dotprod/k_neon_dotprod.h"

static inline int32x4_t dot32_sdot(int8x16_t w0, int8x16_t w1, int8x16_t x0, int8x16_t x1) {
    return vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0), w1, x1);
}

void ml_k_matvec_q8_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC_NEON_BODY(LOADW_Q8_NEON, dot32_sdot)
}

void ml_k_matvec_q4_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC_NEON_BODY(LOADW_Q4_NEON, dot32_sdot)
}

/* Packed kernels: 4 rows per group, one 32-bit lane per row; each step is one
 * sdot of 16 weight bytes (4 rows x 4 bytes) against 4 activation bytes.
 * The by-element form (vdotq_laneq_s32) takes those 4 bytes straight from a
 * lane of the block's activation vectors, so no broadcast is needed. Q4
 * nibbles (0..15) are valid signed bytes, so they go into sdot directly and
 * 8 * sum(x) is subtracted per block. */
#define PK 4
#define PK_TILE 4
#define PK_MAX_BLOCKS 2048   /* cols <= 65536, enforced by the loader */

static inline void pk_store(float *y, float32x4_t acc, int row, int r1) {
    if (row + PK <= r1) { vst1q_f32(y + row, acc); return; }
    float t[PK];
    vst1q_f32(t, acc);
    for (int i = 0; row + i < r1; i++) y[row + i] = t[i];
}

/* Integer sums of one Q8 block for the 4 rows of a group. */
static inline int32x4_t q8_block(const int8_t *p, const int8_t *xb) {
    const int8x16_t x0 = vld1q_s8(xb), x1 = vld1q_s8(xb + 16);
    int32x4_t s = vdupq_n_s32(0);
    s = vdotq_laneq_s32(s, vld1q_s8(p + 0 * 16), x0, 0);
    s = vdotq_laneq_s32(s, vld1q_s8(p + 1 * 16), x0, 1);
    s = vdotq_laneq_s32(s, vld1q_s8(p + 2 * 16), x0, 2);
    s = vdotq_laneq_s32(s, vld1q_s8(p + 3 * 16), x0, 3);
    s = vdotq_laneq_s32(s, vld1q_s8(p + 4 * 16), x1, 0);
    s = vdotq_laneq_s32(s, vld1q_s8(p + 5 * 16), x1, 1);
    s = vdotq_laneq_s32(s, vld1q_s8(p + 6 * 16), x1, 2);
    s = vdotq_laneq_s32(s, vld1q_s8(p + 7 * 16), x1, 3);
    return s;
}

/* Same for Q4: step v's low nibbles pair with x[4v..], high with x[16+4v..]. */
static inline int32x4_t q4_block(const uint8_t *p, const int8_t *xb, int xsum) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t x0 = vld1q_s8(xb), x1 = vld1q_s8(xb + 16);
    int32x4_t s = vdupq_n_s32(0);
#define Q4_STEP(v)                                                                          \
    do {                                                                                    \
        const uint8x16_t w = vld1q_u8(p + (v) * 16);                                        \
        s = vdotq_laneq_s32(s, vreinterpretq_s8_u8(vandq_u8(w, m4)), x0, v);                \
        s = vdotq_laneq_s32(s, vreinterpretq_s8_u8(vshrq_n_u8(w, 4)), x1, v);               \
    } while (0)
    Q4_STEP(0); Q4_STEP(1); Q4_STEP(2); Q4_STEP(3);
#undef Q4_STEP
    return vsubq_s32(s, vdupq_n_s32(xsum));
}

static void xsums(int *xsum, const int8_t *xq, int nb) {
    for (int b = 0; b < nb; b++) {
        int sum = 0;
        for (int j = 0; j < QK; j++) sum += xq[b * QK + j];
        xsum[b] = 8 * sum;
    }
}

void ml_k_matvec_q8_pk_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    for (int row = r0; row < r1; row += PK) {
        const int8_t *p = (const int8_t *)W->pq + (size_t)(row / PK) * nb * (8 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++, p += 8 * PK * 4)
            acc = vfmaq_f32(acc, vcvtq_f32_s32(q8_block(p, x->q + b * QK)), vmulq_n_f32(vld1q_f32(d + b * PK), x->d[b]));
        pk_store(y, acc, row, r1);
    }
}

void ml_k_matvec_q4_pk_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    int xsum[PK_MAX_BLOCKS];
    xsums(xsum, x->q, nb);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (4 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++, p += 4 * PK * 4)
            acc = vfmaq_f32(acc, vcvtq_f32_s32(q4_block(p, x->q + b * QK, xsum[b])), vmulq_n_f32(vld1q_f32(d + b * PK), x->d[b]));
        pk_store(y, acc, row, r1);
    }
}

/* Batched: the weight block stays in L1 across the tile's tokens; per token
 * the arithmetic is the matvec's. */
void ml_k_matmul_q8_pk_neon_dotprod(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    const int nb = W->cols / QK;
    for (int t0 = 0; t0 < X->n; t0 += PK_TILE) {
        const int T = X->n - t0 < PK_TILE ? X->n - t0 : PK_TILE;
        for (int row = r0; row < r1; row += PK) {
            const int8_t *p = (const int8_t *)W->pq + (size_t)(row / PK) * nb * (8 * PK * 4);
            const float *d = W->pd + (size_t)(row / PK) * nb * PK;
            float32x4_t acc[PK_TILE];
            for (int t = 0; t < PK_TILE; t++) acc[t] = vdupq_n_f32(0.0f);
            for (int b = 0; b < nb; b++, p += 8 * PK * 4) {
                const float32x4_t dv = vld1q_f32(d + b * PK);
                for (int t = 0; t < T; t++)
                    acc[t] = vfmaq_f32(acc[t], vcvtq_f32_s32(q8_block(p, X->q + (size_t)(t0 + t) * X->k + b * QK)),
                                       vmulq_n_f32(dv, X->d[(size_t)(t0 + t) * nb + b]));
            }
            for (int t = 0; t < T; t++) pk_store(Y + (size_t)(t0 + t) * ldy, acc[t], row, r1);
        }
    }
}

void ml_k_matmul_q4_pk_neon_dotprod(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    const int nb = W->cols / QK;
    static _Thread_local int xsum[PK_TILE][PK_MAX_BLOCKS];
    for (int t0 = 0; t0 < X->n; t0 += PK_TILE) {
        const int T = X->n - t0 < PK_TILE ? X->n - t0 : PK_TILE;
        for (int t = 0; t < T; t++) xsums(xsum[t], X->q + (size_t)(t0 + t) * X->k, nb);
        for (int row = r0; row < r1; row += PK) {
            const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (4 * PK * 4);
            const float *d = W->pd + (size_t)(row / PK) * nb * PK;
            float32x4_t acc[PK_TILE];
            for (int t = 0; t < PK_TILE; t++) acc[t] = vdupq_n_f32(0.0f);
            for (int b = 0; b < nb; b++, p += 4 * PK * 4) {
                const float32x4_t dv = vld1q_f32(d + b * PK);
                for (int t = 0; t < T; t++)
                    acc[t] = vfmaq_f32(acc[t], vcvtq_f32_s32(q4_block(p, X->q + (size_t)(t0 + t) * X->k + b * QK, xsum[t][b])),
                                       vmulq_n_f32(dv, X->d[(size_t)(t0 + t) * nb + b]));
            }
            for (int t = 0; t < T; t++) pk_store(Y + (size_t)(t0 + t) * ldy, acc[t], row, r1);
        }
    }
}
