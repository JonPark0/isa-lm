#include "kernels/arm_common.h"
#include "kernels/neon_i8mm/k_neon_i8mm.h"

/* Batched packed kernels on smmla, which multiplies a 2x8 int8 matrix by the
 * transpose of another and adds the 2x2 int32 result: twice the work of an
 * sdot per instruction. The operands come from the NEON dotprod layout
 * (4-row groups, 4-byte steps): zipping the 32-bit lanes of steps 2j and
 * 2j + 1 gives rows 0-1 and rows 2-3 as 8 consecutive bytes each. Activations
 * are paired by token the same way. The block sums are the same integers the
 * dotprod matvec computes, and are scaled with the same float operations, so
 * every token's result equals the matvec's bit for bit. */
#define PK 4
#define PK_TILE 8
#define PK_MAX_BLOCKS 2048   /* cols <= 65536, enforced by the loader */

static inline void pk_store(float *y, float32x4_t acc, int row, int r1) {
    if (row + PK <= r1) { vst1q_f32(y + row, acc); return; }
    float t[PK];
    vst1q_f32(t, acc);
    for (int i = 0; row + i < r1; i++) y[row + i] = t[i];
}

static void xsums(int *xsum, const int8_t *xq, int nb) {
    for (int b = 0; b < nb; b++) {
        int sum = 0;
        for (int j = 0; j < QK; j++) sum += xq[b * QK + j];
        xsum[b] = 8 * sum;
    }
}

/* Steps 2j and 2j + 1 of a group block -> rows 0-1 and rows 2-3, 8 bytes each. */
static inline void zip_steps(int8x16_t *w01, int8x16_t *w23, int8x16_t a, int8x16_t b) {
    *w01 = vreinterpretq_s8_s32(vzip1q_s32(vreinterpretq_s32_s8(a), vreinterpretq_s32_s8(b)));
    *w23 = vreinterpretq_s8_s32(vzip2q_s32(vreinterpretq_s32_s8(a), vreinterpretq_s32_s8(b)));
}

/* Block sums of the group's 4 rows for tokens a and b (one block each). */
static inline void mm_block(int32x4_t *sa, int32x4_t *sb, const int8x16_t w01[4], const int8x16_t w23[4],
                            const int8_t *xa, const int8_t *xb) {
    const int64x2_t a0 = vreinterpretq_s64_s8(vld1q_s8(xa)), a1 = vreinterpretq_s64_s8(vld1q_s8(xa + 16));
    const int64x2_t b0 = vreinterpretq_s64_s8(vld1q_s8(xb)), b1 = vreinterpretq_s64_s8(vld1q_s8(xb + 16));
    const int8x16_t xx[4] = {
        vreinterpretq_s8_s64(vzip1q_s64(a0, b0)), vreinterpretq_s8_s64(vzip2q_s64(a0, b0)),
        vreinterpretq_s8_s64(vzip1q_s64(a1, b1)), vreinterpretq_s8_s64(vzip2q_s64(a1, b1)),
    };
    int32x4_t m01 = vdupq_n_s32(0), m23 = vdupq_n_s32(0);
    for (int j = 0; j < 4; j++) {
        m01 = vmmlaq_s32(m01, w01[j], xx[j]);   /* [r0a, r0b, r1a, r1b] */
        m23 = vmmlaq_s32(m23, w23[j], xx[j]);   /* [r2a, r2b, r3a, r3b] */
    }
    *sa = vuzp1q_s32(m01, m23);
    *sb = vuzp2q_s32(m01, m23);
}

static inline void q8_operands(int8x16_t w01[4], int8x16_t w23[4], const int8_t *p) {
    for (int j = 0; j < 4; j++) zip_steps(&w01[j], &w23[j], vld1q_s8(p + 32 * j), vld1q_s8(p + 32 * j + 16));
}

/* Q4: vector v holds step v in its low nibbles and step v + 4 in its high ones;
 * nibbles (0..15) are valid signed bytes, and 8 * sum(x) is subtracted later. */
static inline void q4_operands(int8x16_t w01[4], int8x16_t w23[4], const uint8_t *p) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int j = 0; j < 2; j++) {
        const uint8x16_t a = vld1q_u8(p + 32 * j), b = vld1q_u8(p + 32 * j + 16);
        zip_steps(&w01[j], &w23[j], vreinterpretq_s8_u8(vandq_u8(a, m4)), vreinterpretq_s8_u8(vandq_u8(b, m4)));
        zip_steps(&w01[j + 2], &w23[j + 2], vreinterpretq_s8_u8(vshrq_n_u8(a, 4)), vreinterpretq_s8_u8(vshrq_n_u8(b, 4)));
    }
}

#define PK_MATMUL(OPERANDS, PTYPE, STEPS, Q4)                                     \
    const int nb = W->cols / QK;                                                  \
    static _Thread_local int xsum[PK_TILE][PK_MAX_BLOCKS];                        \
    for (int t0 = 0; t0 < X->n; t0 += PK_TILE) {                                  \
        const int T = X->n - t0 < PK_TILE ? X->n - t0 : PK_TILE;                  \
        if (Q4) for (int t = 0; t < T; t++) xsums(xsum[t], X->q + (size_t)(t0 + t) * X->k, nb); \
        for (int row = r0; row < r1; row += PK) {                                 \
            const PTYPE *p = (const PTYPE *)W->pq + (size_t)(row / PK) * nb * (STEPS * PK * 4); \
            const float *d = W->pd + (size_t)(row / PK) * nb * PK;                \
            float32x4_t acc[PK_TILE];                                             \
            for (int t = 0; t < PK_TILE; t++) acc[t] = vdupq_n_f32(0.0f);         \
            for (int b = 0; b < nb; b++, p += STEPS * PK * 4) {                   \
                const float32x4_t dv = vld1q_f32(d + b * PK);                     \
                int8x16_t w01[4], w23[4];                                         \
                OPERANDS(w01, w23, p);                                            \
                for (int t = 0; t < T; t += 2) {                                  \
                    const int u = t + 1 < T ? t + 1 : t;   /* odd tail: pair with itself */ \
                    int32x4_t s[2];                                               \
                    mm_block(&s[0], &s[1], w01, w23, X->q + (size_t)(t0 + t) * X->k + b * QK, \
                             X->q + (size_t)(t0 + u) * X->k + b * QK);            \
                    for (int i = 0; i < 2 && t + i < T; i++) {                    \
                        const int32x4_t si = Q4 ? vsubq_s32(s[i], vdupq_n_s32(xsum[t + i][b])) : s[i]; \
                        acc[t + i] = vfmaq_f32(acc[t + i], vcvtq_f32_s32(si),     \
                                               vmulq_n_f32(dv, X->d[(size_t)(t0 + t + i) * nb + b])); \
                    }                                                             \
                }                                                                 \
            }                                                                     \
            for (int t = 0; t < T; t++) pk_store(Y + (size_t)(t0 + t) * ldy, acc[t], row, r1); \
        }                                                                         \
    }

void ml_k_matmul_q8_pk_neon_i8mm(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    PK_MATMUL(q8_operands, int8_t, 8, 0)
}

void ml_k_matmul_q4_pk_neon_i8mm(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    PK_MATMUL(q4_operands, uint8_t, 4, 1)
}
