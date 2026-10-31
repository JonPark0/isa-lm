#include "kernels/arm_common.h"
#include "kernels/neon_i8mm/k_neon_i8mm.h"

/* Batched packed kernels on smmla, which multiplies a 2x8 int8 matrix by the
 * transpose of another and adds the 2x2 int32 result: twice the work of an
 * sdot per instruction. The operands come from the NEON dotprod layout
 * (4-row groups, 4-byte steps): zipping the 32-bit lanes of steps 2j and
 * 2j + 1 gives rows 0-1 and rows 2-3 as 8 consecutive bytes each. Activations
 * are paired by token once per tile (make_pairs). The block sums are the same integers the
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

/* Steps 2j and 2j + 1 of a group block -> rows 0-1 and rows 2-3, 8 bytes each. */
static inline void zip_steps(int8x16_t *w01, int8x16_t *w23, int8x16_t a, int8x16_t b) {
    *w01 = vreinterpretq_s8_s32(vzip1q_s32(vreinterpretq_s32_s8(a), vreinterpretq_s32_s8(b)));
    *w23 = vreinterpretq_s8_s32(vzip2q_s32(vreinterpretq_s32_s8(a), vreinterpretq_s32_s8(b)));
}

static inline void q8_operands(int8x16_t w01[4], int8x16_t w23[4], const int8_t *p) {
    zip_steps(&w01[0], &w23[0], vld1q_s8(p), vld1q_s8(p + 16));
    zip_steps(&w01[1], &w23[1], vld1q_s8(p + 32), vld1q_s8(p + 48));
    zip_steps(&w01[2], &w23[2], vld1q_s8(p + 64), vld1q_s8(p + 80));
    zip_steps(&w01[3], &w23[3], vld1q_s8(p + 96), vld1q_s8(p + 112));
}

/* Q4: vector v holds step v in its low nibbles and step v + 4 in its high ones,
 * each q - 8 as a signed 4-bit value (packed ^ 0x88); both are taken as
 * 16 * (q - 8), and the 16 goes in the conversion to float (as in the dotprod
 * kernels). */
static inline void q4_operands(int8x16_t w01[4], int8x16_t w23[4], const int8_t *p) {
    const int8x16_t hi = vdupq_n_s8((int8_t)0xF0);
    _Pragma("GCC unroll 2")
    for (int j = 0; j < 2; j++) {
        const int8x16_t a = vld1q_s8(p + 32 * j), b = vld1q_s8(p + 32 * j + 16);
        zip_steps(&w01[j], &w23[j], vshlq_n_s8(a, 4), vshlq_n_s8(b, 4));
        zip_steps(&w01[j + 2], &w23[j + 2], vandq_s8(a, hi), vandq_s8(b, hi));
    }
}

/* The tile's activations as smmla operands, built once per tile: pair q holds
 * tokens 2q and 2q + 1 (an odd last token is paired with itself), and entry
 * 4b + j holds bytes 8j..8j+7 of block b of both. */
static ML_THREAD_LOCAL int8x16_t xpair[PK_TILE / 2][PK_MAX_BLOCKS * 4];
/* and their scales as smmla lays out its 2x2 results: (x_2q, x_2q+1) twice */
static ML_THREAD_LOCAL float32x4_t xdpair[PK_TILE / 2][PK_MAX_BLOCKS];

static void make_pairs(const ml_qbatch *X, int t0, int T, int nb) {
    for (int q = 0; 2 * q < T; q++) {
        const int ta = t0 + 2 * q, tb = t0 + (2 * q + 1 < T ? 2 * q + 1 : 2 * q);
        const int8_t *xa = X->q + (size_t)ta * X->k;
        const int8_t *xb = X->q + (size_t)tb * X->k;
        for (int b = 0; b < nb; b++) {
            const float da = X->d[(size_t)ta * nb + b], db = X->d[(size_t)tb * nb + b];
            const float xd[4] = { da, db, da, db };
            xdpair[q][b] = vld1q_f32(xd);
            const int64x2_t a0 = vreinterpretq_s64_s8(vld1q_s8(xa + b * QK)), a1 = vreinterpretq_s64_s8(vld1q_s8(xa + b * QK + 16));
            const int64x2_t b0 = vreinterpretq_s64_s8(vld1q_s8(xb + b * QK)), b1 = vreinterpretq_s64_s8(vld1q_s8(xb + b * QK + 16));
            int8x16_t *o = xpair[q] + 4 * b;
            o[0] = vreinterpretq_s8_s64(vzip1q_s64(a0, b0));
            o[1] = vreinterpretq_s8_s64(vzip2q_s64(a0, b0));
            o[2] = vreinterpretq_s8_s64(vzip1q_s64(a1, b1));
            o[3] = vreinterpretq_s8_s64(vzip2q_s64(a1, b1));
        }
    }
}

/* One row group against NP token pairs. Called with a constant NP, so the
 * accumulators stay in registers. They keep smmla's layout, rows 0-1 and rows
 * 2-3 of both tokens of a pair ([r0 t0, r0 t1, r1 t0, r1 t1]), with the
 * scales in the same layout, and are separated into tokens at the end: per
 * token and row the operations are still the matvec's. */
#define GROUP_TILE(OPERANDS, STEPS, CVT, NP)                                \
    do {                                                                          \
        float32x4_t acc[2 * (NP)];                                                \
        _Pragma("GCC unroll 8")                                                   \
        for (int i = 0; i < 2 * (NP); i++) acc[i] = vdupq_n_f32(0.0f);            \
        const int8_t *pp = p;                                                     \
        for (int b = 0; b < nb; b++, pp += STEPS * PK * 4) {                      \
            const float32x4_t dv = vld1q_f32(d + b * PK);                         \
            const float32x4_t d01 = vzip1q_f32(dv, dv), d23 = vzip2q_f32(dv, dv); \
            int8x16_t w01[4], w23[4];                                             \
            OPERANDS(w01, w23, pp);                                               \
            _Pragma("GCC unroll 4")                                               \
            for (int q = 0; q < (NP); q++) {                                      \
                const int8x16_t *xx = xpair[q] + 4 * b;                           \
                int32x4_t m01 = vmmlaq_s32(vdupq_n_s32(0), w01[0], xx[0]);        \
                int32x4_t m23 = vmmlaq_s32(vdupq_n_s32(0), w23[0], xx[0]);        \
                m01 = vmmlaq_s32(m01, w01[1], xx[1]); m23 = vmmlaq_s32(m23, w23[1], xx[1]); \
                m01 = vmmlaq_s32(m01, w01[2], xx[2]); m23 = vmmlaq_s32(m23, w23[2], xx[2]); \
                m01 = vmmlaq_s32(m01, w01[3], xx[3]); m23 = vmmlaq_s32(m23, w23[3], xx[3]); \
                const float32x4_t xd = xdpair[q][b];                              \
                acc[2 * q] = vfmaq_f32(acc[2 * q], CVT(m01), vmulq_f32(d01, xd));  \
                acc[2 * q + 1] = vfmaq_f32(acc[2 * q + 1], CVT(m23), vmulq_f32(d23, xd)); \
            }                                                                     \
        }                                                                         \
        _Pragma("GCC unroll 4")                                                   \
        for (int q = 0; q < (NP); q++) {                                          \
            const float32x4_t a0 = vuzp1q_f32(acc[2 * q], acc[2 * q + 1]);        \
            const float32x4_t a1 = vuzp2q_f32(acc[2 * q], acc[2 * q + 1]);        \
            pk_store(Y + (size_t)(t0 + 2 * q) * ldy, a0, row, r1);                \
            if (2 * q + 1 < T) pk_store(Y + (size_t)(t0 + 2 * q + 1) * ldy, a1, row, r1); \
        }                                                                         \
    } while (0)

/* int32 sums -> float (exact): Q4's are 16x the sums. */
#define CVT_Q8(s) vcvtq_f32_s32(s)
#define CVT_Q4(s) vcvtq_n_f32_s32(s, 4)

#define PK_MATMUL(OPERANDS, STEPS, CVT)                                         \
    const int nb = W->cols / QK;                                                  \
    for (int t0 = 0; t0 < X->n; t0 += PK_TILE) {                                  \
        const int T = X->n - t0 < PK_TILE ? X->n - t0 : PK_TILE;                  \
        make_pairs(X, t0, T, nb);                                                 \
        for (int row = r0; row < r1; row += PK) {                                 \
            const int8_t *p = (const int8_t *)W->pq + (size_t)(row / PK) * nb * (STEPS * PK * 4); \
            const float *d = W->pd + (size_t)(row / PK) * nb * PK;                \
            switch ((T + 1) / 2) {                                                \
            case 4: GROUP_TILE(OPERANDS, STEPS, CVT, 4); break;                 \
            case 3: GROUP_TILE(OPERANDS, STEPS, CVT, 3); break;                 \
            case 2: GROUP_TILE(OPERANDS, STEPS, CVT, 2); break;                 \
            default: GROUP_TILE(OPERANDS, STEPS, CVT, 1); break;                \
            }                                                                     \
        }                                                                         \
    }

void ml_k_matmul_q8_pk_neon_i8mm(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    PK_MATMUL(q8_operands, 8, CVT_Q8)
}

void ml_k_matmul_q4_pk_neon_i8mm(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    PK_MATMUL(q4_operands, 4, CVT_Q4)
}
