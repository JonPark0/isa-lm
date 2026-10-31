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
 * bytes are packed ^ 0x88, so each nibble is q - 8 as a signed 4-bit value:
 * shifting the low nibble up (and masking the high one in place) gives
 * 16 * (q - 8) as a signed byte, and the 16 is removed by the conversion to
 * float (vcvtq_n_f32_s32, exact here), with no per-block sum to subtract. */
#define PK 4
#define PK_TILE 8

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

/* Same for Q4 (16x the sums): step v's low nibbles pair with x[4v..], high
 * with x[16+4v..]. */
static inline int32x4_t q4_block(const int8_t *p, const int8_t *xb) {
    const int8x16_t hi = vdupq_n_s8((int8_t)0xF0);
    const int8x16_t x0 = vld1q_s8(xb), x1 = vld1q_s8(xb + 16);
    int32x4_t s = vdupq_n_s32(0);
#define Q4_STEP(v)                                                                          \
    do {                                                                                    \
        const int8x16_t w = vld1q_s8(p + (v) * 16);                                         \
        s = vdotq_laneq_s32(s, vshlq_n_s8(w, 4), x0, v);                                    \
        s = vdotq_laneq_s32(s, vandq_s8(w, hi), x1, v);                                     \
    } while (0)
    Q4_STEP(0); Q4_STEP(1); Q4_STEP(2); Q4_STEP(3);
#undef Q4_STEP
    return s;
}

void ml_k_matvec_q8_pk_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    for (int row = r0; row < r1; row += PK) {
        const int8_t *p = (const int8_t *)W->pq + (size_t)(row / PK) * nb * (8 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++, p += 8 * PK * 4) {
            ml_prefetch_ahead(p, 8 * PK * 4);
            acc = vfmaq_f32(acc, vcvtq_f32_s32(q8_block(p, x->q + b * QK)), vmulq_n_f32(vld1q_f32(d + b * PK), x->d[b]));
        }
        pk_store(y, acc, row, r1);
    }
}

void ml_k_matvec_q4_pk_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    for (int row = r0; row < r1; row += PK) {
        const int8_t *p = (const int8_t *)W->pq + (size_t)(row / PK) * nb * (4 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++, p += 4 * PK * 4) {
            ml_prefetch_ahead(p, 4 * PK * 4);
            acc = vfmaq_f32(acc, vcvtq_n_f32_s32(q4_block(p, x->q + b * QK), 4), vmulq_n_f32(vld1q_f32(d + b * PK), x->d[b]));
        }
        pk_store(y, acc, row, r1);
    }
}

/* Batched: the weight block stays in L1 across the tile's tokens; per token
 * the arithmetic is the matvec's. GROUP_TILE runs one row group against NT
 * tokens; it is used with a constant NT, so the accumulators stay in
 * registers. */
#define GROUP_TILE(BLOCK, STEPS, NT)                                       \
    do {                                                                          \
        float32x4_t acc[NT];                                                      \
        _Pragma("GCC unroll 8")                                                   \
        for (int t = 0; t < (NT); t++) acc[t] = vdupq_n_f32(0.0f);                \
        const int8_t *pp = p;                                                     \
        for (int b = 0; b < nb; b++, pp += STEPS * PK * 4) {                      \
            const float32x4_t dv = vld1q_f32(d + b * PK);                         \
            _Pragma("GCC unroll 8")                                               \
            for (int t = 0; t < (NT); t++)                                        \
                acc[t] = vfmaq_f32(acc[t], BLOCK(t),                              \
                                   vmulq_n_f32(dv, X->d[(size_t)(t0 + t) * nb + b])); \
        }                                                                         \
        _Pragma("GCC unroll 8")                                                   \
        for (int t = 0; t < (NT); t++) pk_store(Y + (size_t)(t0 + t) * ldy, acc[t], row, r1); \
    } while (0)

#define PK_MATMUL(BLOCK, STEPS)                                                   \
    const int nb = W->cols / QK;                                                  \
    for (int t0 = 0; t0 < X->n; t0 += PK_TILE) {                                  \
        const int T = X->n - t0 < PK_TILE ? X->n - t0 : PK_TILE;                  \
        for (int row = r0; row < r1; row += PK) {                                 \
            const int8_t *p = (const int8_t *)W->pq + (size_t)(row / PK) * nb * (STEPS * PK * 4); \
            const float *d = W->pd + (size_t)(row / PK) * nb * PK;                \
            switch (T) {                                                          \
            case 8: GROUP_TILE(BLOCK, STEPS, 8); break;                           \
            case 7: GROUP_TILE(BLOCK, STEPS, 7); break;                           \
            case 6: GROUP_TILE(BLOCK, STEPS, 6); break;                           \
            case 5: GROUP_TILE(BLOCK, STEPS, 5); break;                           \
            case 4: GROUP_TILE(BLOCK, STEPS, 4); break;                           \
            case 3: GROUP_TILE(BLOCK, STEPS, 3); break;                           \
            case 2: GROUP_TILE(BLOCK, STEPS, 2); break;                           \
            default: GROUP_TILE(BLOCK, STEPS, 1); break;                          \
            }                                                                     \
        }                                                                         \
    }

#define Q8_TOKEN(t) vcvtq_f32_s32(q8_block(pp, X->q + (size_t)(t0 + (t)) * X->k + b * QK))
#define Q4_TOKEN(t) vcvtq_n_f32_s32(q4_block(pp, X->q + (size_t)(t0 + (t)) * X->k + b * QK), 4)

void ml_k_matmul_q8_pk_neon_dotprod(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    PK_MATMUL(Q8_TOKEN, 8)
}

void ml_k_matmul_q4_pk_neon_dotprod(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    PK_MATMUL(Q4_TOKEN, 4)
}
