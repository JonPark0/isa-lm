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
    int xsum[PK_MAX_BLOCKS];
    block_sums(xsum, x->q, nb, 8);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (4 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++, p += 4 * PK * 4) {
            ml_prefetch_ahead(p, 4 * PK * 4);
            acc = vfmaq_f32(acc, vcvtq_f32_s32(q4_block(p, x->q + b * QK, xsum[b])), vmulq_n_f32(vld1q_f32(d + b * PK), x->d[b]));
        }
        pk_store(y, acc, row, r1);
    }
}

/* Batched. Rows go in L2-sized blocks (pk_row_block), each read by every tile
 * of tokens; within a tile a group's block is loaded (Q4: and unpacked) once
 * into 8 vectors and used by every token. Per token the integer sums are the
 * matvec's and are scaled the same way, so results match it bit for bit. */
static inline void q8_weights(int8x16_t w[8], const int8_t *p) {
    for (int k = 0; k < 8; k++) w[k] = vld1q_s8(p + k * 16);
}

/* Q4: vector v holds step v (pairs with x[4v..]) in its low nibbles and step
 * v + 4 (x[16+4v..]) in its high ones. */
static inline void q4_weights(int8x16_t w[8], const uint8_t *p) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int v = 0; v < 4; v++) {
        const uint8x16_t a = vld1q_u8(p + v * 16);
        w[v] = vreinterpretq_s8_u8(vandq_u8(a, m4));
        w[v + 4] = vreinterpretq_s8_u8(vshrq_n_u8(a, 4));
    }
}

/* Integer sums of a block for the group's 4 rows: w[k] against activation bytes 4k..4k+3. */
static inline int32x4_t block_dot(const int8x16_t w[8], const int8_t *xb) {
    const int8x16_t x0 = vld1q_s8(xb), x1 = vld1q_s8(xb + 16);
    int32x4_t s = vdupq_n_s32(0);
    s = vdotq_laneq_s32(s, w[0], x0, 0);
    s = vdotq_laneq_s32(s, w[1], x0, 1);
    s = vdotq_laneq_s32(s, w[2], x0, 2);
    s = vdotq_laneq_s32(s, w[3], x0, 3);
    s = vdotq_laneq_s32(s, w[4], x1, 0);
    s = vdotq_laneq_s32(s, w[5], x1, 1);
    s = vdotq_laneq_s32(s, w[6], x1, 2);
    s = vdotq_laneq_s32(s, w[7], x1, 3);
    return s;
}

/* One row group against NT tokens (a constant, so the accumulators stay in registers). */
#define GROUP_TILE(WEIGHTS, PTYPE, STEPS, Q4, NT)                                 \
    do {                                                                          \
        float32x4_t acc[NT];                                                      \
        _Pragma("GCC unroll 4")                                                   \
        for (int t = 0; t < (NT); t++) acc[t] = vdupq_n_f32(0.0f);                \
        const PTYPE *pp = p;                                                      \
        for (int b = 0; b < nb; b++, pp += STEPS * PK * 4) {                      \
            const float32x4_t dv = vld1q_f32(d + b * PK);                         \
            int8x16_t w[8];                                                       \
            WEIGHTS(w, pp);                                                       \
            _Pragma("GCC unroll 4")                                               \
            for (int t = 0; t < (NT); t++) {                                      \
                int32x4_t s = block_dot(w, X->q + (size_t)(t0 + t) * X->k + b * QK); \
                if (Q4) s = vsubq_s32(s, vdupq_n_s32(xsum[t][b]));                \
                acc[t] = vfmaq_f32(acc[t], vcvtq_f32_s32(s), vmulq_n_f32(dv, X->d[(size_t)(t0 + t) * nb + b])); \
            }                                                                     \
        }                                                                         \
        _Pragma("GCC unroll 4")                                                   \
        for (int t = 0; t < (NT); t++) pk_store(Y + (size_t)(t0 + t) * ldy, acc[t], row, r1); \
    } while (0)

#define PK_MATMUL(WEIGHTS, PTYPE, STEPS, Q4)                                      \
    const int nb = W->cols / QK;                                                  \
    static ML_THREAD_LOCAL int xsum[PK_TILE][PK_MAX_BLOCKS];                      \
    for (int rb = pk_row_block(W, PK), b0 = r0; b0 < r1; b0 += rb) {              \
        const int b1 = r1 - b0 < rb ? r1 : b0 + rb;                               \
        for (int t0 = 0; t0 < X->n; t0 += PK_TILE) {                              \
            const int T = X->n - t0 < PK_TILE ? X->n - t0 : PK_TILE;              \
            if (Q4) for (int t = 0; t < T; t++) block_sums(xsum[t], X->q + (size_t)(t0 + t) * X->k, nb, 8); \
            for (int row = b0; row < b1; row += PK) {                             \
                const PTYPE *p = (const PTYPE *)W->pq + (size_t)(row / PK) * nb * (STEPS * PK * 4); \
                const float *d = W->pd + (size_t)(row / PK) * nb * PK;            \
                switch (T) {                                                      \
                case 4: GROUP_TILE(WEIGHTS, PTYPE, STEPS, Q4, 4); break;          \
                case 3: GROUP_TILE(WEIGHTS, PTYPE, STEPS, Q4, 3); break;          \
                case 2: GROUP_TILE(WEIGHTS, PTYPE, STEPS, Q4, 2); break;          \
                default: GROUP_TILE(WEIGHTS, PTYPE, STEPS, Q4, 1); break;         \
                }                                                                 \
            }                                                                     \
        }                                                                         \
    }

void ml_k_matmul_q8_pk_neon_dotprod(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    PK_MATMUL(q8_weights, int8_t, 8, 0)
}

void ml_k_matmul_q4_pk_neon_dotprod(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    PK_MATMUL(q4_weights, uint8_t, 4, 1)
}
