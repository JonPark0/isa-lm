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
 * sdot of 16 weight bytes against a 4-byte activation broadcast. Q4 nibbles
 * (0..15) are valid signed bytes, so they go into sdot directly and
 * 8 * sum(x) is subtracted per block. */
#define PK 4
#define PK_MAX_BLOCKS 2048   /* cols <= 65536, enforced by the loader */

static inline int32x4_t bcast4(const int8_t *p) {
    int32_t v;
    __builtin_memcpy(&v, p, 4);
    return vdupq_n_s32(v);
}

static inline void pk_store(float *y, float32x4_t acc, int row, int r1) {
    if (row + PK <= r1) { vst1q_f32(y + row, acc); return; }
    float t[PK];
    vst1q_f32(t, acc);
    for (int i = 0; row + i < r1; i++) y[row + i] = t[i];
}

void ml_k_matvec_q8_pk_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    for (int row = r0; row < r1; row += PK) {
        const int8_t *p = (const int8_t *)W->pq + (size_t)(row / PK) * nb * (8 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++, p += 8 * PK * 4) {
            const int8_t *xb = x->q + b * QK;
            int32x4_t s = vdupq_n_s32(0);
            for (int k = 0; k < 8; k++)
                s = vdotq_s32(s, vld1q_s8(p + k * PK * 4), vreinterpretq_s8_s32(bcast4(xb + 4 * k)));
            acc = vfmaq_f32(acc, vcvtq_f32_s32(s), vmulq_n_f32(vld1q_f32(d + b * PK), x->d[b]));
        }
        pk_store(y, acc, row, r1);
    }
}

void ml_k_matvec_q4_pk_neon_dotprod(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    int xsum[PK_MAX_BLOCKS];
    for (int b = 0; b < nb; b++) {
        int s = 0;
        for (int j = 0; j < QK; j++) s += x->q[b * QK + j];
        xsum[b] = 8 * s;
    }
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int row = r0; row < r1; row += PK) {
        const uint8_t *p = W->pq + (size_t)(row / PK) * nb * (4 * PK * 4);
        const float *d = W->pd + (size_t)(row / PK) * nb * PK;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++, p += 4 * PK * 4) {
            const int8_t *xb = x->q + b * QK;
            int32x4_t s = vdupq_n_s32(0);
            for (int v = 0; v < 4; v++) {
                const uint8x16_t w = vld1q_u8(p + v * PK * 4);
                s = vdotq_s32(s, vreinterpretq_s8_u8(vandq_u8(w, m4)), vreinterpretq_s8_s32(bcast4(xb + 4 * v)));
                s = vdotq_s32(s, vreinterpretq_s8_u8(vshrq_n_u8(w, 4)), vreinterpretq_s8_s32(bcast4(xb + 16 + 4 * v)));
            }
            s = vsubq_s32(s, vdupq_n_s32(xsum[b]));
            acc = vfmaq_f32(acc, vcvtq_f32_s32(s), vmulq_n_f32(vld1q_f32(d + b * PK), x->d[b]));
        }
        pk_store(y, acc, row, r1);
    }
}
