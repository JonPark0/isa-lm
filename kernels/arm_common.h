/* NEON helpers shared by the ARM kernel files; each has an ARMv7 fallback for
 * the AArch64-only instructions it would otherwise use. */
#ifndef ML_ARM_COMMON_H
#define ML_ARM_COMMON_H
#include <arm_neon.h>
#include "core/ml.h"

static inline float hsum_f32x4(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(s, s), 0);
#endif
}

static inline float hmax_f32x4(float32x4_t v) {
#if defined(__aarch64__)
    return vmaxvq_f32(v);
#else
    float32x2_t m = vmax_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpmax_f32(m, m), 0);
#endif
}

/* a + b * c: fused where the ISA has it. */
static inline float32x4_t mla_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
#if defined(__aarch64__) || defined(__ARM_FEATURE_FMA)
    return vfmaq_f32(a, b, c);
#else
    return vmlaq_f32(a, b, c);
#endif
}

static inline float32x4_t div_f32(float32x4_t a, float32x4_t b) {
#if defined(__aarch64__)
    return vdivq_f32(a, b);
#else
    float32x4_t r = vrecpeq_f32(b);   /* estimate, refined by two Newton steps */
    r = vmulq_f32(r, vrecpsq_f32(b, r));
    r = vmulq_f32(r, vrecpsq_f32(b, r));
    return vmulq_f32(a, r);
#endif
}

static inline float32x4_t floor_f32(float32x4_t x) {
#if defined(__aarch64__)
    return vrndmq_f32(x);
#else
    float32x4_t t = vcvtq_f32_s32(vcvtq_s32_f32(x));
    return vsubq_f32(t, vreinterpretq_f32_u32(vandq_u32(vcgtq_f32(t, x),
                                                        vreinterpretq_u32_f32(vdupq_n_f32(1.0f)))));
#endif
}

/* exp(x): Cephes expf polynomial with range reduction, inputs clamped to
 * [-88, 88]. */
static inline float32x4_t exp_f32x4(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-88.3762626647949f)), vdupq_n_f32(88.3762626647949f));
    const float32x4_t t = floor_f32(mla_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.44269504088896341f)));
    x = vsubq_f32(x, vmulq_f32(t, vdupq_n_f32(0.693359375f)));
    x = vsubq_f32(x, vmulq_f32(t, vdupq_n_f32(-2.12194440e-4f)));
    const float32x4_t z = vmulq_f32(x, x);
    float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
    y = mla_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
    y = mla_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
    y = mla_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
    y = mla_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
    y = mla_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
    y = vaddq_f32(mla_f32(x, y, z), vdupq_n_f32(1.0f));
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(t), vdupq_n_s32(0x7f)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(e));
}

/* Q4 block (16 bytes) -> int8 elements 0..15 (lo) and 16..31 (hi). */
static inline void unpack_q4_neon(const uint8_t *p, int8x16_t *lo, int8x16_t *hi) {
    const uint8x16_t w = vld1q_u8(p);
    const int8x16_t eight = vdupq_n_s8(8);
    *lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w, vdupq_n_u8(0x0F))), eight);
    *hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w, 4)), eight);
}

/* xsum[b] = offset * sum of the 32 activation quants of block b. */
static inline void block_sums(int *xsum, const int8_t *xq, int nb, int offset) {
    for (int b = 0; b < nb; b++) {
        const int8x16_t x0 = vld1q_s8(xq + b * QK), x1 = vld1q_s8(xq + b * QK + 16);
#if defined(__aarch64__)
        xsum[b] = offset * (vaddlvq_s8(x0) + vaddlvq_s8(x1));
#else
        const int32x4_t s4 = vpaddlq_s16(vaddq_s16(vpaddlq_s8(x0), vpaddlq_s8(x1)));
        const int64x2_t s2 = vpaddlq_s32(s4);
        xsum[b] = offset * (int)(vgetq_lane_s64(s2, 0) + vgetq_lane_s64(s2, 1));
#endif
    }
}

/* Rows of W per pass of a batched kernel over its tokens: about PK_ROW_BYTES of
 * packed weights, so that the pass stays in L2 while every tile of tokens
 * reads it (without it, each tile streams all of r0..r1 from memory again). */
#ifndef PK_ROW_BYTES
#define PK_ROW_BYTES (512 * 1024)
#endif
static inline int pk_row_block(const ml_qtensor *W, int group) {
    const size_t row = (W->type == ML_Q8_0 ? (size_t)W->cols : (size_t)W->cols / 2) + W->cols / QK * sizeof(float);
    const int rows = (int)(PK_ROW_BYTES / row) / group * group;
    return rows > group ? rows : group;
}

/* Row loop shared by the NEON matvecs: 4 rows per pass reuse one activation
 * block. LOADW(row, b, w0, w1) loads elements 0..15 / 16..31; DOT(w0, w1, x0,
 * x1) -> int32x4 partial sums. */
#define MATVEC_NEON_BODY(LOADW, DOT)                                              \
    const int nb = W->cols / QK;                                                  \
    for (int r = r0; r < r1; r += 4) {                                            \
        const int nr = r1 - r < 4 ? r1 - r : 4;                                   \
        float32x4_t acc[4];                                                       \
        for (int i = 0; i < 4; i++) acc[i] = vdupq_n_f32(0.0f);                   \
        for (int b = 0; b < nb; b++) {                                            \
            const int8x16_t x0 = vld1q_s8(x->q + b * QK), x1 = vld1q_s8(x->q + b * QK + 16); \
            for (int i = 0; i < nr; i++) {                                        \
                int8x16_t w0, w1;                                                 \
                LOADW(r + i, b, w0, w1);                                          \
                const int32x4_t s = DOT(w0, w1, x0, x1);                          \
                const float sc = W->d[(size_t)(r + i) * nb + b] * x->d[b];        \
                acc[i] = mla_f32(acc[i], vcvtq_f32_s32(s), vdupq_n_f32(sc));      \
            }                                                                     \
        }                                                                         \
        for (int i = 0; i < nr; i++) y[r + i] = hsum_f32x4(acc[i]);               \
    }

#define LOADW_Q8_NEON(row, b, w0, w1) do {                                        \
        const int8_t *p = W->q + (size_t)(row) * W->cols + (b) * QK;             \
        w0 = vld1q_s8(p); w1 = vld1q_s8(p + 16); } while (0)

#define LOADW_Q4_NEON(row, b, w0, w1)                                             \
    unpack_q4_neon((const uint8_t *)W->q + ((size_t)(row) * nb + (b)) * (QK / 2), &w0, &w1)

#endif
