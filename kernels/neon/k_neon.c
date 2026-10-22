#include <math.h>
#include <string.h>
#include "kernels/arm_common.h"
#include "kernels/neon/k_neon.h"

/* 32 int8 pairs -> int32x4 via widening multiplies (|w*x| <= 16129 fits int16)
 * and pairwise accumulate. Works on ARMv7 and AArch64. */
static inline int32x4_t dot32_neon(int8x16_t w0, int8x16_t w1, int8x16_t x0, int8x16_t x1) {
    int32x4_t s = vdupq_n_s32(0);
    s = vpadalq_s16(s, vmull_s8(vget_low_s8(w0), vget_low_s8(x0)));
    s = vpadalq_s16(s, vmull_s8(vget_high_s8(w0), vget_high_s8(x0)));
    s = vpadalq_s16(s, vmull_s8(vget_low_s8(w1), vget_low_s8(x1)));
    s = vpadalq_s16(s, vmull_s8(vget_high_s8(w1), vget_high_s8(x1)));
    return s;
}

void ml_k_matvec_q8_neon(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC_NEON_BODY(LOADW_Q8_NEON, dot32_neon)
}

void ml_k_matvec_q4_neon(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC_NEON_BODY(LOADW_Q4_NEON, dot32_neon)
}

void ml_k_quantize_q8_neon(ml_qact *out, const float *x, int k) {
    const uint32x4_t sign = vdupq_n_u32(0x80000000u);
    const float32x4_t half = vdupq_n_f32(0.5f);
    for (int b = 0; b < k / QK; b++) {
        const float *xb = x + b * QK;
        float32x4_t v[8], m = vdupq_n_f32(0.0f);
        for (int j = 0; j < 8; j++) {
            v[j] = vld1q_f32(xb + 4 * j);
            m = vmaxq_f32(m, vabsq_f32(v[j]));
        }
        const float amax = hmax_f32x4(m);
        if (amax < ML_QUANT_TINY) {
            out->d[b] = 0.0f;
            memset(out->q + b * QK, 0, QK);
            continue;
        }
        const float id = 127.0f / amax;
        out->d[b] = amax / 127.0f;
        const float32x4_t vid = vdupq_n_f32(id);
        int16x8_t h[4];
        for (int j = 0; j < 4; j++) {
            int32x4_t i32[2];
            for (int u = 0; u < 2; u++) {
                const float32x4_t vv = v[2 * j + u];
                /* trunc(v*id + copysign(0.5, v)); plain mul + add matches scalar rounding */
                const float32x4_t t = vaddq_f32(vmulq_f32(vv, vid), vbslq_f32(sign, vv, half));
                i32[u] = vcvtq_s32_f32(t);
            }
            h[j] = vcombine_s16(vqmovn_s32(i32[0]), vqmovn_s32(i32[1]));
        }
        const int8x16_t lim = vdupq_n_s8(-127);
        vst1q_s8(out->q + b * QK, vmaxq_s8(vcombine_s8(vqmovn_s16(h[0]), vqmovn_s16(h[1])), lim));
        vst1q_s8(out->q + b * QK + 16, vmaxq_s8(vcombine_s8(vqmovn_s16(h[2]), vqmovn_s16(h[3])), lim));
    }
}

void ml_k_rmsnorm_neon(float *o, const float *x, const float *w, int n, float eps) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t v = vld1q_f32(x + i);
        acc = mla_f32(acc, v, v);
    }
    float ss = hsum_f32x4(acc);
    for (; i < n; i++) ss += x[i] * x[i];
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (i = 0; i + 4 <= n; i += 4)
        vst1q_f32(o + i, vmulq_f32(vmulq_n_f32(vld1q_f32(x + i), inv), vld1q_f32(w + i)));
    for (; i < n; i++) o[i] = x[i] * inv * w[i];
}

void ml_k_softmax_neon(float *x, int n) {
    int i = 0;
    float32x4_t vm = vdupq_n_f32(x[0]);
    for (; i + 4 <= n; i += 4) vm = vmaxq_f32(vm, vld1q_f32(x + i));
    float mx = hmax_f32x4(vm);
    for (; i < n; i++) mx = fmaxf(mx, x[i]);
    if (mx == -INFINITY) {   /* every entry masked: uniform */
        for (i = 0; i < n; i++) x[i] = 1.0f / (float)n;
        return;
    }
    const float32x4_t vmx = vdupq_n_f32(mx);
    float32x4_t vs = vdupq_n_f32(0.0f);
    for (i = 0; i + 4 <= n; i += 4) {
        const float32x4_t e = exp_f32x4(vsubq_f32(vld1q_f32(x + i), vmx));
        vst1q_f32(x + i, e);
        vs = vaddq_f32(vs, e);
    }
    float sum = hsum_f32x4(vs);
    for (; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    const float inv = 1.0f / sum;
    for (i = 0; i + 4 <= n; i += 4) vst1q_f32(x + i, vmulq_n_f32(vld1q_f32(x + i), inv));
    for (; i < n; i++) x[i] *= inv;
}

static void rope_heads_neon(float *v, const float *c, const float *s, int hd, int nh) {
    static const float alt_v[4] = { -1.0f, 1.0f, -1.0f, 1.0f };
    const float32x4_t alt = vld1q_f32(alt_v);
    for (int h = 0; h < nh; h++) {
        float *p = v + h * hd;
        int i = 0;   /* pair index */
        for (; 2 * i + 4 <= hd; i += 2) {
            const float32x4_t a = vld1q_f32(p + 2 * i);
            const float32x2x2_t cz = vzip_f32(vld1_f32(c + i), vld1_f32(c + i));
            const float32x2x2_t sz = vzip_f32(vld1_f32(s + i), vld1_f32(s + i));
            const float32x4_t cd = vcombine_f32(cz.val[0], cz.val[1]);
            const float32x4_t sd = vcombine_f32(sz.val[0], sz.val[1]);
            const float32x4_t sw = vrev64q_f32(a);   /* swap within pairs */
            vst1q_f32(p + 2 * i, vaddq_f32(vmulq_f32(a, cd), vmulq_f32(vmulq_f32(sw, sd), alt)));
        }
        for (; 2 * i < hd; i++) {
            const float a = p[2 * i], b = p[2 * i + 1];
            p[2 * i] = a * c[i] - b * s[i];
            p[2 * i + 1] = a * s[i] + b * c[i];
        }
    }
}

void ml_k_rope_neon(float *q, float *k, const float *cos_t, const float *sin_t,
                    int head_dim, int n_heads, int n_kv_heads) {
    rope_heads_neon(q, cos_t, sin_t, head_dim, n_heads);
    rope_heads_neon(k, cos_t, sin_t, head_dim, n_kv_heads);
}

void ml_k_silu_mul_neon(float *g, const float *u, int n) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t v = vld1q_f32(g + i);
        const float32x4_t sig = div_f32(v, vaddq_f32(one, exp_f32x4(vnegq_f32(v))));
        vst1q_f32(g + i, vmulq_f32(sig, vld1q_f32(u + i)));
    }
    for (; i < n; i++) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}

void ml_k_add_neon(float *y, const float *x, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vaddq_f32(vld1q_f32(y + i), vld1q_f32(x + i)));
    for (; i < n; i++) y[i] += x[i];
}

float ml_k_dot_f32_neon(const float *a, const float *b, int n) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 = mla_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
        a1 = mla_f32(a1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    for (; i + 4 <= n; i += 4) a0 = mla_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
    float s = hsum_f32x4(vaddq_f32(a0, a1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

void ml_k_axpy_f32_neon(float *y, float a, const float *x, int n) {
    const float32x4_t va = vdupq_n_f32(a);
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, mla_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
    for (; i < n; i++) y[i] += a * x[i];
}
