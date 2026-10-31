#include <math.h>
#include "kernels/scalar/k_scalar.h"

void ml_k_matvec_q8_scalar(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    for (int r = r0; r < r1; r++) {
        const int8_t *wq = W->q + (size_t)r * W->cols;
        const float *wd = W->d + (size_t)r * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            int32_t s = 0;
            for (int j = 0; j < QK; j++) s += wq[b * QK + j] * x->q[b * QK + j];
            sum += (float)s * wd[b] * x->d[b];
        }
        y[r] = sum;
    }
}

void ml_k_matvec_q4_scalar(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    for (int r = r0; r < r1; r++) {
        const uint8_t *wq = (const uint8_t *)W->q + (size_t)r * nb * (QK / 2);
        const float *wd = W->d + (size_t)r * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = wq + b * (QK / 2);
            const int8_t *xq = x->q + b * QK;
            int32_t s = 0;
            for (int j = 0; j < QK / 2; j++) {
                s += ((p[j] & 0xF) - 8) * xq[j];
                s += ((p[j] >> 4) - 8) * xq[j + QK / 2];
            }
            sum += (float)s * wd[b] * x->d[b];
        }
        y[r] = sum;
    }
}

void ml_k_quantize_q8_scalar(ml_qact *out, const float *x, int k) {
    for (int b = 0; b < k / QK; b++) {
        const float *xb = x + b * QK;
        float amax = 0.0f;
        for (int j = 0; j < QK; j++) amax = fmaxf(amax, fabsf(xb[j]));
        if (amax < ML_QUANT_TINY) {
            out->d[b] = 0.0f;
            for (int j = 0; j < QK; j++) out->q[b * QK + j] = 0;
            continue;
        }
        const float id = 127.0f / amax;
        out->d[b] = amax / 127.0f;
        for (int j = 0; j < QK; j++) {
            float v = truncf(xb[j] * id + copysignf(0.5f, xb[j]));
            v = fminf(fmaxf(v, -127.0f), 127.0f);
            out->q[b * QK + j] = (int8_t)v;
        }
    }
}

void ml_k_rmsnorm_scalar(float *o, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * inv * w[i];
}

void ml_k_softmax_scalar(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) mx = fmaxf(mx, x[i]);
    if (mx == -INFINITY) {   /* every entry masked: uniform */
        for (int i = 0; i < n; i++) x[i] = 1.0f / (float)n;
        return;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static void rope_heads(float *v, const float *c, const float *s, int hd, int nh) {
    for (int h = 0; h < nh; h++) {
        float *p = v + h * hd;
        for (int i = 0; i < hd / 2; i++) {
            const float a = p[2 * i], b = p[2 * i + 1];
            p[2 * i] = a * c[i] - b * s[i];
            p[2 * i + 1] = a * s[i] + b * c[i];
        }
    }
}

void ml_k_rope_scalar(float *q, float *k, const float *cos_t, const float *sin_t,
                      int head_dim, int n_heads, int n_kv_heads) {
    rope_heads(q, cos_t, sin_t, head_dim, n_heads);
    rope_heads(k, cos_t, sin_t, head_dim, n_kv_heads);
}

void ml_k_silu_mul_scalar(float *g, const float *u, int n) {
    for (int i = 0; i < n; i++) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}

void ml_k_gelu_mul_scalar(float *g, const float *u, int n) {
    for (int i = 0; i < n; i++) {
        const float x = g[i];
        g[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))) * u[i];
    }
}

void ml_k_add_scalar(float *y, const float *x, int n) {
    for (int i = 0; i < n; i++) y[i] += x[i];
}

void ml_k_dot_f32_scalar(float *s, const float *q, const float *k, size_t ldk, int rows, int n, float scale) {
    for (int t = 0; t < rows; t++, k += ldk) {
        float acc = 0.0f;
        for (int i = 0; i < n; i++) acc += q[i] * k[i];
        s[t] = acc * scale;
    }
}

void ml_k_axpy_f32_scalar(float *y, const float *a, const float *x, size_t ldx, int rows, int n) {
    for (int t = 0; t < rows; t++, x += ldx)
        for (int i = 0; i < n; i++) y[i] += a[t] * x[i];
}

/* Q8 KV cache: one scale per row (dk[t * lddk]). s[t] = sum(q k_t) * (dq * scale) * dk_t. */
void ml_k_dot_q8_scalar(float *s, const int8_t *q, float dq, const int8_t *k, const float *dk, size_t ldk, size_t lddk, int rows, int n, float scale) {
    const float c = dq * scale;
    for (int t = 0; t < rows; t++) {
        const int8_t *kt = k + (size_t)t * ldk;
        int32_t sum = 0;
        for (int i = 0; i < n; i++) sum += q[i] * kt[i];
        s[t] = (float)sum * c * dk[(size_t)t * lddk];
    }
}

/* y += sum_t (a[t] * dx_t) * x_t, rows added in order. */
void ml_k_axpy_q8_scalar(float *y, const float *a, const int8_t *x, const float *dx, size_t ldx, size_t lddx, int rows, int n) {
    for (int t = 0; t < rows; t++) {
        const int8_t *xt = x + (size_t)t * ldx;
        const float w = a[t] * dx[(size_t)t * lddx];
        for (int i = 0; i < n; i++) y[i] += w * (float)xt[i];
    }
}
