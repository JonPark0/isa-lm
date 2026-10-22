#include <math.h>
#include "core/quant_ref.h"

void ml_dequant_row(float *out, const ml_qtensor *W, int row) {
    const int nb = W->cols / QK;
    const float *d = W->d + (size_t)row * nb;
    if (W->type == ML_Q8_0) {
        const int8_t *q = W->q + (size_t)row * W->cols;
        for (int i = 0; i < W->cols; i++) out[i] = q[i] * d[i / QK];
        return;
    }
    const uint8_t *q = (const uint8_t *)W->q + (size_t)row * nb * (QK / 2);
    for (int b = 0; b < nb; b++)
        for (int j = 0; j < QK / 2; j++) {
            const uint8_t v = q[b * (QK / 2) + j];
            out[b * QK + j] = ((v & 0xF) - 8) * d[b];
            out[b * QK + j + QK / 2] = ((v >> 4) - 8) * d[b];
        }
}

void ml_quantize_row_q4(int8_t *q, float *d, const float *x, int k) {
    for (int b = 0; b < k / QK; b++) {
        const float *xb = x + b * QK;
        float amax = 0.0f;
        for (int j = 0; j < QK; j++) amax = fmaxf(amax, fabsf(xb[j]));
        const float id = amax != 0.0f ? 7.0f / amax : 0.0f;
        d[b] = amax / 7.0f;
        uint8_t nib[QK];
        for (int j = 0; j < QK; j++) {
            float v = truncf(xb[j] * id + copysignf(0.5f, xb[j]));
            v = fminf(fmaxf(v, -8.0f), 7.0f);
            nib[j] = (uint8_t)((int)v + 8);
        }
        for (int j = 0; j < QK / 2; j++)
            q[b * (QK / 2) + j] = (int8_t)(nib[j] | (nib[j + QK / 2] << 4));
    }
}
