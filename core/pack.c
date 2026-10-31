#include "core/pack.h"

static int ngroups(const ml_qtensor *W, int G) { return (W->rows + G - 1) / G; }

size_t ml_pack_quant_bytes(const ml_qtensor *W, int G) {
    const size_t per_row = W->type == ML_Q4_0 ? (size_t)W->cols / 2 : (size_t)W->cols;
    return (size_t)ngroups(W, G) * G * per_row;
}

int ml_pack_f16(const ml_qtensor *W, int d16) { return d16 && W->dh; }

size_t ml_pack_scale_bytes(const ml_qtensor *W, int G, int d16) {
    return (size_t)ngroups(W, G) * G * (W->cols / QK) * (ml_pack_f16(W, d16) ? sizeof(uint16_t) : sizeof(float));
}

void ml_pack_groups(const ml_qtensor *W, uint8_t *pq, void *pd, int G, int q8_offset, int d16, int g0, int g1) {
    const int nb = W->cols / QK;
    const int q4 = W->type == ML_Q4_0;
    const int steps = q4 ? 4 : 8;                  /* 4-byte steps per block */
    const size_t block_bytes = (size_t)steps * G * 4;
    const int f16 = ml_pack_f16(W, d16);
    for (int g = g0; g < g1; g++)
        for (int b = 0; b < nb; b++) {
            uint8_t *dst = pq + ((size_t)g * nb + b) * block_bytes;
            const size_t ds = ((size_t)g * nb + b) * G;
            for (int r = 0; r < G; r++) {
                const int row = g * G + r;
                const int live = row < W->rows;
                if (f16) ((uint16_t *)pd)[ds + r] = live ? W->dh[(size_t)row * nb + b] : 0;
                else ((float *)pd)[ds + r] = live ? W->d[(size_t)row * nb + b] : 0.0f;
                for (int k = 0; k < steps; k++)
                    for (int j = 0; j < 4; j++) {
                        uint8_t v;
                        if (q4) v = live ? (uint8_t)W->q[((size_t)row * nb + b) * (QK / 2) + 4 * k + j] : 0x88;
                        else v = (uint8_t)((live ? W->q[(size_t)row * W->cols + b * QK + 4 * k + j] : 0) + q8_offset);
                        dst[(size_t)k * G * 4 + r * 4 + j] = v;
                    }
            }
        }
}

void ml_pack(const ml_qtensor *W, uint8_t *pq, void *pd, int G, int q8_offset, int d16) {
    ml_pack_groups(W, pq, pd, G, q8_offset, d16, 0, ngroups(W, G));
}
