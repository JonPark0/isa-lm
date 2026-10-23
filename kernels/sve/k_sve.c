#include <arm_sve.h>
#include <math.h>
#include <string.h>
#include "kernels/sve/k_sve.h"

#define MAX_BLOCKS 2048   /* cols <= 65536, enforced by the loader */

/* Each vector covers svcntb() weights; int32 lane j of svdot holds 4 products
 * of block (off/4 + j) / 8, so lanes are scaled by a gather of per-block
 * scales and reduced once per row. */
void ml_k_matvec_q8_sve(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int cols = W->cols, nb = cols / QK;
    float sc[MAX_BLOCKS];
    for (int r = r0; r < r1; r++) {
        const int8_t *wq = W->q + (size_t)r * cols;
        const float *wd = W->d + (size_t)r * nb;
        for (int b = 0; b < nb; b++) sc[b] = wd[b] * x->d[b];
        svfloat32_t acc = svdup_f32(0.0f);
        for (int off = 0; off < cols; off += (int)svcntb()) {
            const svbool_t pb = svwhilelt_b8(off, cols);
            const svbool_t pw = svwhilelt_b32(off / 4, cols / 4);
            const svint32_t s = svdot_s32(svdup_s32(0), svld1_s8(pb, wq + off), svld1_s8(pb, x->q + off));
            const svuint32_t blk = svlsr_n_u32_x(pw, svindex_u32((uint32_t)off / 4, 1), 3);
            const svfloat32_t scl = svld1_gather_u32index_f32(pw, sc, blk);
            acc = svmla_f32_m(pw, acc, svcvt_f32_s32_x(pw, s), scl);
        }
        y[r] = svaddv_f32(svptrue_b32(), acc);
    }
}

/* Q4: one block per step (16 packed bytes always fit the 128-bit minimum VL);
 * the block's integer sum is reduced exactly, then scaled. */
void ml_k_matvec_q4_sve(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    const int nb = W->cols / QK;
    const svbool_t p16 = svwhilelt_b8(0, 16);
    for (int r = r0; r < r1; r++) {
        const uint8_t *wq = (const uint8_t *)W->q + (size_t)r * nb * (QK / 2);
        const float *wd = W->d + (size_t)r * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            const svuint8_t p = svld1_u8(p16, wq + b * (QK / 2));
            const svint8_t lo = svsub_n_s8_x(p16, svreinterpret_s8_u8(svand_n_u8_x(p16, p, 0x0F)), 8);
            const svint8_t hi = svsub_n_s8_x(p16, svreinterpret_s8_u8(svlsr_n_u8_x(p16, p, 4)), 8);
            /* inactive x bytes load as zero, so inactive weight lanes add nothing */
            svint32_t s = svdot_s32(svdup_s32(0), lo, svld1_s8(p16, x->q + b * QK));
            s = svdot_s32(s, hi, svld1_s8(p16, x->q + b * QK + 16));
            sum += (float)svaddv_s32(svptrue_b32(), s) * wd[b] * x->d[b];
        }
        y[r] = sum;
    }
}

void ml_k_quantize_q8_sve(ml_qact *out, const float *x, int k) {
    const int vl = (int)svcntw();
    for (int b = 0; b < k / QK; b++) {
        const float *xb = x + b * QK;
        float amax = 0.0f;
        for (int j = 0; j < QK; j += vl) {
            const svbool_t pg = svwhilelt_b32(j, QK);
            amax = fmaxf(amax, svmaxv_f32(pg, svabs_f32_x(pg, svld1_f32(pg, xb + j))));
        }
        if (amax < ML_QUANT_TINY) {
            out->d[b] = 0.0f;
            memset(out->q + b * QK, 0, QK);
            continue;
        }
        const float id = 127.0f / amax;
        out->d[b] = amax / 127.0f;
        for (int j = 0; j < QK; j += vl) {
            const svbool_t pg = svwhilelt_b32(j, QK);
            const svfloat32_t v = svld1_f32(pg, xb + j);
            /* copysign(0.5, v) = 0.5 with v's sign bit; plain mul + add matches scalar rounding */
            const svuint32_t sgn = svand_n_u32_x(pg, svreinterpret_u32_f32(v), 0x80000000u);
            const svfloat32_t h = svreinterpret_f32_u32(svorr_n_u32_x(pg, sgn, 0x3F000000u));
            const svfloat32_t t = svadd_f32_x(pg, svmul_n_f32_x(pg, v, id), h);
            svint32_t q = svcvt_s32_f32_x(pg, t);   /* rounds toward zero */
            q = svmin_n_s32_x(pg, svmax_n_s32_x(pg, q, -127), 127);
            svst1b_s32(pg, out->q + b * QK + j, q);
        }
    }
}

void ml_k_rmsnorm_sve(float *o, const float *x, const float *w, int n, float eps) {
    svfloat32_t acc = svdup_f32(0.0f);
    for (int i = 0; i < n; i += (int)svcntw()) {
        const svbool_t pg = svwhilelt_b32(i, n);
        const svfloat32_t v = svld1_f32(pg, x + i);
        acc = svmla_f32_m(pg, acc, v, v);
    }
    const float inv = 1.0f / sqrtf(svaddv_f32(svptrue_b32(), acc) / (float)n + eps);
    for (int i = 0; i < n; i += (int)svcntw()) {
        const svbool_t pg = svwhilelt_b32(i, n);
        svst1_f32(pg, o + i, svmul_f32_x(pg, svmul_n_f32_x(pg, svld1_f32(pg, x + i), inv), svld1_f32(pg, w + i)));
    }
}

void ml_k_add_sve(float *y, const float *x, int n) {
    for (int i = 0; i < n; i += (int)svcntw()) {
        const svbool_t pg = svwhilelt_b32(i, n);
        svst1_f32(pg, y + i, svadd_f32_x(pg, svld1_f32(pg, y + i), svld1_f32(pg, x + i)));
    }
}

/* Scores of 4 rows per pass, predicated over n. */
void ml_k_dot_f32_sve(float *s, const float *q, const float *k, size_t ldk, int rows, int n, float scale) {
    const svbool_t all = svptrue_b32();
    int t = 0;
    for (; t + 4 <= rows; t += 4) {
        const float *k0 = k + (size_t)t * ldk, *k1 = k0 + ldk, *k2 = k1 + ldk, *k3 = k2 + ldk;
        svfloat32_t a0 = svdup_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
        for (int i = 0; i < n; i += (int)svcntw()) {
            const svbool_t pg = svwhilelt_b32(i, n);
            const svfloat32_t qv = svld1_f32(pg, q + i);
            a0 = svmla_f32_m(pg, a0, qv, svld1_f32(pg, k0 + i));
            a1 = svmla_f32_m(pg, a1, qv, svld1_f32(pg, k1 + i));
            a2 = svmla_f32_m(pg, a2, qv, svld1_f32(pg, k2 + i));
            a3 = svmla_f32_m(pg, a3, qv, svld1_f32(pg, k3 + i));
        }
        s[t] = svaddv_f32(all, a0) * scale; s[t + 1] = svaddv_f32(all, a1) * scale;
        s[t + 2] = svaddv_f32(all, a2) * scale; s[t + 3] = svaddv_f32(all, a3) * scale;
    }
    for (; t < rows; t++) {
        const float *k0 = k + (size_t)t * ldk;
        svfloat32_t a0 = svdup_f32(0.0f);
        for (int i = 0; i < n; i += (int)svcntw()) {
            const svbool_t pg = svwhilelt_b32(i, n);
            a0 = svmla_f32_m(pg, a0, svld1_f32(pg, q + i), svld1_f32(pg, k0 + i));
        }
        s[t] = svaddv_f32(all, a0) * scale;
    }
}

/* y += sum_t a[t] * x_t, four vectors of outputs at a time held in registers
 * across all rows. */
void ml_k_axpy_f32_sve(float *y, const float *a, const float *x, size_t ldx, int rows, int n) {
    const int vl = (int)svcntw();
    for (int i = 0; i < n; i += 4 * vl) {
        const svbool_t p0 = svwhilelt_b32(i, n), p1 = svwhilelt_b32(i + vl, n);
        const svbool_t p2 = svwhilelt_b32(i + 2 * vl, n), p3 = svwhilelt_b32(i + 3 * vl, n);
        svfloat32_t y0 = svld1_f32(p0, y + i), y1 = svld1_f32(p1, y + i + vl);
        svfloat32_t y2 = svld1_f32(p2, y + i + 2 * vl), y3 = svld1_f32(p3, y + i + 3 * vl);
        const float *xt = x + i;
        for (int t = 0; t < rows; t++, xt += ldx) {
            y0 = svmla_n_f32_x(p0, y0, svld1_f32(p0, xt), a[t]);
            y1 = svmla_n_f32_x(p1, y1, svld1_f32(p1, xt + vl), a[t]);
            y2 = svmla_n_f32_x(p2, y2, svld1_f32(p2, xt + 2 * vl), a[t]);
            y3 = svmla_n_f32_x(p3, y3, svld1_f32(p3, xt + 3 * vl), a[t]);
        }
        svst1_f32(p0, y + i, y0); svst1_f32(p1, y + i + vl, y1);
        svst1_f32(p2, y + i + 2 * vl, y2); svst1_f32(p3, y + i + 3 * vl, y3);
    }
}
