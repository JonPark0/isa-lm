/* Every bound kernel (ml_k_*) vs its scalar version, and scalar vs a naive
 * double-precision reference written here. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "config/ml_config.h"
#include "core/quant_ref.h"
#include "tests/tt.h"

#define MAXN 8192
static uint32_t rng = 777;
static uint32_t urand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float frand(float lo, float hi) { return lo + (hi - lo) * (float)(urand() & 0xFFFFFF) / (float)0xFFFFFF; }

static void *xalloc(size_t n) { void *p = ml_aligned_alloc(n); memset(p, 0, n); return p; }
#define free ml_aligned_free

/* ---------- matvec ---------- */

typedef struct { ml_qtensor W; int8_t *q; float *d; } wbuf;

static wbuf make_w(int rows, int cols, int type, int unit_scale) {
    wbuf b;
    size_t nq = type == ML_Q4_0 ? (size_t)rows * cols / 2 : (size_t)rows * cols;
    b.q = xalloc(nq);
    b.d = xalloc(sizeof(float) * rows * cols / QK);
    for (size_t i = 0; i < nq; i++)
        b.q[i] = type == ML_Q4_0 ? (int8_t)(urand() & 0xFF) : (int8_t)((int)(urand() % 255) - 127);
    for (int i = 0; i < rows * cols / QK; i++) b.d[i] = unit_scale ? 1.0f : frand(0.001f, 0.05f);
    b.W = (ml_qtensor){ b.q, b.d, rows, cols, (uint8_t)type };
    return b;
}

static ml_qact make_x(int n, int unit_scale) {
    ml_qact a = { xalloc(n), xalloc(sizeof(float) * n / QK), n };
    for (int i = 0; i < n; i++) a.q[i] = (int8_t)((int)(urand() % 255) - 127);
    for (int i = 0; i < n / QK; i++) a.d[i] = unit_scale ? 1.0f : frand(0.001f, 0.05f);
    return a;
}

static int wval(const ml_qtensor *W, int r, int c) {
    if (W->type == ML_Q8_0) return W->q[(size_t)r * W->cols + c];
    size_t blk = ((size_t)r * W->cols + c) / QK;
    int j = c % QK;
    uint8_t byte = (uint8_t)W->q[blk * 16 + (j & 15)];
    return (j < 16 ? (byte & 0xF) : (byte >> 4)) - 8;
}

/* Condition scale of the last ref_matvec_row: sum of |w*x*dw*dx| over all
 * products. SIMD kernels reduce lanes in float, so error is bounded by this. */
static double ref_scale;

static double ref_matvec_row(const ml_qtensor *W, const ml_qact *x, int r) {
    double s = 0;
    ref_scale = 0;
    for (int b = 0; b < W->cols / QK; b++) {
        long long si = 0, sa = 0;
        for (int j = 0; j < QK; j++) {
            const long long p = (long long)wval(W, r, b * QK + j) * x->q[b * QK + j];
            si += p;
            sa += p < 0 ? -p : p;
        }
        const double sc = (double)W->d[(size_t)r * (W->cols / QK) + b] * x->d[b];
        s += (double)si * sc;
        ref_scale += (double)sa * fabs(sc);
    }
    return s;
}

typedef void (*matvec_fn)(float *, const ml_qtensor *, const ml_qact *, int, int);

static void check_matvec(int type, int unit) {
    static const int rows_l[] = { 1, 3, 4, 7, 64 }, cols_l[] = { 32, 96, 256 }, r0_l[] = { 0, 1 };
    matvec_fn bound = type == ML_Q8_0 ? ml_k_matvec_q8 : ml_k_matvec_q4;
    matvec_fn scal = type == ML_Q8_0 ? ml_k_matvec_q8_scalar : ml_k_matvec_q4_scalar;
    for (int ri = 0; ri < 5; ri++) for (int ci = 0; ci < 3; ci++) for (int zi = 0; zi < 2; zi++) {
        int rows = rows_l[ri], cols = cols_l[ci], r0 = r0_l[zi];
        if (r0 >= rows) continue;
        wbuf w = make_w(rows, cols, type, unit);
        ml_qact x = make_x(cols, unit);
        float *y1 = xalloc(sizeof(float) * rows), *y2 = xalloc(sizeof(float) * rows);
        for (int i = 0; i < rows; i++) y1[i] = y2[i] = 12345.0f;
        bound(y1, &w.W, &x, r0, rows);
        scal(y2, &w.W, &x, r0, rows);
        for (int r = 0; r < rows; r++) {
            if (r < r0) { TT_ASSERT(y1[r] == 12345.0f); TT_ASSERT(y2[r] == 12345.0f); continue; }
            double ref = ref_matvec_row(&w.W, &x, r);
            if (unit) { TT_ASSERT(y1[r] == (float)ref); TT_ASSERT(y2[r] == (float)ref); }
            else { TT_CLOSE(y1[r], y2[r], 1e-5, ref_scale); TT_CLOSE(y2[r], ref, 1e-5, ref_scale); }
        }
        free(w.q); free(w.d); free(x.q); free(x.d); free(y1); free(y2);
    }
}

static void test_matvec_q8_exact(void)  { check_matvec(ML_Q8_0, 1); }
static void test_matvec_q8_scaled(void) { check_matvec(ML_Q8_0, 0); }
static void test_matvec_q4_exact(void)  { check_matvec(ML_Q4_0, 1); }
static void test_matvec_q4_scaled(void) { check_matvec(ML_Q4_0, 0); }

static void test_quantize_q8_bitexact(void) {
    static const int ks[] = { 32, 4096 };
    for (int t = 0; t < 2; t++) {
        int k = ks[t];
        float *x = xalloc(sizeof(float) * k);
        for (int i = 0; i < k; i++) x[i] = frand(-4, 4);
        if (k > 64) for (int i = 32; i < 64; i++) x[i] = 0.0f;   /* zero block */
        if (k > 96) for (int i = 64; i < 96; i++) x[i] = (i & 1) ? 1e-37f : -3e-37f;   /* 127/amax = inf */
        ml_qact a = make_x(k, 1), b = make_x(k, 1);
        ml_k_quantize_q8(&a, x, k);
        ml_k_quantize_q8_scalar(&b, x, k);
        TT_ASSERT(memcmp(a.q, b.q, k) == 0);
        TT_ASSERT(memcmp(a.d, b.d, sizeof(float) * k / QK) == 0);
        free(x); free(a.q); free(a.d); free(b.q); free(b.d);
    }
}

/* ---------- float kernels ---------- */

static const int sizes[] = { 1, 7, 32, 288, 4097 };
static float *A, *B, *C, *O1, *O2, *W;

static void fill(float *p, int n, float lo, float hi) { for (int i = 0; i < n; i++) p[i] = frand(lo, hi); }

static void test_rmsnorm(void) {
    for (int s = 0; s < 5; s++) {
        int n = sizes[s];
        fill(A, n, -2, 2); fill(W, n, 0.5f, 1.5f);
        ml_k_rmsnorm(O1, A, W, n, 1e-5f);
        ml_k_rmsnorm_scalar(O2, A, W, n, 1e-5f);
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double)A[i] * A[i];
        double inv = 1.0 / sqrt(ss / n + 1e-5);
        for (int i = 0; i < n; i++) { TT_NEAR(O1[i], O2[i], 1e-5); TT_NEAR(O2[i], A[i] * inv * W[i], 1e-5); }
    }
}

static void test_softmax(void) {
    static const int ns[] = { 1, 5, 256 };
    for (int s = 0; s < 3; s++) {
        int n = ns[s];
        fill(A, n, -10, 10);
        memcpy(O1, A, sizeof(float) * n); memcpy(O2, A, sizeof(float) * n);
        ml_k_softmax(O1, n);
        ml_k_softmax_scalar(O2, n);
        double mx = A[0], sum = 0, s1 = 0;
        for (int i = 1; i < n; i++) if (A[i] > mx) mx = A[i];
        for (int i = 0; i < n; i++) sum += exp(A[i] - mx);
        for (int i = 0; i < n; i++) { TT_NEAR(O1[i], O2[i], 1e-5); TT_NEAR(O2[i], exp(A[i] - mx) / sum, 1e-5); s1 += O1[i]; }
        TT_NEAR(s1, 1.0, 1e-5);
    }
}

static void test_softmax_extreme(void) {
    float v[4] = { 1e4f, 9999.0f, -1e30f, 0.0f };
    ml_k_softmax(v, 4);
    double s = 0;
    for (int i = 0; i < 4; i++) { TT_ASSERT(isfinite(v[i])); s += v[i]; }
    TT_NEAR(s, 1.0, 1e-5);
    TT_ASSERT(v[2] == 0.0f);
    TT_NEAR(v[0], 1.0 / (1.0 + exp(-1.0)), 1e-5);
}

static void test_softmax_all_neg_inf(void) {
    for (int n = 1; n <= 9; n++) {
        float v[9], w[9];
        for (int i = 0; i < n; i++) v[i] = w[i] = -INFINITY;
        ml_k_softmax(v, n);
        ml_k_softmax_scalar(w, n);
        for (int i = 0; i < n; i++) { TT_NEAR(v[i], 1.0 / n, 1e-6); TT_NEAR(w[i], 1.0 / n, 1e-6); }
    }
}

static void test_silu_mul(void) {
    for (int s = 0; s < 5; s++) {
        int n = sizes[s];
        fill(A, n, -8, 8); fill(B, n, -2, 2);
        memcpy(O1, A, sizeof(float) * n); memcpy(O2, A, sizeof(float) * n);
        ml_k_silu_mul(O1, B, n);
        ml_k_silu_mul_scalar(O2, B, n);
        for (int i = 0; i < n; i++) {
            TT_NEAR(O1[i], O2[i], 1e-5);
            TT_NEAR(O2[i], A[i] / (1.0 + exp(-(double)A[i])) * B[i], 1e-5);
        }
    }
}

static void test_add(void) {
    for (int s = 0; s < 5; s++) {
        int n = sizes[s];
        fill(A, n, -2, 2); fill(B, n, -2, 2);
        memcpy(O1, A, sizeof(float) * n); memcpy(O2, A, sizeof(float) * n);
        ml_k_add(O1, B, n);
        ml_k_add_scalar(O2, B, n);
        for (int i = 0; i < n; i++) { TT_ASSERT(O1[i] == O2[i]); TT_ASSERT(O2[i] == A[i] + B[i]); }
    }
}

static void test_dot_f32(void) {
    for (int s = 0; s < 5; s++) {
        int n = sizes[s];
        fill(A, n, 0, 1); fill(B, n, 0, 1);   /* positive: no cancellation */
        double ref = 0;
        for (int i = 0; i < n; i++) ref += (double)A[i] * B[i];
        float r1 = ml_k_dot_f32(A, B, n), r2 = ml_k_dot_f32_scalar(A, B, n);
        TT_NEAR(r1, r2, 1e-5);
        TT_NEAR(r2, ref, 1e-5);
    }
}

static void test_axpy_f32(void) {
    for (int s = 0; s < 5; s++) {
        int n = sizes[s];
        fill(A, n, -2, 2); fill(B, n, -2, 2);
        memcpy(O1, A, sizeof(float) * n); memcpy(O2, A, sizeof(float) * n);
        const float a = 0.37f;
        ml_k_axpy_f32(O1, a, B, n);
        ml_k_axpy_f32_scalar(O2, a, B, n);
        for (int i = 0; i < n; i++) {
            double sc = fabs(A[i]) + fabs(a * B[i]);
            TT_CLOSE(O1[i], O2[i], 1e-5, sc);
            TT_CLOSE(O2[i], A[i] + (double)a * B[i], 1e-5, sc);
        }
    }
}

static void test_rope(void) {
    int hd = 48, nh = 6, nkv = 2, half = hd / 2;
    float cs[24], sn[24];
    for (int i = 0; i < half; i++) {
        double ang = 5.0 * pow(10000.0, -2.0 * i / hd);
        cs[i] = (float)cos(ang); sn[i] = (float)sin(ang);
    }
    float q1[288], q2[288], k1[96], k2[96];
    fill(A, hd * nh, -1, 1); fill(B, hd * nkv, -1, 1);
    memcpy(q1, A, sizeof q1); memcpy(q2, A, sizeof q2);
    memcpy(k1, B, sizeof k1); memcpy(k2, B, sizeof k2);
    ml_k_rope(q1, k1, cs, sn, hd, nh, nkv);
    ml_k_rope_scalar(q2, k2, cs, sn, hd, nh, nkv);
    for (int h = 0; h < nh; h++) for (int i = 0; i < half; i++) {
        int o = h * hd + 2 * i;
        double a = A[o], b = A[o + 1];
        double sc = fabs(a) + fabs(b);
        TT_CLOSE(q1[o], q2[o], 1e-5, sc); TT_CLOSE(q1[o + 1], q2[o + 1], 1e-5, sc);
        TT_CLOSE(q2[o], a * cs[i] - b * sn[i], 1e-5, sc);
        TT_CLOSE(q2[o + 1], a * sn[i] + b * cs[i], 1e-5, sc);
    }
    for (int h = 0; h < nkv; h++) for (int i = 0; i < half; i++) {
        int o = h * hd + 2 * i;
        double a = B[o], b = B[o + 1];
        double sc = fabs(a) + fabs(b);
        TT_CLOSE(k1[o], k2[o], 1e-5, sc); TT_CLOSE(k1[o + 1], k2[o + 1], 1e-5, sc);
        TT_CLOSE(k2[o], a * cs[i] - b * sn[i], 1e-5, sc);
        TT_CLOSE(k2[o + 1], a * sn[i] + b * cs[i], 1e-5, sc);
    }
}

#define SHOW(op) printf("%-12s -> %s\n", #op, ML_STR(ml_k_##op))

int TT_ENTRY(int argc, char **argv) {
    (void)argc; (void)argv;
    A = xalloc(4 * MAXN); B = xalloc(4 * MAXN); C = xalloc(4 * MAXN);
    O1 = xalloc(4 * MAXN); O2 = xalloc(4 * MAXN); W = xalloc(4 * MAXN);
    printf("ISA: %s\n", ML_ISA_NAME);
    SHOW(matvec_q8); SHOW(matvec_q4); SHOW(quantize_q8); SHOW(rmsnorm); SHOW(softmax);
    SHOW(rope); SHOW(silu_mul); SHOW(add); SHOW(dot_f32); SHOW(axpy_f32);
    TT_RUN(test_matvec_q8_exact);
    TT_RUN(test_matvec_q8_scaled);
    TT_RUN(test_matvec_q4_exact);
    TT_RUN(test_matvec_q4_scaled);
    TT_RUN(test_quantize_q8_bitexact);
    TT_RUN(test_rmsnorm);
    TT_RUN(test_softmax);
    TT_RUN(test_softmax_extreme);
    TT_RUN(test_softmax_all_neg_inf);
    TT_RUN(test_silu_mul);
    TT_RUN(test_add);
    TT_RUN(test_dot_f32);
    TT_RUN(test_axpy_f32);
    TT_RUN(test_rope);
    TT_MAIN_END("test_kernels");
}
