/* Every bound kernel (ml_k_*) vs its scalar version, and scalar vs a naive
 * double-precision reference written here. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "config/ml_config.h"
#include "core/pack.h"
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
    b.W = (ml_qtensor){ b.q, b.d, rows, cols, (uint8_t)type, NULL, NULL };
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

/* Packed (row-interleaved) kernels, when the build has them: pack W with the
 * bound layout, run ml_k_matvec_*_pk, compare to the scalar row-major result. */
static void check_matvec_pk(int type, int unit) {
#if ML_PACK_G
    static const int rows_l[] = { 1, 3, ML_PACK_G, ML_PACK_G + 1, 3 * ML_PACK_G + 5 }, cols_l[] = { 32, 96, 256, 8288 };
    for (int ri = 0; ri < 5; ri++) for (int ci = 0; ci < 4; ci++) for (int r0 = 0; r0 <= ML_PACK_G; r0 += ML_PACK_G) {
        const int rows = rows_l[ri], cols = cols_l[ci];
        if (r0 >= rows) continue;
        wbuf w = make_w(rows, cols, type, unit);
        ml_qtensor P = w.W;
        uint8_t *pq = xalloc(ml_pack_quant_bytes(&w.W, ML_PACK_G));
        float *pd = xalloc(ml_pack_scale_bytes(&w.W, ML_PACK_G));
        ml_pack(&w.W, pq, pd, ML_PACK_G, ML_PACK_Q8_OFFSET);
        P.pq = pq; P.pd = pd;
        ml_qact x = make_x(cols, unit);
        float *y1 = xalloc(sizeof(float) * rows), *y2 = xalloc(sizeof(float) * rows);
        for (int i = 0; i < rows; i++) y1[i] = 12345.0f;
        if (type == ML_Q8_0) { ml_k_matvec_q8_pk(y1, &P, &x, r0, rows); ml_k_matvec_q8_scalar(y2, &w.W, &x, 0, rows); }
        else { ml_k_matvec_q4_pk(y1, &P, &x, r0, rows); ml_k_matvec_q4_scalar(y2, &w.W, &x, 0, rows); }
        for (int r = 0; r < rows; r++) {
            if (r < r0) { TT_ASSERT(y1[r] == 12345.0f); continue; }
            ref_matvec_row(&w.W, &x, r);
            if (unit) TT_ASSERT(y1[r] == y2[r]);
            else TT_CLOSE(y1[r], y2[r], 1e-5, ref_scale);
        }
        free(w.q); free(w.d); free(pq); free(pd); free(x.q); free(x.d); free(y1); free(y2);
    }
#else
    (void)type; (void)unit;
#endif
}

/* Batched (prefill) kernels: every token's column must equal the packed
 * matvec result bit for bit, for token counts around the kernel's tile.
 * 8288 columns cross the 8192-column chunk of the sse2 kernels. */
static void check_matmul_pk_cols(int type, int unit, int cols) {
#if ML_PACK_G
    static const int ns[] = { 1, 3, 4, 5, 9, 17 };
    const int rows = 3 * ML_PACK_G + 5, r0 = ML_PACK_G;
    wbuf w = make_w(rows, cols, type, unit);
    ml_qtensor P = w.W;
    uint8_t *pq = xalloc(ml_pack_quant_bytes(&w.W, ML_PACK_G));
    float *pd = xalloc(ml_pack_scale_bytes(&w.W, ML_PACK_G));
    ml_pack(&w.W, pq, pd, ML_PACK_G, ML_PACK_Q8_OFFSET);
    P.pq = pq; P.pd = pd;
    for (int ni = 0; ni < 6; ni++) {
        const int n = ns[ni], ldy = rows + 3;
        ml_qbatch X = { xalloc(n * cols), xalloc(sizeof(float) * n * cols / QK), n, cols };
        for (int t = 0; t < n; t++) {
            ml_qact xt = make_x(cols, unit);
            memcpy(X.q + t * cols, xt.q, cols);
            memcpy(X.d + t * (cols / QK), xt.d, sizeof(float) * cols / QK);
            free(xt.q); free(xt.d);
        }
        float *Y = xalloc(sizeof(float) * n * ldy), *yv = xalloc(sizeof(float) * rows);
        for (int i = 0; i < n * ldy; i++) Y[i] = 12345.0f;
        if (type == ML_Q8_0) ml_k_matmul_q8_pk(Y, ldy, &P, &X, r0, rows);
        else ml_k_matmul_q4_pk(Y, ldy, &P, &X, r0, rows);
        for (int t = 0; t < n; t++) {
            ml_qact xt = { X.q + t * cols, X.d + t * (cols / QK), cols };
            if (type == ML_Q8_0) ml_k_matvec_q8_pk(yv, &P, &xt, r0, rows);
            else ml_k_matvec_q4_pk(yv, &P, &xt, r0, rows);
            for (int r = 0; r < ldy; r++) {
                if (r < r0 || r >= rows) TT_ASSERT(Y[t * ldy + r] == 12345.0f);
                else TT_ASSERT(Y[t * ldy + r] == yv[r]);
            }
        }
        free(X.q); free(X.d); free(Y); free(yv);
    }
    free(w.q); free(w.d); free(pq); free(pd);
#else
    (void)type; (void)unit; (void)cols;
#endif
}

static void check_matmul_pk(int type, int unit) {
    check_matmul_pk_cols(type, unit, 96);
    check_matmul_pk_cols(type, unit, 8288);
}

static void test_matmul_q8_pk(void) { check_matmul_pk(ML_Q8_0, 0); }
static void test_matmul_q4_pk(void) { check_matmul_pk(ML_Q4_0, 0); }

static void test_matvec_q8_pk_exact(void)  { check_matvec_pk(ML_Q8_0, 1); }
static void test_matvec_q8_pk_scaled(void) { check_matvec_pk(ML_Q8_0, 0); }
static void test_matvec_q4_pk_exact(void)  { check_matvec_pk(ML_Q4_0, 1); }
static void test_matvec_q4_pk_scaled(void) { check_matvec_pk(ML_Q4_0, 0); }

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

/* Attention scores: rows of K with a row stride, against one query. */
static void test_dot_f32(void) {
    static const int rows_l[] = { 1, 3, 4, 5, 9 };
    for (int s = 0; s < 5; s++) for (int ri = 0; ri < 5; ri++) {
        const int n = sizes[s] < 300 ? sizes[s] : 300, rows = rows_l[ri];
        const size_t ld = (size_t)n + 3;
        float *K = xalloc(sizeof(float) * ld * rows), r1[9], r2[9];
        fill(A, n, 0, 1); fill(K, (int)(ld * rows), 0, 1);   /* positive: no cancellation */
        ml_k_dot_f32(r1, A, K, ld, rows, n, 0.125f);
        ml_k_dot_f32_scalar(r2, A, K, ld, rows, n, 0.125f);
        for (int t = 0; t < rows; t++) {
            double ref = 0;
            for (int i = 0; i < n; i++) ref += (double)A[i] * K[t * ld + i];
            TT_NEAR(r1[t], r2[t], 1e-5);
            TT_NEAR(r2[t], ref * 0.125, 1e-5);
        }
        free(K);
    }
}

/* Weighted sum over V rows: y += sum_t a[t] * x_t. */
static void test_axpy_f32(void) {
    static const int rows_l[] = { 1, 3, 4, 5, 9 };
    for (int s = 0; s < 5; s++) for (int ri = 0; ri < 5; ri++) {
        const int n = sizes[s], rows = rows_l[ri];
        const size_t ld = (size_t)n + 3;
        float *X = xalloc(sizeof(float) * ld * rows), a[9];
        fill(A, n, -2, 2); fill(X, (int)(ld * rows), -2, 2); fill(a, rows, 0, 1);
        memcpy(O1, A, sizeof(float) * n); memcpy(O2, A, sizeof(float) * n);
        ml_k_axpy_f32(O1, a, X, ld, rows, n);
        ml_k_axpy_f32_scalar(O2, a, X, ld, rows, n);
        for (int i = 0; i < n; i++) {
            double ref = A[i], sc = fabs(A[i]);
            for (int t = 0; t < rows; t++) { ref += (double)a[t] * X[t * ld + i]; sc += fabs(a[t] * X[t * ld + i]); }
            TT_CLOSE(O1[i], O2[i], 1e-5, sc);
            TT_CLOSE(O2[i], ref, 1e-5, sc);
        }
        free(X);
    }
}

static void fill_q8(int8_t *q, size_t n) {
    for (size_t i = 0; i < n; i++) q[i] = (int8_t)(rand() % 255 - 127);
}

/* Q8 KV cache scores: integer dot products, one scale per row (strided). */
static void test_dot_q8(void) {
    static const int rows_l[] = { 1, 3, 4, 5, 9 };
    for (int s = 0; s < 5; s++) for (int ri = 0; ri < 5; ri++) {
        const int n = sizes[s] < 300 ? sizes[s] : 300, rows = rows_l[ri];
        const size_t ld = (size_t)n + 5, ldd = 3;
        int8_t *K = xalloc(ld * rows), *q = xalloc((size_t)n);
        float dk[27], r1[9], r2[9];
        fill_q8(K, ld * rows); fill_q8(q, (size_t)n); fill(dk, 27, 0.001f, 0.05f);
        ml_k_dot_q8(r1, q, 0.02f, K, dk, ld, ldd, rows, n, 0.125f);
        ml_k_dot_q8_scalar(r2, q, 0.02f, K, dk, ld, ldd, rows, n, 0.125f);
        for (int t = 0; t < rows; t++) {
            long long ref = 0;
            for (int i = 0; i < n; i++) ref += q[i] * K[t * ld + i];
            TT_ASSERT(r1[t] == r2[t]);   /* exact integer sum, same float ops */
            TT_NEAR(r2[t], (double)ref * 0.02 * 0.125 * dk[t * ldd], 1e-5);
        }
        free(K); free(q);
    }
}

/* Q8 KV cache mix: y += sum_t (a[t] * dx_t) * x_t. */
static void test_axpy_q8(void) {
    static const int rows_l[] = { 1, 3, 4, 5, 9 };
    for (int s = 0; s < 5; s++) for (int ri = 0; ri < 5; ri++) {
        const int n = sizes[s], rows = rows_l[ri];
        const size_t ld = (size_t)n + 5, ldd = 2;
        int8_t *X = xalloc(ld * rows);
        float a[9], dx[18];
        fill_q8(X, ld * rows); fill(A, n, -2, 2); fill(a, rows, 0, 1); fill(dx, 18, 0.001f, 0.05f);
        memcpy(O1, A, sizeof(float) * n); memcpy(O2, A, sizeof(float) * n);
        ml_k_axpy_q8(O1, a, X, dx, ld, ldd, rows, n);
        ml_k_axpy_q8_scalar(O2, a, X, dx, ld, ldd, rows, n);
        for (int i = 0; i < n; i++) {
            double ref = A[i], sc = fabs(A[i]);
            for (int t = 0; t < rows; t++) {
                const double term = (double)a[t] * dx[t * ldd] * X[t * ld + i];
                ref += term; sc += fabs(term);
            }
            TT_CLOSE(O1[i], O2[i], 1e-5, sc);
            TT_CLOSE(O2[i], ref, 1e-5, sc);
        }
        free(X);
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
    SHOW(rope); SHOW(silu_mul); SHOW(add); SHOW(dot_f32); SHOW(axpy_f32); SHOW(dot_q8); SHOW(axpy_q8);
#if ML_PACK_G
    SHOW(matvec_q8_pk); SHOW(matvec_q4_pk); SHOW(matmul_q8_pk); SHOW(matmul_q4_pk);
    printf("pack: %d rows, q8 offset %d\n", ML_PACK_G, ML_PACK_Q8_OFFSET);
#else
    printf("pack: none\n");
#endif
    TT_RUN(test_matvec_q8_exact);
    TT_RUN(test_matvec_q8_scaled);
    TT_RUN(test_matvec_q4_exact);
    TT_RUN(test_matvec_q4_scaled);
    TT_RUN(test_matvec_q8_pk_exact);
    TT_RUN(test_matvec_q8_pk_scaled);
    TT_RUN(test_matvec_q4_pk_exact);
    TT_RUN(test_matvec_q4_pk_scaled);
    TT_RUN(test_matmul_q8_pk);
    TT_RUN(test_matmul_q4_pk);
    TT_RUN(test_quantize_q8_bitexact);
    TT_RUN(test_rmsnorm);
    TT_RUN(test_softmax);
    TT_RUN(test_softmax_extreme);
    TT_RUN(test_softmax_all_neg_inf);
    TT_RUN(test_silu_mul);
    TT_RUN(test_add);
    TT_RUN(test_dot_f32);
    TT_RUN(test_axpy_f32);
    TT_RUN(test_dot_q8);
    TT_RUN(test_axpy_q8);
    TT_RUN(test_rope);
    TT_MAIN_END("test_kernels");
}
