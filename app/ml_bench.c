/* ml_bench: kernel throughput at model-sized shapes, and model tokens/s. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config/ml_config.h"
#include "core/clock.h"
#include "core/pack.h"
#include "model/pack_model.h"
#include "model/transformer.h"

static uint32_t rng = 1;
static uint32_t urand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static void *xalloc(size_t n) {
    uint8_t *p = ml_aligned_alloc(n);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)urand();
    return p;
}

/* Runs fn until at least 0.2 s have passed; returns seconds per call. */
#define TIME_IT(iters, call) do {                                                 \
        double t0 = ml_now(), dt; long n = 0;                                      \
        do { for (int it_ = 0; it_ < (iters); it_++) { call; } n += (iters);      \
             dt = ml_now() - t0; } while (dt < 0.2);                               \
        per = dt / (double)n; } while (0)

static void bench_matvec(int type, int rows, int cols, int iters) {
    const size_t nq = type == ML_Q8_0 ? (size_t)rows * cols : (size_t)rows * cols / 2;
    float *d = xalloc(sizeof(float) * rows * cols / QK);
    for (int i = 0; i < rows * cols / QK; i++) d[i] = 0.01f;
    ml_qtensor W = { xalloc(nq), d, rows, cols, (uint8_t)type, NULL, NULL, NULL };
    ml_qact x = { xalloc(cols), xalloc(sizeof(float) * cols / QK), cols };
    for (int i = 0; i < cols / QK; i++) x.d[i] = 0.01f;
    float *y = xalloc(sizeof(float) * rows);
    double per;
    if (type == ML_Q8_0) TIME_IT(iters, ml_k_matvec_q8(y, &W, &x, 0, rows));
    else TIME_IT(iters, ml_k_matvec_q4(y, &W, &x, 0, rows));
    const double bytes = (double)nq + sizeof(float) * (double)rows * cols / QK;
    printf("%-12s %5dx%-5d %12.0f %8.2f %8.2f\n", type == ML_Q8_0 ? "matvec_q8" : "matvec_q4",
           rows, cols, per * 1e9, bytes / per / 1e9, 2.0 * rows * cols / per / 1e9);
#if ML_PACK_G
    ml_qtensor P = W;
    uint8_t *pq = ml_aligned_alloc(ml_pack_quant_bytes(&W, ML_PACK_G));
    float *pd = ml_aligned_alloc(ml_pack_scale_bytes(&W, ML_PACK_G));
    ml_pack(&W, pq, pd, ML_PACK_G, ML_PACK_Q8_OFFSET, ML_PACK_Q4_XOR);
    P.pq = pq;
    P.pd = pd;
    if (type == ML_Q8_0) TIME_IT(iters, ml_k_matvec_q8_pk(y, &P, &x, 0, rows));
    else TIME_IT(iters, ml_k_matvec_q4_pk(y, &P, &x, 0, rows));
    printf("%-12s %5dx%-5d %12.0f %8.2f %8.2f\n", type == ML_Q8_0 ? "matvec_q8_pk" : "matvec_q4_pk",
           rows, cols, per * 1e9, bytes / per / 1e9, 2.0 * rows * cols / per / 1e9);
    /* batched: ML_BATCH_MAX tokens per call, as in prompt prefill */
    const int nt = ML_BATCH_MAX;
    ml_qbatch X = { xalloc((size_t)nt * cols), xalloc(sizeof(float) * nt * (cols / QK)), nt, cols };
    for (int i = 0; i < nt * (cols / QK); i++) X.d[i] = 0.01f;
    float *Y = xalloc(sizeof(float) * nt * rows);
    if (type == ML_Q8_0) TIME_IT(1, ml_k_matmul_q8_pk(Y, rows, &P, &X, 0, rows));
    else TIME_IT(1, ml_k_matmul_q4_pk(Y, rows, &P, &X, 0, rows));
    printf("%-12s %5dx%-5d %12.0f %8.2f %8.2f\n", type == ML_Q8_0 ? "matmul_q8_pk" : "matmul_q4_pk",
           rows, cols, per * 1e9, bytes / per / 1e9, 2.0 * rows * cols * nt / per / 1e9);
    ml_aligned_free(X.q); ml_aligned_free(X.d); ml_aligned_free(Y);
    ml_aligned_free(pq);
    ml_aligned_free(pd);
#endif
    ml_aligned_free((void *)W.q); ml_aligned_free(d); ml_aligned_free(x.q); ml_aligned_free(x.d); ml_aligned_free(y);
}

static void bench_quantize(int k, int iters) {
    float *v = xalloc(sizeof(float) * k);
    for (int i = 0; i < k; i++) v[i] = (float)(int)(urand() % 2001 - 1000) / 250.0f;
    ml_qact a = { xalloc(k), xalloc(sizeof(float) * k / QK), k };
    double per;
    TIME_IT(iters, ml_k_quantize_q8(&a, v, k));
    printf("%-12s %11d %12.0f %8.2f %8s\n", "quantize_q8", k, per * 1e9, 4.0 * k / per / 1e9, "-");
    ml_aligned_free(v); ml_aligned_free(a.q); ml_aligned_free(a.d);
}

int ml_main(int argc, char **argv) {
    const char *model = NULL;
    int nthreads = 1, tokens = 64, no_pack = 0, kv_type = ML_F32;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-pack")) no_pack = 1;
        else if (!strcmp(argv[i], "--kv") && i + 1 < argc && !strcmp(argv[i + 1], "q8")) { kv_type = ML_Q8_0; i++; }
        else if (!strcmp(argv[i], "--kv") && i + 1 < argc && !strcmp(argv[i + 1], "f32")) { kv_type = ML_F32; i++; }
        else if (argv[i][0] != '-') model = argv[i];
        else { fprintf(stderr, "usage: ml_bench [MODEL.mlb] [-t threads] [-n tokens] [--no-pack] [--kv f32|q8]\n"); return 2; }
    }
    printf("isa-lm %s, ISA: %s  (matvec_q8 -> %s, matvec_q4 -> %s, quantize_q8 -> %s)\n", ML_VERSION, ML_ISA_NAME,
           ML_STR(ml_k_matvec_q8), ML_STR(ml_k_matvec_q4), ML_STR(ml_k_quantize_q8));
    printf("%-12s %11s %12s %8s %8s\n", "op", "shape", "ns/iter", "GB/s", "GOPS");
    bench_matvec(ML_Q8_0, 768, 768, 20);
    bench_matvec(ML_Q8_0, 4096, 4096, 1);
    bench_matvec(ML_Q4_0, 768, 768, 20);
    bench_matvec(ML_Q4_0, 4096, 4096, 1);
    /* the FFN shapes of a 0.5B model (Qwen2.5): long rows weigh on per-block work */
    bench_matvec(ML_Q8_0, 4864, 896, 1);
    bench_matvec(ML_Q8_0, 896, 4864, 1);
    bench_matvec(ML_Q4_0, 4864, 896, 1);
    bench_matvec(ML_Q4_0, 896, 4864, 1);
    bench_quantize(4096, 100);
    if (!model) return 0;

    ml_model m;
    ml_status st = ml_model_load(&m, model);
    if (st != ML_OK) { fprintf(stderr, "error: %s: %s\n", model, ml_status_str(st)); return 1; }
    ml_state s;
    if ((st = ml_state_init(&s, &m, nthreads, kv_type)) != ML_OK || (!no_pack && (st = ml_model_pack(&m, s.pool)) != ML_OK)) {
        fprintf(stderr, "error: %s\n", ml_status_str(st));
        ml_model_free(&m);
        return 1;
    }
    if (tokens > m.cfg.max_seq / 2) tokens = m.cfg.max_seq / 2;
    /* prefill `tokens` prompt tokens, then decode `tokens` more greedily */
    int *prompt = malloc(sizeof(int) * (size_t)tokens);
    for (int i = 0; i < tokens; i++) prompt[i] = 1 + i % (m.cfg.vocab - 1);
    double t0 = ml_now();
    ml_prefill(&s, prompt, tokens, 0, NULL);
    const double tp = ml_now() - t0;
    int tok = prompt[tokens - 1];
    t0 = ml_now();
    for (int p = tokens; p < 2 * tokens; p++) {
        const float *lg = ml_forward(&s, tok, p);
        if (!lg) break;
        int best = 0;
        for (int i = 1; i < m.cfg.vocab; i++) if (lg[i] > lg[best]) best = i;
        tok = best;
    }
    const double td = ml_now() - t0;
    printf("model %s: prefill %.1f tok/s, decode %.2f tok/s (%d tokens each, threads %d, packed %d)\n", model,
           tokens / tp, tokens / td, tokens, ml_pool_size(s.pool), m.packed ? ml_pack_group() : 0);
    free(prompt);
    ml_state_free(&s);
    ml_model_free(&m);
    return 0;
}
