/* Loader tests. argv[1] = random Q8 model, argv[2] = random Q4 model, each with
 * a sidecar <model>.emb5.f32 holding dequantized embedding row 5. */
#include <stdlib.h>
#include <string.h>
#include "core/quant_ref.h"
#include "model/mlb.h"
#include "tests/tt.h"

static const char *q8_path, *q4_path;
static char tmp_path[1024];

static unsigned char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n);
    if (fread(b, 1, *n, f) != *n) { free(b); b = NULL; }
    fclose(f);
    return b;
}

static void spit(const char *path, const unsigned char *b, size_t n) {
    FILE *f = fopen(path, "wb");
    fwrite(b, 1, n, f);
    fclose(f);
}

static void check_model(const char *path, int wtype) {
    ml_model m;
    TT_EQ_INT(ml_model_load(&m, path), ML_OK);
    TT_EQ_INT(m.cfg.dim, 64);
    TT_EQ_INT(m.cfg.n_layers, 2);
    TT_EQ_INT(m.cfg.n_heads, 4);
    TT_EQ_INT(m.cfg.n_kv_heads, 2);
    TT_EQ_INT(m.cfg.hidden_dim, 128);
    TT_EQ_INT(m.cfg.vocab, 96);
    TT_EQ_INT(m.cfg.max_seq, 32);
    TT_EQ_INT(m.cfg.head_dim, 16);
    TT_EQ_INT(m.cfg.kv_dim, 32);
    TT_EQ_INT(m.cfg.wtype, wtype);
    TT_ASSERT(m.cfg.rope_theta == 10000.0f);
    TT_ASSERT(m.cfg.norm_eps == 1e-5f);
    TT_EQ_INT(m.layers[1].wk.rows, 32);
    TT_EQ_INT(m.layers[1].wk.cols, 64);
    TT_EQ_INT(m.layers[0].w2.rows, 64);
    TT_EQ_INT(m.layers[0].w2.cols, 128);
    TT_EQ_INT(m.wcls.rows, 96);
    TT_EQ_INT((uintptr_t)m.layers[1].w3.q % 64, 0);
    TT_EQ_INT((uintptr_t)m.layers[1].w3.d % 64, 0);
    TT_EQ_INT((uintptr_t)m.final_norm % 64, 0);

    char side[1100];
    snprintf(side, sizeof side, "%s.emb5.f32", path);
    size_t n = 0;
    float *ref = (float *)slurp(side, &n);
    TT_ASSERT(ref && n == 64 * sizeof(float));
    float row[64];
    ml_dequant_row(row, &m.tok_emb, 5);
    if (ref) TT_ASSERT(memcmp(row, ref, sizeof row) == 0);
    free(ref);
    ml_model_free(&m);
}

static void test_load_ok(void) {
    check_model(q8_path, ML_Q8_0);
    check_model(q4_path, ML_Q4_0);
}

/* Loads a patched copy of the Q8 model and returns the status. */
static ml_status load_patched(size_t keep_bytes, size_t off, uint32_t val) {
    size_t n;
    unsigned char *b = slurp(q8_path, &n);
    if (off) memcpy(b + off, &val, 4);
    spit(tmp_path, b, keep_bytes ? keep_bytes : n);
    free(b);
    ml_model m;
    ml_status s = ml_model_load(&m, tmp_path);
    if (s == ML_OK) ml_model_free(&m);
    remove(tmp_path);
    return s;
}

static void test_load_rejects_truncated(void) {
    size_t n;
    free(slurp(q8_path, &n));
    TT_EQ_INT(load_patched(n / 2, 0, 0), ML_E_FORMAT);
    TT_EQ_INT(load_patched(100, 0, 0), ML_E_FORMAT);   /* shorter than header */
}

static void test_load_rejects_bad_magic(void) {
    TT_EQ_INT(load_patched(0, 0, 0), ML_OK);   /* unpatched copy loads */
    size_t n;
    unsigned char *b = slurp(q8_path, &n);
    b[0] = 'X';
    spit(tmp_path, b, n);
    free(b);
    ml_model m;
    TT_EQ_INT(ml_model_load(&m, tmp_path), ML_E_FORMAT);
    remove(tmp_path);
    TT_EQ_INT(load_patched(0, 4, 2), ML_E_FORMAT);      /* version 2 */
}

static void test_load_rejects_bad_shape(void) {
    TT_EQ_INT(load_patched(0, 8, 48), ML_E_SHAPE);      /* dim % 32 != 0 */
    TT_EQ_INT(load_patched(0, 20, 3), ML_E_SHAPE);      /* n_heads % n_kv_heads != 0 */
    TT_EQ_INT(load_patched(0, 16, 0), ML_E_SHAPE);      /* n_heads == 0 */
    TT_EQ_INT(load_patched(0, 44, 7), ML_E_FORMAT);     /* unknown wtype */
    TT_EQ_INT(load_patched(0, 28, 0x40000000u), ML_E_SHAPE);   /* absurd vocab */
}

static void test_load_missing_file(void) {
    ml_model m;
    TT_EQ_INT(ml_model_load(&m, "/nonexistent/model.mlb"), ML_E_IO);
}

int TT_ENTRY(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: test_load Q8.mlb Q4.mlb\n"); return 2; }
    q8_path = argv[1];
    q4_path = argv[2];
    snprintf(tmp_path, sizeof tmp_path, "%s.tmp", q8_path);
    TT_RUN(test_load_ok);
    TT_RUN(test_load_rejects_truncated);
    TT_RUN(test_load_rejects_bad_magic);
    TT_RUN(test_load_rejects_bad_shape);
    TT_RUN(test_load_missing_file);
    TT_MAIN_END("test_load");
}
