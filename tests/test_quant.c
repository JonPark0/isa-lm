#include <math.h>
#include <string.h>
#include "config/ml_config.h"
#include "core/quant_ref.h"
#include "tests/tt.h"

static uint32_t rng = 12345;
static float frand(float lo, float hi) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return lo + (hi - lo) * (float)(rng & 0xFFFFFF) / (float)0xFFFFFF;
}

static int8_t qbuf[4096];
static float dbuf[4096 / QK];
static float xbuf[4096];

static void test_quantize_roundtrip(void) {
    int k = 256;
    for (int i = 0; i < k; i++) xbuf[i] = frand(-3, 3);
    ml_qact a = { qbuf, dbuf, k };
    ml_k_quantize_q8(&a, xbuf, k);
    for (int i = 0; i < k; i++) {
        float d = dbuf[i / QK];
        TT_ASSERT(fabsf(xbuf[i] - qbuf[i] * d) <= d / 2 + 1e-7f);
    }
}

static void test_quantize_zero_block(void) {
    for (int i = 0; i < 64; i++) xbuf[i] = i < 32 ? 0.0f : frand(-1, 1);
    ml_qact a = { qbuf, dbuf, 64 };
    ml_k_quantize_q8(&a, xbuf, 64);
    TT_ASSERT(dbuf[0] == 0.0f);
    for (int i = 0; i < 32; i++) TT_EQ_INT(qbuf[i], 0);
    for (int i = 0; i < 2; i++) TT_ASSERT(!isnan(dbuf[i]));
}

static void test_quantize_clamp(void) {
    for (int i = 0; i < 32; i++) xbuf[i] = frand(-1, 1);
    xbuf[3] = -5.0f;   /* amax, negative */
    xbuf[9] = 4.999f;
    ml_qact a = { qbuf, dbuf, 32 };
    ml_k_quantize_q8(&a, xbuf, 32);
    TT_EQ_INT(qbuf[3], -127);
    for (int i = 0; i < 32; i++) TT_ASSERT(qbuf[i] >= -127 && qbuf[i] <= 127);
}

static void test_quantize_ties(void) {
    /* amax = 127 gives id = 1 exactly, so x is its own scaled value. */
    for (int i = 0; i < 32; i++) xbuf[i] = 0.0f;
    xbuf[0] = 127.0f;
    xbuf[1] = 0.5f; xbuf[2] = -0.5f; xbuf[3] = 1.5f; xbuf[4] = 2.5f; xbuf[5] = -2.5f;
    ml_qact a = { qbuf, dbuf, 32 };
    ml_k_quantize_q8(&a, xbuf, 32);
    TT_ASSERT(dbuf[0] == 1.0f);
    TT_EQ_INT(qbuf[1], 1);
    TT_EQ_INT(qbuf[2], -1);
    TT_EQ_INT(qbuf[3], 2);
    TT_EQ_INT(qbuf[4], 3);
    TT_EQ_INT(qbuf[5], -3);
}

/* amax below ~3.7e-37 makes 127/amax overflow to inf. Such blocks are
 * numerically zero; every ISA must quantize them to d = 0, q = 0. */
static void test_quantize_tiny_block(void) {
    for (int i = 0; i < 32; i++) xbuf[i] = (i % 3 == 0) ? 0.0f : (i % 3 == 1 ? 1e-37f : -2e-37f);
    ml_qact a = { qbuf, dbuf, 32 };
    ml_k_quantize_q8(&a, xbuf, 32);
    TT_ASSERT(dbuf[0] == 0.0f);
    for (int i = 0; i < 32; i++) TT_EQ_INT(qbuf[i], 0);
}

static void test_q4_roundtrip(void) {
    int k = 64;
    for (int i = 0; i < k; i++) xbuf[i] = frand(-2, 2);
    ml_quantize_row_q4(qbuf, dbuf, xbuf, k);
    ml_qtensor W = { qbuf, dbuf, 1, k, ML_Q4_0, NULL, NULL, NULL };
    float out[64];
    ml_dequant_row(out, &W, 0);
    for (int i = 0; i < k; i++) {
        float d = dbuf[i / QK];
        TT_ASSERT(fabsf(xbuf[i] - out[i]) <= d / 2 + 1e-6f);
    }
    /* layout: low nibble = element 0..15, high nibble = 16..31 */
    uint8_t b0 = (uint8_t)qbuf[0];
    TT_NEAR(((b0 & 0xF) - 8) * dbuf[0], out[0], 1e-6);
    TT_NEAR(((b0 >> 4) - 8) * dbuf[0], out[16], 1e-6);
}

int TT_ENTRY(int argc, char **argv) {
    (void)argc; (void)argv;
    TT_RUN(test_quantize_roundtrip);
    TT_RUN(test_quantize_zero_block);
    TT_RUN(test_quantize_clamp);
    TT_RUN(test_quantize_ties);
    TT_RUN(test_quantize_tiny_block);
    TT_RUN(test_q4_roundtrip);
    TT_MAIN_END("test_quant");
}
