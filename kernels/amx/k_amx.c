#include <immintrin.h>
#include <string.h>
#include "kernels/amx/k_amx.h"
#include "kernels/x86_pack.h"

/* The avx512vnni packed layout is AMX's B operand as it is: per 16-row group
 * and block, 8 rows of 64 bytes, each holding 4 consecutive quants of all
 * 16 rows. One tile dot product gives a block's int32 sums for up to 16
 * tokens x 16 rows, and AVX-512 applies the scales. Q8 weights are unsigned
 * (+ 128): the sum tile starts at -128 * sum(x) per token (loaded from nxs,
 * built once per tile of tokens), so tdpbsud leaves the matvec's sums. Q4
 * nibbles are widened to q - 8 as signed bytes for tdpbssd, which needs no
 * correction.
 * Per token and row the operations are the matvec's, in the same order, so
 * prefill matches decoding bit for bit.
 *
 * Tiles: 0, 1 sums (two blocks in flight); 2, 3 activations; 4, 5 weights. */
#define PK 16
#define TOK 16   /* tokens per tile */

typedef struct {
    uint8_t palette, start_row, reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} tile_config;

static void tiles_configure(int m) {
    tile_config c;
    memset(&c, 0, sizeof c);
    c.palette = 1;
    for (int i = 0; i < 2; i++) { c.rows[i] = (uint8_t)m; c.colsb[i] = PK * 4; }   /* m x 16 int32 */
    for (int i = 2; i < 4; i++) { c.rows[i] = (uint8_t)m; c.colsb[i] = QK; }       /* m x 32 int8 */
    for (int i = 4; i < 6; i++) { c.rows[i] = QK / 4; c.colsb[i] = PK * 4; }       /* 8 x 64 uint8 */
    _tile_loadconfig(&c);
}

/* acc[t] += sums[t] * (d[row] * dx[t][b]) for the M tokens of a tile. */
ML_INLINE void apply_scales(__m512 *acc, const int32_t *sums, const float *dx, int nb, int b, __m512 dv, const int M) {
    ML_UNROLL(16) for (int t = 0; t < M; t++)
        acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512((const void *)(sums + t * PK))),
                                 _mm512_mul_ps(dv, _mm512_set1_ps(dx[(size_t)t * nb + b])), acc[t]);
}

/* Q4: the group's block as B, nibbles widened to q - 8 (low nibbles hold
 * quants 0-15 of the block, high ones 16-31: k-steps 0-3 and 4-7). */
static inline void widen_q4(uint8_t *bt, const uint8_t *p) {
    const __m512i m4 = _mm512_set1_epi8(0x0F), eight = _mm512_set1_epi8(8);
    for (int v = 0; v < 4; v++) {
        const __m512i w = _mm512_loadu_si512((const void *)(p + v * PK * 4));
        _mm512_store_si512((void *)(bt + v * PK * 4), _mm512_sub_epi8(_mm512_and_si512(w, m4), eight));
        _mm512_store_si512((void *)(bt + (v + 4) * PK * 4), _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(w, 4), m4), eight));
    }
}

/* Q8's starting sums: nxs[b][t][r] = -128 * sum(x_t, block b) for all 16 r. */
static ML_THREAD_LOCAL _Alignas(64) int32_t nxs[PK_MAX_BLOCKS][TOK * PK];

static void start_sums(const ml_qbatch *X, int t0, int M, int nb) {
    for (int b = 0; b < nb; b++)
        for (int t = 0; t < M; t++) {
            const __m256i v = _mm256_loadu_si256((const __m256i *)(X->q + (size_t)(t0 + t) * X->k + b * QK));
            /* psadbw on x ^ 0x80 sums x + 128 */
            const __m256i sad = _mm256_sad_epu8(_mm256_xor_si256(v, _mm256_set1_epi8((char)0x80)), _mm256_setzero_si256());
            const __m128i h = _mm_add_epi64(_mm256_castsi256_si128(sad), _mm256_extracti128_si256(sad, 1));
            const int sum = (int)(_mm_cvtsi128_si64(h) + _mm_extract_epi64(h, 1)) - 32 * 128;
            _mm512_store_si512((void *)(nxs[b] + t * PK), _mm512_set1_epi32(-128 * sum));
        }
}

/* One 16-row group for the M tokens from t0. Blocks go two at a time: their
 * tile products are stored to a ring of four sums buffers, and the scales of
 * the previous two blocks are applied meanwhile, so that no load waits for
 * the tile store just issued. Blocks are still scaled in order. */
ML_INLINE void group_tokens(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int row, int r1, int t0,
                            const int M, const int q4) {
    const int nb = W->cols / QK;
    const size_t bq = q4 ? 4 * PK * 4 : 8 * PK * 4;   /* packed bytes per block */
    const uint8_t *p = W->pq + (size_t)(row / PK) * nb * bq;
    const float *d = W->pd + (size_t)(row / PK) * nb * PK;
    const int8_t *xq = X->q + (size_t)t0 * X->k;
    const float *dx = X->d + (size_t)t0 * nb;
    _Alignas(64) int32_t sums[4][TOK * PK];
    _Alignas(64) uint8_t bt[2][8 * PK * 4];
    __m512 acc[TOK];
    ML_UNROLL(16) for (int t = 0; t < M; t++) acc[t] = _mm512_setzero_ps();
    for (int b = 0; b < nb; b += 2) {
        const int two = b + 1 < nb;
        _tile_loadd(2, xq + b * QK, X->k);
        if (q4) { _tile_zero(0); widen_q4(bt[0], p + b * bq); _tile_loadd(4, bt[0], PK * 4); _tile_dpbssd(0, 2, 4); }
        else { _tile_loadd(0, nxs[b], PK * 4); _tile_loadd(4, p + b * bq, PK * 4); _tile_dpbsud(0, 2, 4); }
        if (two) {
            _tile_loadd(3, xq + (b + 1) * QK, X->k);
            if (q4) { _tile_zero(1); widen_q4(bt[1], p + (b + 1) * bq); _tile_loadd(5, bt[1], PK * 4); _tile_dpbssd(1, 3, 5); }
            else { _tile_loadd(1, nxs[b + 1], PK * 4); _tile_loadd(5, p + (b + 1) * bq, PK * 4); _tile_dpbsud(1, 3, 5); }
        }
        if (b >= 2) {   /* blocks b-2 and b-1, stored one pass ago */
            apply_scales(acc, sums[(b - 2) & 3], dx, nb, b - 2, _mm512_loadu_ps(d + (b - 2) * PK), M);
            apply_scales(acc, sums[(b - 1) & 3], dx, nb, b - 1, _mm512_loadu_ps(d + (b - 1) * PK), M);
        }
        _tile_stored(0, sums[b & 3], PK * 4);
        if (two) _tile_stored(1, sums[(b + 1) & 3], PK * 4);
    }
    for (int b = nb >= 2 ? (nb - 1) & ~1 : 0; b < nb; b++)   /* the last one or two blocks */
        apply_scales(acc, sums[b & 3], dx, nb, b, _mm512_loadu_ps(d + b * PK), M);
    const __mmask16 live = row + PK <= r1 ? (__mmask16)0xFFFF : (__mmask16)((1u << (r1 - row)) - 1);
    ML_UNROLL(16) for (int t = 0; t < M; t++) _mm512_mask_storeu_ps(Y + (size_t)(t0 + t) * ldy + row, live, acc[t]);
}

/* M tokens (a constant per instantiation, so that acc stays in registers). */
#define GROUPS(M) for (int row = b0; row < b1; row += PK) group_tokens(Y, ldy, W, X, row, b1, t0, M, q4)

static void matmul_amx(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1, int q4) {
    const int nb = W->cols / QK;
    int configured = 0;   /* token count the tiles are configured for */
    /* rows in L2-sized blocks, each read by every tile of tokens */
    for (int rb = pk_row_block(W, PK), b0 = r0; b0 < r1; b0 += rb) {
        const int b1 = r1 - b0 < rb ? r1 : b0 + rb;
        for (int t0 = 0; t0 < X->n; t0 += TOK) {
            const int m = X->n - t0 < TOK ? X->n - t0 : TOK;
            if (!q4) start_sums(X, t0, m, nb);
            if (m != configured) { tiles_configure(m); configured = m; }
            switch (m) {
            case 16: GROUPS(16); break;
            case 15: GROUPS(15); break;
            case 14: GROUPS(14); break;
            case 13: GROUPS(13); break;
            case 12: GROUPS(12); break;
            case 11: GROUPS(11); break;
            case 10: GROUPS(10); break;
            case 9: GROUPS(9); break;
            case 8: GROUPS(8); break;
            case 7: GROUPS(7); break;
            case 6: GROUPS(6); break;
            case 5: GROUPS(5); break;
            case 4: GROUPS(4); break;
            case 3: GROUPS(3); break;
            case 2: GROUPS(2); break;
            default: GROUPS(1); break;
            }
        }
    }
    if (configured) _tile_release();
}

void ml_k_matmul_q8_pk_amx(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    matmul_amx(Y, ldy, W, X, r0, r1, 0);
}

void ml_k_matmul_q4_pk_amx(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1) {
    matmul_amx(Y, ldy, W, X, r0, r1, 1);
}
