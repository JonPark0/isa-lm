/* Row-interleaved ("packed") weight layout used by the ml_k_matvec_*_pk kernels.
 *
 * Rows are taken in groups of G (4, 8 or 16 = the kernel's SIMD lane count);
 * the last group is padded with zero-scale rows. For group g and block b the
 * quants are stored step by step, each step holding 4 bytes for each of the G
 * rows, so one vector load gives every row of the group 4 bytes and one
 * 4-byte activation broadcast feeds a dot-product instruction for all of them:
 *
 *   Q8: 8 steps; step k, row r: q[row][32b + 4k + j] + q8_offset   (j = 0..3)
 *   Q4: 4 steps; step v, row r: the original packed bytes 4v..4v+3 of the
 *       block, i.e. low nibbles = elements 4v+j, high nibbles = 16+4v+j
 *
 * Scales are stored as [g][b][G] floats. q8_offset is 128 for kernels whose
 * dot instruction takes an unsigned weight operand (vpdpbusd), else 0.
 */
#ifndef ML_PACK_H
#define ML_PACK_H
#include "core/ml.h"

size_t ml_pack_quant_bytes(const ml_qtensor *W, int G);
size_t ml_pack_scale_bytes(const ml_qtensor *W, int G);
/* Packs row groups [g0, g1) of W. */
void ml_pack_groups(const ml_qtensor *W, uint8_t *pq, float *pd, int G, int q8_offset, int g0, int g1);
void ml_pack(const ml_qtensor *W, uint8_t *pq, float *pd, int G, int q8_offset);

#endif
