/* Non-kernel quantization helpers (load-time / test use, scalar only). */
#ifndef ML_QUANT_REF_H
#define ML_QUANT_REF_H
#include "core/ml.h"

void ml_dequant_row(float *out, const ml_qtensor *W, int row);
/* Q4_0: q receives cols/2 packed bytes (low nibble = elem 0..15 of a block). */
void ml_quantize_row_q4(int8_t *q, float *d, const float *x, int k);

#endif
