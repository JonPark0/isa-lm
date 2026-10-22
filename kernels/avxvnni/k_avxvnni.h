/* AVX-VNNI (VEX vpdpbusd, 256-bit) integer dot products. Other ops fall back
 * to AVX2. */
#ifndef ML_K_AVXVNNI_H
#define ML_K_AVXVNNI_H
#include "core/ml.h"

#define ML_HAS_MATVEC_Q8_AVXVNNI
#define ML_HAS_MATVEC_Q4_AVXVNNI
#define ML_HAS_MATVEC_PK_AVXVNNI   /* packed layout: 8-row groups, Q8 offset 128 */

void ml_k_matvec_q8_avxvnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_avxvnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q8_pk_avxvnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_pk_avxvnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);

#endif
