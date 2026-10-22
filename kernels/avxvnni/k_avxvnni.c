#include "kernels/avxvnni/k_avxvnni.h"
#include "kernels/x86_avx.h"

/* vpdpbusd: u8 x s8, 4 products summed per int32 lane, no saturation. */
static inline __m256i dot32_vnni(__m256i w, __m256i x) {
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(x, x), _mm256_sign_epi8(w, x));
}

void ml_k_matvec_q8_avxvnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC256_BODY(LOADW_Q8_256, dot32_vnni)
}

void ml_k_matvec_q4_avxvnni(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1) {
    MATVEC256_BODY(LOADW_Q4_256, dot32_vnni)
}
