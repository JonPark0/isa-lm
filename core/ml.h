/* Core types shared by every layer. */
#ifndef ML_H
#define ML_H
#include <stddef.h>
#include <stdint.h>

typedef enum {
    ML_OK = 0,
    ML_E_IO,
    ML_E_FORMAT,
    ML_E_SHAPE,
    ML_E_NOMEM,
    ML_E_ISA
} ml_status;

#define QK 32          /* quantization block size */
#define ML_ALIGN 64    /* alignment of every tensor array */

enum { ML_F32 = 0, ML_Q8_0 = 1, ML_Q4_0 = 2 };

/* Quantized weight matrix, SoA: q[rows*cols] (Q4: rows*cols/2 packed bytes),
 * d[rows*cols/QK] scales. */
typedef struct {
    const int8_t *q;
    const float *d;
    int rows, cols;
    uint8_t type;
} ml_qtensor;

/* Q8_0 quantized activation vector of length n. */
typedef struct {
    int8_t *q;
    float *d;
    int n;
} ml_qact;

const char *ml_status_str(ml_status s);

/* *r = a * b; returns 0 if that overflows size_t (32-bit targets!). */
static inline int ml_mul_size(size_t a, size_t b, size_t *r) {
    if (b && a > (size_t)-1 / b) return 0;
    *r = a * b;
    return 1;
}

/* 64 B-aligned allocation (C11 aligned_alloc is missing on MSVC/MinGW). */
void *ml_aligned_alloc(size_t n);
void ml_aligned_free(void *p);

#endif
