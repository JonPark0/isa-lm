/* Minimal test harness: each test binary is a list of TT_RUN calls. */
#ifndef ML_TT_H
#define ML_TT_H
#include <math.h>
#include <stdio.h>

static int tt_pass, tt_fail, tt_cur_fail;

#define TT_ASSERT(c) do { if (!(c)) { \
    fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__, #c); \
    tt_cur_fail = 1; } } while (0)

#define TT_EQ_INT(a, b) do { long long tt_a = (long long)(a), tt_b = (long long)(b); \
    if (tt_a != tt_b) { fprintf(stderr, "%s:%d: %s == %s failed (%lld != %lld)\n", \
        __FILE__, __LINE__, #a, #b, tt_a, tt_b); tt_cur_fail = 1; } } while (0)

static inline int tt_near(double a, double b, double rtol) {
    double m = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    if (m < 1e-6) m = 1e-6;
    return fabs(a - b) <= rtol * m;
}

#define TT_NEAR(a, b, rtol) do { double tt_x = (a), tt_y = (b); \
    if (!tt_near(tt_x, tt_y, rtol)) { fprintf(stderr, "%s:%d: %s ~ %s failed (%.9g vs %.9g)\n", \
        __FILE__, __LINE__, #a, #b, tt_x, tt_y); tt_cur_fail = 1; } } while (0)

#define TT_RUN(fn) do { tt_cur_fail = 0; fn(); \
    if (tt_cur_fail) { tt_fail++; fprintf(stderr, "FAIL %s\n", #fn); } else tt_pass++; } while (0)

#define TT_MAIN_END(name) do { \
    if (tt_fail) printf("%s: %d passed, %d failed\n", name, tt_pass, tt_fail); \
    else printf("%s: %d passed\n", name, tt_pass); \
    return tt_fail ? 1 : 0; } while (0)

/* Executables enter through main_guard.c once the ISA guard exists. */
#ifndef TT_ENTRY
#define TT_ENTRY main
#endif

#endif
