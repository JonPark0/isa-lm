#include "config/ml_config.h"
#include "core/isa_check.h"

#define REQUIRE(cond, name) do { if (!(cond)) { *missing = (name); return ML_E_ISA; } } while (0)

const char *ml_isa_build_name(void) { return ML_ISA_NAME; }

#if defined(__x86_64__) || defined(_M_X64)

#ifdef _MSC_VER
#include <intrin.h>
static void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
    int t[4], mx[4];
    __cpuid(mx, (int)(leaf & 0x80000000u));
    r[0] = r[1] = r[2] = r[3] = 0;
    if ((unsigned)mx[0] < leaf) return;
    __cpuidex(t, (int)leaf, (int)sub);
    for (int i = 0; i < 4; i++) r[i] = (unsigned)t[i];
}
static unsigned long long xgetbv0(void) { return _xgetbv(0); }
#else
#include <cpuid.h>
static void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
    if (!__get_cpuid_count(leaf, sub, &r[0], &r[1], &r[2], &r[3])) r[0] = r[1] = r[2] = r[3] = 0;
}
static unsigned long long xgetbv0(void) {
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
}
#endif

#define BIT(r, n) (((r) >> (n)) & 1u)

ml_status ml_isa_check(const char **missing) {
    unsigned l1[4], l7[4], l71[4];
    cpuid(1, 0, l1);
    cpuid(7, 0, l7);
    cpuid(7, 1, l71);
    const int osxsave = BIT(l1[2], 27);
    const unsigned long long xcr0 = osxsave ? xgetbv0() : 0;
    (void)l71; (void)xcr0;
    *missing = NULL;

    REQUIRE(BIT(l1[3], 26), "sse2");
#if defined(ML_ISA_SSE41) || defined(ML_ISA_AVX2) || defined(ML_ISA_AVXVNNI) || defined(ML_ISA_AVX512VNNI)
    REQUIRE(BIT(l1[2], 9), "ssse3");
    REQUIRE(BIT(l1[2], 19), "sse4.1");
#endif
#if defined(ML_ISA_AVX2) || defined(ML_ISA_AVXVNNI) || defined(ML_ISA_AVX512VNNI)
    REQUIRE(BIT(l1[2], 28), "avx");
    REQUIRE(osxsave && (xcr0 & 0x6) == 0x6, "os-avx-state");
    REQUIRE(BIT(l1[2], 12), "fma");
    REQUIRE(BIT(l7[1], 5), "avx2");
#endif
#if defined(ML_ISA_AVXVNNI)
    REQUIRE(BIT(l71[0], 4), "avx-vnni");
#endif
#if defined(ML_ISA_AVX512VNNI)
    REQUIRE(BIT(l7[1], 16), "avx512f");
    REQUIRE(BIT(l7[1], 30), "avx512bw");
    REQUIRE(BIT(l7[2], 11), "avx512-vnni");
    REQUIRE((xcr0 & 0xE6) == 0xE6, "os-avx512-state");
#endif
    return ML_OK;
}

#elif (defined(__aarch64__) || defined(__arm__)) && defined(__linux__)

#include <sys/auxv.h>

ml_status ml_isa_check(const char **missing) {
    const unsigned long hw = getauxval(AT_HWCAP);
    (void)hw;
    *missing = NULL;
#if defined(__aarch64__)
#if defined(ML_ISA_NEON) || defined(ML_ISA_NEON_DOTPROD) || defined(ML_ISA_NEON_I8MM) || defined(ML_ISA_SVE)
    REQUIRE(hw & (1ul << 1), "neon");
#endif
#if defined(ML_ISA_NEON_DOTPROD) || defined(ML_ISA_NEON_I8MM) || defined(ML_ISA_SVE)
    REQUIRE(hw & (1ul << 20), "asimddp");
#endif
#if defined(ML_ISA_NEON_I8MM)
    REQUIRE(getauxval(AT_HWCAP2) & (1ul << 13), "i8mm");
#endif
#if defined(ML_ISA_SVE)
    REQUIRE(hw & (1ul << 22), "sve");
#endif
#else
#if defined(ML_ISA_NEON)
    REQUIRE(hw & (1ul << 12), "neon");
#endif
#endif
    return ML_OK;
}

#else

/* No portable runtime query: trust the build configuration. */
ml_status ml_isa_check(const char **missing) {
    *missing = NULL;
    return ML_OK;
}

#endif
