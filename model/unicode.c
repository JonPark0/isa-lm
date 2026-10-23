#include "model/unicode.h"

#include <stdlib.h>

static int in_ranges(const uint32_t (*r)[2], int n, uint32_t cp) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid - 1;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}

int ml_uni_is_letter(uint32_t cp) {
    if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
    return in_ranges(ml_uni_letter, ml_uni_letter_n, cp);
}

int ml_uni_is_number(uint32_t cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return in_ranges(ml_uni_number, ml_uni_number_n, cp);
}

/* The White_Space property, which \s matches in Oniguruma's Unicode mode. */
int ml_uni_is_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

uint32_t ml_utf8_decode(const unsigned char *s, int n, int *len) {
    const unsigned c = s[0];
    int need;
    uint32_t cp, min;
    if (c < 0x80) { *len = 1; return c; }
    if ((c & 0xE0) == 0xC0) { need = 1; cp = c & 0x1F; min = 0x80; }
    else if ((c & 0xF0) == 0xE0) { need = 2; cp = c & 0x0F; min = 0x800; }
    else if ((c & 0xF8) == 0xF0) { need = 3; cp = c & 0x07; min = 0x10000; }
    else { *len = 1; return 0xFFFD; }
    if (need >= n) { *len = 1; return 0xFFFD; }
    for (int i = 1; i <= need; i++) {
        if ((s[i] & 0xC0) != 0x80) { *len = 1; return 0xFFFD; }
        cp = (cp << 6) | (s[i] & 0x3F);
    }
    if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) { *len = 1; return 0xFFFD; }
    *len = need + 1;
    return cp;
}

/* ---- NFC ---------------------------------------------------------------- */

enum { HS = 0xAC00, HL = 0x1100, HV = 0x1161, HT = 0x11A7, NV = 21, NT = 28, NS = 11172 };
#define RAW 0x80000000u   /* a byte of an invalid sequence */

static int ccc(uint32_t cp) {
    if (cp < 0x300 || (cp & RAW)) return 0;
    int lo = 0, hi = ml_uni_ccc_n - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        if (cp < ml_uni_ccc_tab[mid].lo) hi = mid - 1;
        else if (cp > ml_uni_ccc_tab[mid].hi) lo = mid + 1;
        else return ml_uni_ccc_tab[mid].ccc;
    }
    return 0;
}

/* Appends the full canonical decomposition of cp to d; returns its length. */
static int decompose(uint32_t cp, uint32_t *d) {
    if (cp - HS < NS) {
        const uint32_t i = cp - HS, t = i % NT;
        d[0] = HL + i / (NV * NT);
        d[1] = HV + i % (NV * NT) / NT;
        if (t) d[2] = HT + t;
        return t ? 3 : 2;
    }
    if (cp >= 0xC0 && !(cp & RAW)) {
        int lo = 0, hi = ml_uni_decomp_n - 1;
        while (lo <= hi) {
            const int mid = (lo + hi) / 2;
            const ml_uni_decomp *e = &ml_uni_decomp_tab[mid];
            if (cp < e->cp) hi = mid - 1;
            else if (cp > e->cp) lo = mid + 1;
            else {
                for (int k = 0; k < e->len; k++) d[k] = ml_uni_decomp_pool[e->off + k];
                return e->len;
            }
        }
    }
    d[0] = cp;
    return 1;
}

/* The primary composite of a + b, or 0. */
static uint32_t compose(uint32_t a, uint32_t b) {
    if (a - HL < 19 && b - HV < NV) return HS + ((a - HL) * NV + (b - HV)) * NT;
    if (a - HS < NS && (a - HS) % NT == 0 && b - HT - 1 < NT - 1) return a + (b - HT);
    int lo = 0, hi = ml_uni_comp_n - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        const ml_uni_comp *e = &ml_uni_comp_tab[mid];
        if (a < e->a || (a == e->a && b < e->b)) hi = mid - 1;
        else if (a > e->a || b > e->b) lo = mid + 1;
        else return e->c;
    }
    return 0;
}

static int put_utf8(uint32_t cp, unsigned char *o) {
    if (cp & RAW) { o[0] = (unsigned char)cp; return 1; }
    if (cp < 0x80) { o[0] = (unsigned char)cp; return 1; }
    if (cp < 0x800) { o[0] = (unsigned char)(0xC0 | cp >> 6); o[1] = (unsigned char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        o[0] = (unsigned char)(0xE0 | cp >> 12); o[1] = (unsigned char)(0x80 | (cp >> 6 & 0x3F));
        o[2] = (unsigned char)(0x80 | (cp & 0x3F));
        return 3;
    }
    o[0] = (unsigned char)(0xF0 | cp >> 18); o[1] = (unsigned char)(0x80 | (cp >> 12 & 0x3F));
    o[2] = (unsigned char)(0x80 | (cp >> 6 & 0x3F)); o[3] = (unsigned char)(0x80 | (cp & 0x3F));
    return 4;
}

int ml_uni_nfc(const unsigned char *s, int n, unsigned char *out) {
    /* decomposition: under 2 code points per input byte (U+01D5: 2 bytes -> 3) */
    uint32_t *cp = malloc(sizeof(uint32_t) * (3 * (size_t)n + 4));
    if (!cp) return -1;
    int m = 0;
    for (int i = 0; i < n;) {
        int len;
        uint32_t c = ml_utf8_decode(s + i, n - i, &len);
        if (c == 0xFFFD && len == 1) c = RAW | s[i];   /* a real U+FFFD has length 3 */
        m += decompose(c, cp + m);
        i += len;
    }
    /* canonical ordering: stable sort of each run of non-starters by class */
    for (int i = 1; i < m; i++) {
        const uint32_t c = cp[i];
        const int k = ccc(c);
        if (!k) continue;
        int j = i;
        while (j > 0 && ccc(cp[j - 1]) > k) { cp[j] = cp[j - 1]; j--; }
        cp[j] = c;
    }
    /* composition */
    int w = m ? 1 : 0, st = 0, last = m && ccc(cp[0]) ? 256 : 0;
    for (int i = 1; i < m; i++) {
        const uint32_t c = cp[i];
        const int k = ccc(c);
        const uint32_t x = (cp[st] & RAW) || (c & RAW) ? 0 : compose(cp[st], c);
        if (x && (last < k || last == 0)) {
            cp[st] = x;
            continue;
        }
        if (k == 0) st = w;
        last = k;
        cp[w++] = c;
    }
    int o = 0;
    for (int i = 0; i < w; i++) o += put_utf8(cp[i], out + o);
    free(cp);
    return o;
}
