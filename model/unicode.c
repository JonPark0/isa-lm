#include "model/unicode.h"

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
