/* Unicode character classes for the byte-level BPE pre-tokenizer: letters
 * (\p{L}), numbers (\p{N}) and white space (\s, the White_Space property);
 * NFC normalization (Qwen2 tokenizers). */
#ifndef ML_UNICODE_H
#define ML_UNICODE_H
#include <stdint.h>

extern const uint32_t ml_uni_letter[][2];
extern const int ml_uni_letter_n;
extern const uint32_t ml_uni_number[][2];
extern const int ml_uni_number_n;

typedef struct { uint32_t cp; uint16_t off; uint8_t len; } ml_uni_decomp;   /* full canonical */
typedef struct { uint32_t lo, hi; uint8_t ccc; } ml_uni_ccc;               /* non-zero classes */
typedef struct { uint32_t a, b, c; } ml_uni_comp;                           /* a + b -> c, sorted */
extern const uint32_t ml_uni_decomp_pool[];
extern const ml_uni_decomp ml_uni_decomp_tab[];
extern const int ml_uni_decomp_n;
extern const ml_uni_ccc ml_uni_ccc_tab[];
extern const int ml_uni_ccc_n;
extern const ml_uni_comp ml_uni_comp_tab[];
extern const int ml_uni_comp_n;

int ml_uni_is_letter(uint32_t cp);
int ml_uni_is_number(uint32_t cp);
int ml_uni_is_space(uint32_t cp);
/* Decodes the UTF-8 code point at s (n bytes left); *len gets its length.
 * Invalid or truncated sequences decode as U+FFFD with length 1. */
uint32_t ml_utf8_decode(const unsigned char *s, int n, int *len);
/* NFC of the UTF-8 text s[0, n) into out, which holds 3 * n bytes (the most
 * NFC can expand UTF-8); returns the output length, or -1 when out of memory.
 * Bytes of invalid sequences are copied and do not combine. */
int ml_uni_nfc(const unsigned char *s, int n, unsigned char *out);

#endif
