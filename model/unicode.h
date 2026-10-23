/* Unicode character classes for the byte-level BPE pre-tokenizer: letters
 * (\p{L}), numbers (\p{N}) and white space (\s, the White_Space property). */
#ifndef ML_UNICODE_H
#define ML_UNICODE_H
#include <stdint.h>

extern const uint32_t ml_uni_letter[][2];
extern const int ml_uni_letter_n;
extern const uint32_t ml_uni_number[][2];
extern const int ml_uni_number_n;

int ml_uni_is_letter(uint32_t cp);
int ml_uni_is_number(uint32_t cp);
int ml_uni_is_space(uint32_t cp);
/* Decodes the UTF-8 code point at s (n bytes left); *len gets its length.
 * Invalid or truncated sequences decode as U+FFFD with length 1. */
uint32_t ml_utf8_decode(const unsigned char *s, int n, int *len);

#endif
