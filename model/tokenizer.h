/* BPE tokenizer in llama2.c tokenizer.bin format. */
#ifndef ML_TOKENIZER_H
#define ML_TOKENIZER_H
#include "core/ml.h"

#define ML_TOK_BOS 1
#define ML_TOK_EOS 2

typedef struct { const char *str; int id; } ml_token_index;

typedef struct {
    char **vocab;
    float *scores;
    ml_token_index *sorted;
    int vocab_size;
    unsigned max_len;
    unsigned char byte_pieces[512];   /* "\x00\0\x01\0..." for <0xXX> tokens */
} ml_tokenizer;

ml_status ml_tokenizer_load(ml_tokenizer *t, const char *path, int vocab);
/* Encodes text; returns the token count (<= max), or -1 if out does not fit. */
int ml_encode(ml_tokenizer *t, const char *text, int bos, int eos, int *out, int max);
const char *ml_decode(ml_tokenizer *t, int prev, int tok);
void ml_tokenizer_free(ml_tokenizer *t);

#endif
