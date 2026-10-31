/* Tokenizers: llama2.c tokenizer.bin (score-based BPE), and the MLT1 format
 * that tools/convert_tokenizer.py writes from a Hugging Face tokenizer.json:
 * SentencePiece-style BPE (Llama 2, TinyLlama, Mistral) or byte-level BPE
 * with the Llama 3 pre-tokenizer. MLT1 tokenizers apply merges by rank, as
 * the Hugging Face tokenizers library does. */
#ifndef ML_TOKENIZER_H
#define ML_TOKENIZER_H
#include <stdio.h>
#include "core/ml.h"

#define ML_TOK_BOS 1   /* llama2.c tokenizer.bin */
#define ML_TOK_EOS 2

enum { ML_TOK_LLAMA2C = 0, ML_TOK_SPM = 1, ML_TOK_BYTE_BPE = 2 };

typedef struct { const char *str; int id; } ml_token_index;

typedef struct {
    int kind;                      /* ML_TOK_* */
    char **vocab;                  /* pieces, NUL-terminated (MLT1: raw bytes) */
    int vocab_size;
    /* llama2.c */
    float *scores;
    ml_token_index *sorted;
    unsigned max_len;
    unsigned char byte_pieces[512];   /* "\x00\0\x01\0..." for <0xXX> tokens */
    /* MLT1 */
    int *piece_len;
    unsigned char *attr;           /* ML_TOKA_* bits */
    char **text;                   /* what decoding prints (SentencePiece: U+2581 -> ' ') */
    int bos, eos;                  /* -1 if none */
    unsigned flags;                /* ML_TOKF_* */
    int byte_id[256];              /* SentencePiece byte-fallback tokens, -1 if absent */
    int unk;
    int *str_slot;                 /* piece -> id hash table */
    unsigned str_mask;
    uint64_t *merge_key;           /* (left << 32 | right) -> rank, merged id */
    int32_t *merge_rank, *merge_out;
    unsigned merge_mask;
    int *added;                    /* tokens matched literally in text, longest first */
    int n_added;
} ml_tokenizer;

enum { ML_TOKA_ADDED = 1, ML_TOKA_BYTE = 2 };
enum { ML_TOKF_ADD_BOS = 1, ML_TOKF_IGNORE_MERGES = 2, ML_TOKF_PREPEND_FIRST = 4,
       ML_TOKF_NFC = 8,          /* byte-level: NFC-normalize text between added tokens */
       ML_TOKF_DIGITS1 = 16,     /* byte-level: \p{N} chunks of one digit (Qwen2), not 1-3 */
       ML_TOKF_GPT2 = 32,        /* byte-level: the GPT-2 pre-tokenizer regex, not Llama 3's */
       ML_TOKF_ISOLATE_DIGITS = 64,   /* byte-level: each \p{N} split off before the regex (SmolLM2) */
       ML_TOKF_ALL = 127 };

/* Loads either format (detected by the "MLT1" magic). vocab: the model's
 * vocabulary size; llama2.c files hold exactly that many tokens, MLT1 files
 * at most that many. vocab <= 0 accepts an MLT1 file of any size. */
ml_status ml_tokenizer_load(ml_tokenizer *t, const char *path, int vocab);
/* Encodes text, with the BOS / EOS tokens if bos / eos are set and the
 * tokenizer has them (MLT1: BOS only if its template adds one); returns the token count (<= max), or -1 if out does not
 * fit or memory runs out. */
int ml_encode(ml_tokenizer *t, const char *text, int bos, int eos, int *out, int max);
/* Text of token tok following token prev (strips the space after BOS). */
const char *ml_decode(ml_tokenizer *t, int prev, int tok);
int ml_tokenizer_bos(const ml_tokenizer *t);
/* Id of the token whose text is exactly `text` (added tokens included), or -1. */
int ml_tokenizer_find(const ml_tokenizer *t, const char *text);
int ml_tokenizer_eos(const ml_tokenizer *t);
void ml_tokenizer_free(ml_tokenizer *t);

/* model/bpe.c: the MLT1 tokenizers. */
ml_status ml_bpe_load(ml_tokenizer *t, FILE *f, int vocab);
int ml_bpe_encode(const ml_tokenizer *t, const char *text, int bos, int eos, int *out, int max);
void ml_bpe_free(ml_tokenizer *t);
int ml_bpe_find(const ml_tokenizer *t, const char *text);

#endif
