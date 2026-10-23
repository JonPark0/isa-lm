/* Tokenizer tests. argv[1], argv[2]: MLT1 files converted from
 * tests/data/spm_tiny.json and bpe_tiny.json; argv[3]: a llama2.c
 * tokenizer.bin of 96 tokens. Expected ids come from the Hugging Face
 * tokenizers library (tests/check_tokenizer.py compares much more text when
 * that library is installed). */
#include <stdlib.h>
#include <string.h>
#include "model/chat.h"
#include "model/tokenizer.h"
#include "model/unicode.h"
#include "tests/tt.h"

static const char *spm_path, *bpe_path, *l2c_path;

static void check_ids(ml_tokenizer *t, const char *text, const int *want, int n) {
    int got[256];
    const int k = ml_encode(t, text, 1, 0, got, 256);
    TT_EQ_INT(k, n);
    for (int i = 0; i < n && i < k; i++)
        if (got[i] != want[i]) { fprintf(stderr, "  %s: id %d is %d, want %d\n", text, i, got[i], want[i]); TT_ASSERT(0); break; }
}

/* Decoding the ids after BOS gives the text back. */
static void check_roundtrip(ml_tokenizer *t, const char *text) {
    int ids[256];
    const int n = ml_encode(t, text, 1, 0, ids, 256);
    TT_ASSERT(n > 0);
    char out[1024] = "";
    for (int i = 1; i < n; i++) strncat(out, ml_decode(t, ids[i - 1], ids[i]), sizeof out - strlen(out) - 1);
    if (strcmp(out, text)) fprintf(stderr, "  decoded '%s', want '%s'\n", out, text);
    TT_ASSERT(!strcmp(out, text));
}

#define IDS(...) (const int[]){ __VA_ARGS__ }, (int)(sizeof((int[]){ __VA_ARGS__ }) / sizeof(int))

static void test_spm(void) {
    ml_tokenizer t;
    TT_EQ_INT(ml_tokenizer_load(&t, spm_path, 0), ML_OK);
    TT_EQ_INT(t.kind, ML_TOK_SPM);
    TT_EQ_INT(ml_tokenizer_bos(&t), 1);
    TT_EQ_INT(ml_tokenizer_eos(&t), 2);
    check_ids(&t, "hello world", IDS(1, 313));
    check_ids(&t, "it's what they'll do", IDS(1, 382));
    check_ids(&t, "naïve café 日本語", IDS(1, 372, 370));
    check_ids(&t, "  a  b ", IDS(1, 290, 290, 290, 271, 290, 290, 272, 290));
    check_ids(&t, "x\xF0\x9F\x99\x82y", IDS(1, 290, 123, 243, 162, 156, 133, 287));   /* byte fallback */
    check_ids(&t, "<s>hello</s>", IDS(1, 1, 307, 2));                                  /* added tokens */
    check_roundtrip(&t, "hello world");
    check_roundtrip(&t, "x\xF0\x9F\x99\x82y");
    check_roundtrip(&t, "tabs\tand\nnewlines");
    ml_tokenizer_free(&t);
}

static void test_byte_bpe(void) {
    ml_tokenizer t;
    TT_EQ_INT(ml_tokenizer_load(&t, bpe_path, 0), ML_OK);
    TT_EQ_INT(t.kind, ML_TOK_BYTE_BPE);
    TT_EQ_INT(ml_tokenizer_bos(&t), 0);
    check_ids(&t, "hello world", IDS(0, 265, 266));
    check_ids(&t, "it's what they'll do", IDS(0, 287, 272, 313, 316, 273, 308));
    check_ids(&t, "naïve café 日本語", IDS(0, 320, 327, 222, 330));
    check_ids(&t, "tabs\tand\nnewlines\n\n", IDS(0, 321, 325, 200, 329, 304));
    check_ids(&t, "123 4567 89", IDS(0, 318, 222, 319, 24, 222, 276));
    check_ids(&t, "  a  b ", IDS(0, 222, 222, 66, 222, 222, 67, 222));
    check_ids(&t, "<|begin_of_text|>hi", IDS(0, 0, 73, 74));
    check_roundtrip(&t, "it's what they'll do");
    check_roundtrip(&t, "naïve café 日本語 \xF0\x9F\x99\x82\n\n  x");
    ml_tokenizer_free(&t);
}

/* Output too small, model vocabulary too small, damaged files. */
static void test_limits(void) {
    ml_tokenizer t;
    TT_EQ_INT(ml_tokenizer_load(&t, bpe_path, 0), ML_OK);
    int ids[3];
    TT_EQ_INT(ml_encode(&t, "hello world and more", 1, 0, ids, 3), -1);
    ml_tokenizer_free(&t);
    TT_EQ_INT(ml_tokenizer_load(&t, bpe_path, 100), ML_E_SHAPE);

    FILE *f = fopen(bpe_path, "rb");
    TT_ASSERT(f != NULL);
    if (!f) return;
    unsigned char buf[1 << 16];
    const size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    const char *tmp = "build/test_tokenizer.tmp";
    for (size_t cut = 8; cut < n; cut += n / 7) {   /* truncated */
        f = fopen(tmp, "wb");
        fwrite(buf, 1, cut, f);
        fclose(f);
        TT_EQ_INT(ml_tokenizer_load(&t, tmp, 0), ML_E_FORMAT);
    }
    buf[28] = 0x20;   /* unknown flag */
    f = fopen(tmp, "wb");
    fwrite(buf, 1, n, f);
    fclose(f);
    TT_EQ_INT(ml_tokenizer_load(&t, tmp, 0), ML_E_FORMAT);
    buf[28] = 0;
    buf[4] = 9;   /* version */
    f = fopen(tmp, "wb");
    fwrite(buf, 1, n, f);
    fclose(f);
    TT_EQ_INT(ml_tokenizer_load(&t, tmp, 0), ML_E_FORMAT);
    remove(tmp);
}

static void check_nfc(const char *in, const char *want) {
    unsigned char out[64];
    const int n = ml_uni_nfc((const unsigned char *)in, (int)strlen(in), out);
    TT_ASSERT(n == (int)strlen(want) && !memcmp(out, want, (size_t)n));
    if (n != (int)strlen(want) || memcmp(out, want, (size_t)n)) fprintf(stderr, "  NFC of '%s'\n", in);
}

/* NFC and single-digit numbers (Qwen2 flags; check_tokenizer.py compares the
 * real Qwen2.5 tokenizer with the tokenizers library). */
static void test_qwen2_flags(void) {
    check_nfc("cafe\xCC\x81", "caf\xC3\xA9");                              /* e + U+0301 */
    check_nfc("a\xCC\x81\xCC\xA3", "\xE1\xBA\xA1\xCC\x81");                   /* reordered, then a+U+0323 */
    check_nfc("\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8", "\xEA\xB0\x81");     /* jamo -> U+AC01 */
    check_nfc("\xEA\xB0\x80\xE1\x86\xA8", "\xEA\xB0\x81");                   /* LV + T */
    check_nfc("\xE2\x84\xAB", "\xC3\x85");                                   /* U+212B singleton */
    check_nfc("\xE0\xA5\x98", "\xE0\xA4\x95\xE0\xA4\xBC");                   /* U+0958 excluded */
    check_nfc("x\xFF\xCC\x81", "x\xFF\xCC\x81");                              /* invalid byte blocks */
    check_nfc("\xCC\x81" "e", "\xCC\x81" "e");                                /* leading mark */

    ml_tokenizer t;
    TT_EQ_INT(ml_tokenizer_load(&t, bpe_path, 0), ML_OK);
    t.flags |= ML_TOKF_NFC | ML_TOKF_DIGITS1;
    int a[16], b[16];
    const int na = ml_encode(&t, "naive cafe\xCC\x81", 1, 0, a, 16), nb = ml_encode(&t, "naive caf\xC3\xA9", 1, 0, b, 16);
    TT_ASSERT(na == nb && na > 0 && !memcmp(a, b, sizeof(int) * (size_t)na));
    check_ids(&t, "123", IDS(0, 18, 19, 20));
    ml_tokenizer_free(&t);
}

/* The llama2.c format still loads and encodes. */
static void test_llama2c(void) {
    ml_tokenizer t;
    TT_EQ_INT(ml_tokenizer_load(&t, l2c_path, 96), ML_OK);
    TT_EQ_INT(t.kind, ML_TOK_LLAMA2C);
    int ids[16];
    const int n = ml_encode(&t, "ab", 1, 0, ids, 16);
    TT_EQ_INT(n, 4);   /* BOS, " ", "a", "b": one character per token */
    TT_EQ_INT(ids[0], ML_TOK_BOS);
    ml_tokenizer_free(&t);
    TT_EQ_INT(ml_tokenizer_load(&t, l2c_path, 0), ML_E_SHAPE);
}

/* Chat templates render as the models' Hugging Face templates do. */
static void check_render(int tpl, const char *sys, const char *user, int first, const char *want) {
    char *got = ml_chat_render(tpl, sys, user, first);
    TT_ASSERT(got != NULL);
    if (got && strcmp(got, want)) fprintf(stderr, "  rendered '%s'\n  want     '%s'\n", got, want);
    TT_ASSERT(got && !strcmp(got, want));
    free(got);
}

static void test_chat_templates(void) {
    TT_EQ_INT(ml_chat_template("llama3"), ML_CHAT_LLAMA3);
    TT_EQ_INT(ml_chat_template("qwen"), ML_CHAT_QWEN);
    TT_EQ_INT(ml_chat_template("vicuna"), -1);
    check_render(ML_CHAT_LLAMA3, "Be brief.", "Hi", 1,
                 "<|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>"
                 "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    check_render(ML_CHAT_LLAMA3, "Be brief.", "More", 0,   /* the system prompt only once */
                 "<|start_header_id|>user<|end_header_id|>\n\nMore<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    check_render(ML_CHAT_ZEPHYR, "Be kind.", "Hi", 1, "<|system|>\nBe kind.</s>\n<|user|>\nHi</s>\n<|assistant|>\n");
    check_render(ML_CHAT_ZEPHYR, NULL, "More", 0, "\n<|user|>\nMore</s>\n<|assistant|>\n");
    check_render(ML_CHAT_LLAMA2, "Be kind.", "Hi", 1, "[INST] <<SYS>>\nBe kind.\n<</SYS>>\n\nHi [/INST]");
    check_render(ML_CHAT_LLAMA2, NULL, "More", 0, "[INST] More [/INST]");
    check_render(ML_CHAT_CHATML, NULL, "Hi", 1, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    check_render(ML_CHAT_QWEN, NULL, "Hi", 1,
                 "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                 "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    check_render(ML_CHAT_QWEN, "Be kind.", "More", 0, "\n<|im_start|>user\nMore<|im_end|>\n<|im_start|>assistant\n");
    char longmsg[600];
    memset(longmsg, 'x', sizeof longmsg - 1);
    longmsg[sizeof longmsg - 1] = '\0';
    char *r = ml_chat_render(ML_CHAT_LLAMA3, longmsg, longmsg, 1);   /* nothing cut */
    TT_ASSERT(r && strstr(r, "<|start_header_id|>assistant<|end_header_id|>\n\n"));
    free(r);
    ml_tokenizer t;   /* stop tokens are found by their text */
    TT_EQ_INT(ml_tokenizer_load(&t, bpe_path, 0), ML_OK);
    TT_ASSERT(ml_tokenizer_find(&t, "<|eot_id|>") > 0);
    TT_EQ_INT(ml_tokenizer_find(&t, "<|no_such_token|>"), -1);
    ml_tokenizer_free(&t);
}

int TT_ENTRY(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: test_tokenizer SPM.mlt BPE.mlt TOKENIZER.bin\n"); return 2; }
    spm_path = argv[1]; bpe_path = argv[2]; l2c_path = argv[3];
    TT_RUN(test_spm);
    TT_RUN(test_byte_bpe);
    TT_RUN(test_limits);
    TT_RUN(test_qwen2_flags);
    TT_RUN(test_llama2c);
    TT_RUN(test_chat_templates);
    TT_MAIN_END("test_tokenizer");
}
