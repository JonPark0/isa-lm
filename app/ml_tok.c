/* ml_tok: encode text to token ids, or decode ids to text, with a tokenizer
 * file (MLT1 from tools/convert_tokenizer.py, or llama2.c tokenizer.bin). */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model/tokenizer.h"

static const char USAGE[] =
    "usage: ml_tok TOKENIZER [options] < input\n"
    "  encodes stdin as UTF-8 text and prints the token ids, space-separated\n"
    "  --decode         read token ids from stdin and print the text instead\n"
    "  --no-bos         do not prepend the BOS token\n"
    "  --vocab N        vocabulary size (needed for llama2.c tokenizer.bin)\n";

static char *read_all(FILE *f, size_t *len) {
    size_t cap = 1 << 16, n = 0;
    char *buf = malloc(cap);
    for (size_t r; buf && (r = fread(buf + n, 1, cap - n - 1, f)) > 0;) {
        n += r;
        if (cap - n < 2) {
            char *b = realloc(buf, cap *= 2);
            if (!b) { free(buf); return NULL; }
            buf = b;
        }
    }
    if (buf) buf[n] = '\0';
    *len = n;
    return buf;
}

int ml_main(int argc, char **argv) {
    const char *path = NULL;
    int decode = 0, bos = 1, vocab = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--decode")) decode = 1;
        else if (!strcmp(argv[i], "--no-bos")) bos = 0;
        else if (!strcmp(argv[i], "--vocab") && i + 1 < argc) vocab = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { printf("%sisa-lm %s\n", USAGE, ML_VERSION); return 0; }
        else if (argv[i][0] != '-' && !path) path = argv[i];
        else { fputs(USAGE, stderr); return 2; }
    }
    if (!path) { fputs(USAGE, stderr); return 2; }
    ml_tokenizer t;
    ml_status st = ml_tokenizer_load(&t, path, vocab);
    if (st != ML_OK) { fprintf(stderr, "error: %s: %s\n", path, ml_status_str(st)); return 1; }
    size_t len;
    char *in = read_all(stdin, &len);
    if (!in) { ml_tokenizer_free(&t); return 1; }
    int rc = 0;
    if (decode) {
        int prev = -1;
        for (char *p = in, *end;; p = end) {
            errno = 0;
            const long id = strtol(p, &end, 10);
            if (end == p) break;
            if (errno || id < 0 || id >= t.vocab_size) { fprintf(stderr, "error: bad token id\n"); rc = 2; break; }
            fputs(ml_decode(&t, prev, (int)id), stdout);
            prev = (int)id;
        }
    } else if (strlen(in) != len) {
        fprintf(stderr, "error: input contains a NUL byte\n");
        rc = 2;
    } else {
        const int max = (int)len * 3 + 8;
        int *ids = malloc(sizeof(int) * (size_t)max);
        const int n = ids ? ml_encode(&t, in, bos, 0, ids, max) : -1;
        if (n < 0) { fprintf(stderr, "error: out of memory\n"); rc = 1; }
        for (int i = 0; i < n; i++) printf(i ? " %d" : "%d", ids[i]);
        if (n >= 0) putchar('\n');
        free(ids);
    }
    free(in);
    ml_tokenizer_free(&t);
    return rc;
}
