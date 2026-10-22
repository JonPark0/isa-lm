#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model/tokenizer.h"

static int cmp_tok(const void *a, const void *b) {
    return strcmp(((const ml_token_index *)a)->str, ((const ml_token_index *)b)->str);
}

ml_status ml_tokenizer_load(ml_tokenizer *t, const char *path, int vocab) {
    memset(t, 0, sizeof *t);
    for (int i = 0; i < 256; i++) t->byte_pieces[i * 2] = (unsigned char)i;
    FILE *f = fopen(path, "rb");
    if (!f) return ML_E_IO;
    t->vocab_size = vocab;
    t->vocab = calloc((size_t)vocab, sizeof(char *));
    t->scores = calloc((size_t)vocab, sizeof(float));
    t->sorted = calloc((size_t)vocab, sizeof(ml_token_index));
    ml_status s = (t->vocab && t->scores && t->sorted) ? ML_OK : ML_E_NOMEM;
    int max_len = 0;
    if (s == ML_OK && (fread(&max_len, 4, 1, f) != 1 || max_len <= 0 || max_len > 4096)) s = ML_E_FORMAT;
    t->max_len = (unsigned)max_len;
    for (int i = 0; s == ML_OK && i < vocab; i++) {
        int len = 0;
        if (fread(&t->scores[i], 4, 1, f) != 1 || fread(&len, 4, 1, f) != 1 || len < 0 || len > max_len) {
            s = ML_E_FORMAT;
            break;
        }
        t->vocab[i] = malloc((size_t)len + 1);
        if (!t->vocab[i]) { s = ML_E_NOMEM; break; }
        if (fread(t->vocab[i], 1, (size_t)len, f) != (size_t)len) { s = ML_E_FORMAT; break; }
        t->vocab[i][len] = '\0';
        t->sorted[i] = (ml_token_index){ t->vocab[i], i };
    }
    fclose(f);
    if (s != ML_OK) { ml_tokenizer_free(t); return s; }
    qsort(t->sorted, (size_t)vocab, sizeof *t->sorted, cmp_tok);
    return ML_OK;
}

void ml_tokenizer_free(ml_tokenizer *t) {
    if (t->vocab)
        for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->scores);
    free(t->sorted);
    memset(t, 0, sizeof *t);
}

static int lookup(const ml_tokenizer *t, const char *s) {
    ml_token_index key = { s, 0 };
    const ml_token_index *r = bsearch(&key, t->sorted, (size_t)t->vocab_size, sizeof key, cmp_tok);
    return r ? r->id : -1;
}

int ml_encode(ml_tokenizer *t, const char *text, int bos, int eos, int *out, int max) {
    const size_t cap = t->max_len * 2 + 3;
    char *buf = malloc(cap > 16 ? cap : 16);
    if (!buf) return -1;
    int n = 0;
#define PUSH(id) do { if (n >= max) { free(buf); return -1; } out[n++] = (id); } while (0)
    if (bos) PUSH(ML_TOK_BOS);
    if (text[0]) {
        int sp = lookup(t, " ");
        if (sp >= 0) PUSH(sp);
    }
    /* One token per UTF-8 code point, falling back to <0xXX> byte tokens. */
    size_t len = 0;
    for (const char *c = text; *c; c++) {
        if ((*c & 0xC0) != 0x80) len = 0;
        buf[len++] = *c;
        buf[len] = '\0';
        if ((c[1] & 0xC0) == 0x80 && len < 4) continue;
        int id = lookup(t, buf);
        if (id >= 0) {
            PUSH(id);
        } else {
            for (size_t i = 0; i < len; i++) {
                char hex[8];
                snprintf(hex, sizeof hex, "<0x%02X>", (unsigned char)buf[i]);
                int b = lookup(t, hex);
                PUSH(b >= 0 ? b : 0);
            }
        }
        len = 0;
    }
    /* Greedy BPE: merge the best-scoring adjacent pair until none merges. */
    const int first = bos ? 1 : 0;
    for (;;) {
        float best = -1e10f;
        int best_id = -1, best_at = -1;
        for (int i = first; i + 1 < n; i++) {
            const char *a = t->vocab[out[i]], *b = t->vocab[out[i + 1]];
            if (strlen(a) + strlen(b) + 1 > cap) continue;
            snprintf(buf, cap, "%s%s", a, b);
            int id = lookup(t, buf);
            if (id >= 0 && t->scores[id] > best) { best = t->scores[id]; best_id = id; best_at = i; }
        }
        if (best_at < 0) break;
        out[best_at] = best_id;
        memmove(out + best_at + 1, out + best_at + 2, sizeof(int) * (size_t)(n - best_at - 2));
        n--;
    }
    if (eos) PUSH(ML_TOK_EOS);
#undef PUSH
    free(buf);
    return n;
}

const char *ml_decode(ml_tokenizer *t, int prev, int tok) {
    if (tok < 0 || tok >= t->vocab_size) return "";
    const char *p = t->vocab[tok];
    if (prev == ML_TOK_BOS && p[0] == ' ') p++;
    unsigned int byte;
    if (strlen(p) == 6 && sscanf(p, "<0x%02X>", &byte) == 1) p = (const char *)t->byte_pieces + byte * 2;
    return p;
}
