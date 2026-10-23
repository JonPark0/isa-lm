/* MLT1 tokenizers (tools/convert_tokenizer.py): BPE with merges applied by
 * rank, lowest rank first and leftmost among equals, as in the Hugging Face
 * tokenizers library.
 *
 * File layout, little-endian:
 *   "MLT1", u32 version (1), u32 kind (ML_TOK_SPM or ML_TOK_BYTE_BPE),
 *   u32 n_vocab, u32 n_merges, i32 bos, i32 eos, u32 flags (ML_TOKF_*),
 *   n_vocab x { u8 attr (ML_TOKA_*), u32 len, len bytes },
 *   n_merges x { u32 left, u32 right, u32 merged }   (in rank order)
 * Byte-level pieces are stored as raw bytes, SentencePiece pieces as UTF-8
 * with U+2581 for spaces, and byte-fallback pieces <0xXX> as their byte. */
#include <stdlib.h>
#include <string.h>
#include "model/tokenizer.h"
#include "model/unicode.h"

static uint32_t hash_bytes(const unsigned char *s, int n) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) h = (h ^ s[i]) * 16777619u;
    return h;
}

static uint32_t hash64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    return (uint32_t)k;
}

static unsigned table_mask(unsigned n) {
    unsigned cap = 16;
    while (cap < 2 * n) cap <<= 1;
    return cap - 1;
}

static int find_piece(const ml_tokenizer *t, const unsigned char *s, int n) {
    for (uint32_t h = hash_bytes(s, n) & t->str_mask;; h = (h + 1) & t->str_mask) {
        const int id = t->str_slot[h];
        if (id < 0) return -1;
        if (t->piece_len[id] == n && !memcmp(t->vocab[id], s, (size_t)n)) return id;
    }
}

/* Rank of the merge (a, b), or -1; *out gets the merged token. */
static int find_merge(const ml_tokenizer *t, int a, int b, int *out) {
    const uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    for (uint32_t h = hash64(key) & t->merge_mask;; h = (h + 1) & t->merge_mask) {
        if (t->merge_rank[h] < 0) return -1;
        if (t->merge_key[h] == key) { *out = t->merge_out[h]; return t->merge_rank[h]; }
    }
}

static int read_u32(FILE *f, uint32_t *v) { return fread(v, 4, 1, f) == 1; }


typedef struct { int len, id; } added_key;

static int cmp_added(const void *a, const void *b) {
    const added_key *x = a, *y = b;
    return x->len != y->len ? y->len - x->len : x->id - y->id;
}

ml_status ml_bpe_load(ml_tokenizer *t, FILE *f, int vocab) {
    uint32_t ver, kind, nv, nm, flags;
    int32_t bos, eos;
    if (!read_u32(f, &ver) || !read_u32(f, &kind) || !read_u32(f, &nv) || !read_u32(f, &nm) ||
        fread(&bos, 4, 1, f) != 1 || fread(&eos, 4, 1, f) != 1 || !read_u32(f, &flags))
        return ML_E_FORMAT;
    if (ver != 1 || (kind != ML_TOK_SPM && kind != ML_TOK_BYTE_BPE) || nv == 0 || nv > (1u << 24) ||
        nm > (1u << 24) || bos < -1 || bos >= (int32_t)nv || eos < -1 || eos >= (int32_t)nv)
        return ML_E_FORMAT;
    if (vocab > 0 && nv > (uint32_t)vocab) return ML_E_SHAPE;
    t->kind = (int)kind;
    t->vocab_size = (int)nv;
    t->bos = bos;
    t->eos = eos;
    t->flags = flags;
    t->unk = -1;
    for (int i = 0; i < 256; i++) t->byte_id[i] = -1;
    t->vocab = calloc(nv, sizeof(char *));
    t->text = calloc(nv, sizeof(char *));
    t->piece_len = calloc(nv, sizeof(int));
    t->attr = calloc(nv, 1);
    t->str_mask = table_mask(nv);
    t->str_slot = malloc(sizeof(int) * (t->str_mask + 1));
    t->merge_mask = table_mask(nm);
    t->merge_key = malloc(sizeof(uint64_t) * (t->merge_mask + 1));
    t->merge_rank = malloc(sizeof(int32_t) * (t->merge_mask + 1));
    t->merge_out = malloc(sizeof(int32_t) * (t->merge_mask + 1));
    if (!t->vocab || !t->text || !t->piece_len || !t->attr || !t->str_slot || !t->merge_key || !t->merge_rank || !t->merge_out)
        return ML_E_NOMEM;
    memset(t->str_slot, 0xFF, sizeof(int) * (t->str_mask + 1));
    memset(t->merge_rank, 0xFF, sizeof(int32_t) * (t->merge_mask + 1));

    for (uint32_t i = 0; i < nv; i++) {
        unsigned char a;
        uint32_t len;
        if (fread(&a, 1, 1, f) != 1 || !read_u32(f, &len) || len > (1u << 16)) return ML_E_FORMAT;
        char *p = malloc(len + 1);
        if (!p) return ML_E_NOMEM;
        t->vocab[i] = p;
        if (fread(p, 1, len, f) != len) return ML_E_FORMAT;
        p[len] = '\0';
        t->piece_len[i] = (int)len;
        t->attr[i] = a;
        if (a & ML_TOKA_BYTE) {
            if (len != 1) return ML_E_FORMAT;
            t->byte_id[(unsigned char)p[0]] = (int)i;
            continue;   /* reached through byte_id, never by string */
        }
        if (len == 0) continue;
        uint32_t h = hash_bytes((const unsigned char *)p, (int)len) & t->str_mask;
        for (; t->str_slot[h] >= 0; h = (h + 1) & t->str_mask)
            if (t->piece_len[t->str_slot[h]] == (int)len && !memcmp(t->vocab[t->str_slot[h]], p, len)) break;
        if (t->str_slot[h] < 0) t->str_slot[h] = (int)i;   /* duplicates: the lowest id wins */
        if (a & ML_TOKA_ADDED) t->n_added++;
    }
    for (uint32_t r = 0; r < nm; r++) {
        uint32_t m[3];
        if (fread(m, 4, 3, f) != 3 || m[0] >= nv || m[1] >= nv || m[2] >= nv) return ML_E_FORMAT;
        const uint64_t key = ((uint64_t)m[0] << 32) | m[1];
        uint32_t h = hash64(key) & t->merge_mask;
        while (t->merge_rank[h] >= 0 && t->merge_key[h] != key) h = (h + 1) & t->merge_mask;
        if (t->merge_rank[h] >= 0) continue;   /* duplicate pair: keep the lower rank */
        t->merge_key[h] = key;
        t->merge_rank[h] = (int32_t)r;
        t->merge_out[h] = (int32_t)m[2];
    }

    /* SentencePiece prints U+2581 as a space. */
    for (uint32_t i = 0; i < nv; i++) {
        if (kind != ML_TOK_SPM || (t->attr[i] & ML_TOKA_BYTE)) { t->text[i] = t->vocab[i]; continue; }
        const char *p = t->vocab[i];
        char *q = malloc((size_t)t->piece_len[i] + 1), *o = q;
        if (!q) return ML_E_NOMEM;
        for (; *p; p++) {
            if ((unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x96 && (unsigned char)p[2] == 0x81) { *o++ = ' '; p += 2; }
            else *o++ = *p;
        }
        *o = '\0';
        t->text[i] = q;
    }
    /* added tokens, longest first, so the longest match wins at a position */
    added_key *keys = malloc(sizeof(added_key) * (size_t)(t->n_added ? t->n_added : 1));
    t->added = malloc(sizeof(int) * (size_t)(t->n_added ? t->n_added : 1));
    if (!keys || !t->added) { free(keys); return ML_E_NOMEM; }
    int k = 0;
    for (uint32_t i = 0; i < nv; i++)
        if ((t->attr[i] & ML_TOKA_ADDED) && !(t->attr[i] & ML_TOKA_BYTE) && t->piece_len[i] > 0)
            keys[k++] = (added_key){ t->piece_len[i], (int)i };
    qsort(keys, (size_t)k, sizeof *keys, cmp_added);
    for (int i = 0; i < k; i++) t->added[i] = keys[i].id;
    t->n_added = k;
    free(keys);
    return ML_OK;
}

void ml_bpe_free(ml_tokenizer *t) {
    for (int i = 0; t->vocab && i < t->vocab_size; i++) {
        if (t->text && t->text[i] != t->vocab[i]) free(t->text[i]);
        free(t->vocab[i]);
    }
    free(t->vocab); free(t->text); free(t->piece_len); free(t->attr); free(t->str_slot);
    free(t->merge_key); free(t->merge_rank); free(t->merge_out); free(t->added);
}

/* Applies merges to sym[0, n) until none applies, lowest rank first and the
 * leftmost among equal ranks, as the tokenizers library does; returns the new
 * length. Symbols form a linked list and candidate pairs sit in a min-heap
 * keyed (rank, left position), so a word of n symbols takes O(n log n). */
typedef struct { int rank, pos; } heap_ent;

static int heap_less(heap_ent a, heap_ent b) { return a.rank != b.rank ? a.rank < b.rank : a.pos < b.pos; }

static void heap_push(heap_ent *h, int *n, heap_ent e) {
    int i = (*n)++;
    while (i > 0 && heap_less(e, h[(i - 1) / 2])) { h[i] = h[(i - 1) / 2]; i = (i - 1) / 2; }
    h[i] = e;
}

static heap_ent heap_pop(heap_ent *h, int *n) {
    const heap_ent top = h[0], last = h[--*n];
    int i = 0;
    for (;;) {
        int c = 2 * i + 1;
        if (c >= *n) break;
        if (c + 1 < *n && heap_less(h[c + 1], h[c])) c++;
        if (!heap_less(h[c], last)) break;
        h[i] = h[c];
        i = c;
    }
    h[i] = last;
    return top;
}

static int merge_word(const ml_tokenizer *t, int *sym, int n) {
    if (n < 2) return n;
    int *next = malloc(sizeof(int) * (size_t)n * 2);
    heap_ent *heap = malloc(sizeof(heap_ent) * (size_t)n * 3);
    if (!next || !heap) { free(next); free(heap); return -1; }
    int *prev = next + n, nh = 0, out;
    for (int i = 0; i < n; i++) { next[i] = i + 1 < n ? i + 1 : -1; prev[i] = i - 1; }
    for (int i = 0; i + 1 < n; i++) {
        const int r = find_merge(t, sym[i], sym[i + 1], &out);
        if (r >= 0) heap_push(heap, &nh, (heap_ent){ r, i });
    }
    while (nh > 0) {
        const heap_ent e = heap_pop(heap, &nh);
        const int i = e.pos, j = next[i];
        /* stale entry: a neighbour has merged since it was pushed */
        if (sym[i] < 0 || j < 0 || find_merge(t, sym[i], sym[j], &out) != e.rank) continue;
        sym[i] = out;
        sym[j] = -1;
        next[i] = next[j];
        if (next[j] >= 0) prev[next[j]] = i;
        if (prev[i] >= 0) {
            const int r = find_merge(t, sym[prev[i]], sym[i], &out);
            if (r >= 0) heap_push(heap, &nh, (heap_ent){ r, prev[i] });
        }
        if (next[i] >= 0) {
            const int r = find_merge(t, sym[i], sym[next[i]], &out);
            if (r >= 0) heap_push(heap, &nh, (heap_ent){ r, i });
        }
    }
    int m = 0;
    for (int i = 0; i >= 0; i = next[i]) sym[m++] = sym[i];
    free(next);
    free(heap);
    return m;
}

typedef struct { int *out, n, max; } emit_buf;

static int emit(emit_buf *e, const int *ids, int n) {
    if (e->n + n > e->max) return -1;
    memcpy(e->out + e->n, ids, sizeof(int) * (size_t)n);
    e->n += n;
    return 0;
}

/* SentencePiece: prepend U+2581 (unless !prepend), turn spaces into U+2581,
 * start from one symbol per character (byte-fallback tokens for characters
 * not in the vocabulary), merge. sym needs room for 3 * (n + 1) entries. */
static int encode_spm(const ml_tokenizer *t, const unsigned char *s, int n, int prepend, int *sym, unsigned char *buf, emit_buf *e) {
    static const unsigned char sp[3] = { 0xE2, 0x96, 0x81 };
    int m = 0;
    if (prepend) { memcpy(buf, sp, 3); m = 3; }
    for (int i = 0; i < n; i++) {
        if (s[i] == ' ') { memcpy(buf + m, sp, 3); m += 3; }
        else buf[m++] = s[i];
    }
    int ns = 0;
    for (int i = 0; i < m;) {
        int len;
        ml_utf8_decode(buf + i, m - i, &len);
        const int id = find_piece(t, buf + i, len);
        if (id >= 0) sym[ns++] = id;
        else
            for (int j = 0; j < len; j++) sym[ns++] = t->byte_id[buf[i + j]] >= 0 ? t->byte_id[buf[i + j]] : (t->unk >= 0 ? t->unk : 0);
        i += len;
    }
    const int k = merge_word(t, sym, ns);
    return k < 0 ? -1 : emit(e, sym, k);
}

/* The Llama 3 pre-tokenizer regex, matched by hand on code points:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3}
 *   | ' '?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 * Returns the end of the chunk that starts at code point k. */
#define IS_L(i) ml_uni_is_letter(cp[i])
#define IS_N(i) ml_uni_is_number(cp[i])
#define IS_S(i) ml_uni_is_space(cp[i])
#define IS_NL(i) (cp[i] == '\r' || cp[i] == '\n')
#define IS_OTHER(i) (!IS_S(i) && !IS_L(i) && !IS_N(i))
static int chunk_end(const uint32_t *cp, int m, int k) {
    if (cp[k] == '\'' && k + 1 < m) {
        const uint32_t a = cp[k + 1] | 0x20, b = k + 2 < m ? cp[k + 2] | 0x20 : 0;
        if (cp[k + 1] < 0x80 && (a == 's' || a == 't' || a == 'm' || a == 'd')) return k + 2;
        if (cp[k + 1] < 0x80 && k + 2 < m && cp[k + 2] < 0x80 &&
            ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')))
            return k + 3;
    }
    int j = k;
    if (!IS_L(k) && !IS_N(k) && !IS_NL(k) && k + 1 < m && IS_L(k + 1)) j = k + 1;
    if (IS_L(j)) {
        while (j < m && IS_L(j)) j++;
        return j;
    }
    if (IS_N(k)) {
        for (j = k; j < m && j < k + 3 && IS_N(j); j++) {}
        return j;
    }
    j = k;
    if (cp[k] == ' ' && k + 1 < m && IS_OTHER(k + 1)) j = k + 1;
    if (IS_OTHER(j)) {
        while (j < m && IS_OTHER(j)) j++;
        while (j < m && IS_NL(j)) j++;
        return j;
    }
    /* white space: up to the run's last newline; else all of it at the end of
     * the text, or all but the last character before a non-space */
    int last_nl = -1;
    for (j = k; j < m && IS_S(j); j++)
        if (IS_NL(j)) last_nl = j;
    if (last_nl >= 0) return last_nl + 1;
    if (j < m && j - k >= 2) return j - 1;
    return j > k ? j : k + 1;
}

/* Byte-level BPE on one pre-tokenized chunk. */
static int encode_chunk(const ml_tokenizer *t, const unsigned char *s, int n, int *sym, emit_buf *e) {
    if (t->flags & ML_TOKF_IGNORE_MERGES) {
        const int id = find_piece(t, s, n);
        if (id >= 0) return emit(e, &id, 1);
    }
    for (int i = 0; i < n; i++) {
        sym[i] = find_piece(t, s + i, 1);
        if (sym[i] < 0) sym[i] = t->unk >= 0 ? t->unk : 0;
    }
    const int k = merge_word(t, sym, n);
    return k < 0 ? -1 : emit(e, sym, k);
}

static int encode_byte_bpe(const ml_tokenizer *t, const unsigned char *s, int n, int *sym, uint32_t *cp, int *off, emit_buf *e) {
    int m = 0;
    for (int i = 0; i < n; m++) {
        int len;
        cp[m] = ml_utf8_decode(s + i, n - i, &len);
        off[m] = i;
        i += len;
    }
    off[m] = n;
    for (int k = 0; k < m;) {
        const int j = chunk_end(cp, m, k);
        if (encode_chunk(t, s + off[k], off[j] - off[k], sym, e) < 0) return -1;
        k = j;
    }
    return 0;
}

/* Earliest added token in s[0, n): its position, or n; *id gets the token. */
static int find_added(const ml_tokenizer *t, const unsigned char *s, int n, int *id) {
    for (int i = 0; i < n; i++)
        for (int a = 0; a < t->n_added; a++) {
            const int k = t->added[a], len = t->piece_len[k];
            if (len <= n - i && (unsigned char)t->vocab[k][0] == s[i] && !memcmp(t->vocab[k], s + i, (size_t)len)) {
                *id = k;
                return i;
            }
        }
    return n;
}

int ml_bpe_encode(const ml_tokenizer *t, const char *text, int bos, int eos, int *out, int max) {
    const unsigned char *s = (const unsigned char *)text;
    const int n = (int)strlen(text);
    emit_buf e = { out, 0, max };
    int *sym = malloc(sizeof(int) * (3 * (size_t)n + 4));
    unsigned char *buf = malloc(3 * (size_t)n + 4);
    uint32_t *cp = malloc(sizeof(uint32_t) * ((size_t)n + 1));
    int *off = malloc(sizeof(int) * ((size_t)n + 1));
    int rc = sym && buf && cp && off ? 0 : -1;
    if (rc == 0 && bos && t->bos >= 0 && (t->flags & ML_TOKF_ADD_BOS)) rc = emit(&e, &t->bos, 1);
    /* added tokens are matched first; the text between them is encoded */
    for (int i = 0; rc == 0 && i < n;) {
        int id = -1;
        const int a = i + find_added(t, s + i, n - i, &id);
        if (a > i)
            rc = t->kind == ML_TOK_SPM ? encode_spm(t, s + i, a - i, !(t->flags & ML_TOKF_PREPEND_FIRST) || i == 0, sym, buf, &e)
                                       : encode_byte_bpe(t, s + i, a - i, sym, cp, off, &e);
        if (rc == 0 && a < n) rc = emit(&e, &id, 1);
        i = a < n ? a + t->piece_len[id] : n;
    }
    if (rc == 0 && eos && t->eos >= 0) rc = emit(&e, &t->eos, 1);
    free(sym); free(buf); free(cp); free(off);
    return rc == 0 ? e.n : -1;
}
