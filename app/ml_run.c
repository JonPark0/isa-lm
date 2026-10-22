/* ml_run: load a .mlb model and generate tokens. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config/ml_config.h"
#include "model/pack_model.h"
#include "model/sampler.h"
#include "model/tokenizer.h"
#include "core/clock.h"
#include "model/transformer.h"

static const char USAGE[] =
    "usage: ml_run MODEL.mlb [options]\n"
    "  -z FILE          tokenizer.bin (needed for -i and text output)\n"
    "  -i TEXT          prompt text\n"
    "  --tokens LIST    prompt as comma-separated token ids\n"
    "  -n N             positions to run, prompt included (default 64, max: model max_seq)\n"
    "  -T F             temperature, 0 = greedy (default 0)\n"
    "  -p F             top-p in [0, 1] (default 0.9)\n"
    "  -s N             random seed (default 1)\n"
    "  -t N             threads (default 1)\n"
    "  --eos ID         stop when ID is generated (default: BOS/EOS ids 1 and 2 with -z,\n"
    "                   otherwise run all -n positions)\n"
    "  --dump-logits F  write float32 logits for every --tokens position, then exit\n"
    "  --no-pack        keep weights row-major (no repacking at load: less memory, slower)\n"
    "  --profile        print time spent per stage (qkv, attention, wo, ffn, classifier, other)\n"
    "  -h, --help       show this help\n";

static int usage(const char *msg) {
    if (msg) fprintf(stderr, "error: %s\n", msg);
    fprintf(stderr, "%sbuild ISA: %s\n", USAGE, ML_ISA_NAME);
    return 2;
}

/* Strict number parsing: the whole string must be consumed and in range. */
static int parse_int(const char *s, long lo, long hi, int *out) {
    char *end;
    errno = 0;
    const long v = strtol(s, &end, 10);
    if (end == s || *end || errno == ERANGE || v < lo || v > hi) return 0;
    *out = (int)v;
    return 1;
}

static int parse_u64(const char *s, unsigned long long *out) {
    char *end;
    errno = 0;
    if (*s == '-' || *s == '+') return 0;
    const unsigned long long v = strtoull(s, &end, 10);
    if (end == s || *end || errno == ERANGE) return 0;
    *out = v;
    return 1;
}

static int parse_float(const char *s, float *out) {
    char *end;
    errno = 0;
    const float v = strtof(s, &end);
    if (end == s || *end || errno == ERANGE) return 0;
    *out = v;
    return 1;
}

/* Parses "1,2,3" into out; returns count, or -1 on an empty, non-numeric or
 * out-of-range entry, or more than max entries. */
static int parse_tokens(const char *s, int *out, int max, int vocab) {
    int n = 0;
    for (;;) {
        char *end;
        errno = 0;
        const long v = strtol(s, &end, 10);
        if (end == s || errno == ERANGE || v < 0 || v >= vocab || n >= max) return -1;
        out[n++] = (int)v;
        if (*end == '\0') return n;
        if (*end != ',') return -1;
        s = end + 1;
    }
}

int ml_main(int argc, char **argv) {
    const char *model_path = NULL, *tok_path = NULL, *prompt = NULL, *tokens_arg = NULL, *dump = NULL;
    int profile = 0, no_pack = 0;
    int steps = 64, nthreads = 1, eos = -1;
    float temp = 0.0f, topp = 0.9f;
    unsigned long long seed = 1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { printf("%sbuild ISA: %s\n", USAGE, ML_ISA_NAME); return 0; }
        if (!strcmp(a, "--profile")) { profile = 1; continue; }
        if (!strcmp(a, "--no-pack")) { no_pack = 1; continue; }
        if (a[0] != '-') { if (model_path) return usage("more than one model given"); model_path = a; continue; }
        if (i + 1 >= argc) return usage("option needs a value");
        const char *v = argv[++i];
        int ok = 1;
        if (!strcmp(a, "-z")) tok_path = v;
        else if (!strcmp(a, "-i")) prompt = v;
        else if (!strcmp(a, "--tokens")) tokens_arg = v;
        else if (!strcmp(a, "-n")) ok = parse_int(v, 1, 1 << 24, &steps);
        else if (!strcmp(a, "-T")) ok = parse_float(v, &temp);
        else if (!strcmp(a, "-p")) ok = parse_float(v, &topp);
        else if (!strcmp(a, "-s")) ok = parse_u64(v, &seed);
        else if (!strcmp(a, "-t")) ok = parse_int(v, 1, 64, &nthreads);
        else if (!strcmp(a, "--eos")) ok = parse_int(v, 0, 1 << 24, &eos);
        else if (!strcmp(a, "--dump-logits")) dump = v;
        else return usage("unknown option");
        if (!ok) { fprintf(stderr, "error: bad value '%s' for %s\n", v, a); return usage(NULL); }
    }
    if (!model_path) return usage(NULL);
    if (prompt && !tok_path) return usage("-i needs -z tokenizer");
    if (prompt && tokens_arg) return usage("give -i or --tokens, not both");
    if (dump && !tokens_arg) return usage("--dump-logits needs --tokens");
    if (!(temp >= 0.0f && temp <= 1e30f)) return usage("-T must be a finite number >= 0");
    if (!(topp >= 0.0f && topp <= 1.0f)) return usage("-p must be in [0, 1]");

    ml_model m;
    ml_status st = ml_model_load(&m, model_path);
    if (st != ML_OK) { fprintf(stderr, "error: %s: %s\n", model_path, ml_status_str(st)); return 1; }
    const ml_config *c = &m.cfg;
    if (eos >= c->vocab) { ml_model_free(&m); return usage("--eos id out of range"); }
    ml_tokenizer tok;
    int have_tok = 0;
    if (tok_path) {
        st = ml_tokenizer_load(&tok, tok_path, c->vocab);
        if (st != ML_OK) { fprintf(stderr, "error: %s: %s\n", tok_path, ml_status_str(st)); ml_model_free(&m); return 1; }
        have_tok = 1;
    }

    const int maxp = c->max_seq + 1;
    int *ptoks = malloc(sizeof(int) * (size_t)(prompt ? strlen(prompt) + 3 : (size_t)maxp));
    int np = 0, rc = 0;
    if (tokens_arg) {
        np = parse_tokens(tokens_arg, ptoks, maxp, c->vocab);
        if (np < 0 || np > c->max_seq) { rc = usage("--tokens: bad id, id out of range, or longer than max_seq"); goto out_tok; }
    } else if (prompt) {
        np = ml_encode(&tok, prompt, 1, 0, ptoks, (int)strlen(prompt) + 3);
    }
    if (np <= 0) { ptoks[0] = ML_TOK_BOS; np = 1; }

    ml_state s;
    if ((st = ml_state_init(&s, &m, nthreads)) != ML_OK) {
        fprintf(stderr, "error: %s\n", ml_status_str(st));
        rc = 1;
        goto out_tok;
    }

    if (!no_pack && (st = ml_model_pack(&m, s.pool)) != ML_OK) {
        fprintf(stderr, "error: packing weights: %s\n", ml_status_str(st));
        rc = 1;
        goto out_state;
    }

    if (dump) {
        FILE *f = fopen(dump, "wb");
        if (!f) { fprintf(stderr, "error: cannot write %s\n", dump); rc = 1; goto out_state; }
        for (int p = 0; p < np; p++) fwrite(ml_forward(&s, ptoks[p], p), sizeof(float), (size_t)c->vocab, f);
        /* ptoks were range-checked against vocab and max_seq above */
        fclose(f);
        goto out_state;
    }

    s.profile = profile;
    ml_sampler smp;
    if (ml_sampler_init(&smp, c->vocab, temp, topp, seed) != ML_OK) { rc = 1; goto out_state; }
    if (steps > c->max_seq) steps = c->max_seq;
    const double t0 = ml_now();
    int token = ptoks[0], pos = 0;
    while (pos < steps) {
        float *logits = ml_forward(&s, token, pos);
        if (!logits) { fprintf(stderr, "error: token %d at position %d out of range\n", token, pos); rc = 1; break; }
        const int next = pos + 1 < np ? ptoks[pos + 1] : ml_sample(&smp, logits, c->vocab);
        pos++;
        /* llama2.c tokenizers use ids 1/2 as BOS/EOS; other vocabularies (e.g.
         * Llama 3) do not, so without -z only an explicit --eos stops early. */
        const int stop = eos >= 0 ? next == eos : (have_tok && (next == ML_TOK_BOS || next == ML_TOK_EOS));
        if (pos >= np && stop) break;
        if (have_tok) fputs(ml_decode(&tok, token, next), stdout);
        else printf("%d ", next);
        fflush(stdout);
        token = next;
    }
    putchar('\n');
    const double dt = ml_now() - t0;
    fprintf(stderr, "positions: %d, tok/s: %.2f, threads: %d, isa: %s, packed: %d\n", pos, dt > 0 ? pos / dt : 0.0,
            ml_pool_size(s.pool), ML_ISA_NAME, m.pack_mem ? ml_pack_group() : 0);
    if (profile) {
        double total = 0.0;
        for (int i = 0; i < ML_PROF_COUNT; i++) total += s.prof[i];
        fprintf(stderr, "profile:");
        for (int i = 0; i < ML_PROF_COUNT; i++)
            fprintf(stderr, " %s %.1f%%", ml_prof_names[i], total > 0 ? 100.0 * s.prof[i] / total : 0.0);
        fprintf(stderr, " (forward %.1f ms over %d positions)\n", total * 1e3, pos);
    }
    ml_sampler_free(&smp);
out_state:
    ml_state_free(&s);
out_tok:
    free(ptoks);
    if (have_tok) ml_tokenizer_free(&tok);
    ml_model_free(&m);
    return rc;
}
