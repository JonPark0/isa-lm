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
    "  -z FILE          tokenizer: llama2.c tokenizer.bin, or .mlt from tools/convert_tokenizer.py\n"
    "                   (needed for -i and text output)\n"
    "  -i TEXT          prompt text\n"
    "  --tokens LIST    prompt as comma-separated token ids\n"
    "  -n N             positions to run, prompt included (default 64, max: model max_seq)\n"
    "  -T F             temperature, 0 = greedy (default 0)\n"
    "  -p F             top-p in [0, 1] (default 0.9)\n"
    "  -s N             random seed (default 1)\n"
    "  -t N             threads (default 1)\n"
    "  --eos ID         stop when ID is generated (default: the tokenizer's BOS/EOS with -z,\n"
    "                   otherwise run all -n positions)\n"
    "  --dump-logits F  write float32 logits for every --tokens position, then exit\n"
    "  --batch N        prompt tokens per batched prefill step, 1..32 (default 32; 1 = token by token)\n"
    "  --no-pack        keep weights row-major (no repacking at load: less memory, slower)\n"
    "  --kv TYPE        KV cache type: f32 (default) or q8 (3.8x smaller, faster on long contexts)\n"
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
    int profile = 0, no_pack = 0, kv_type = ML_F32;
    int steps = 64, nthreads = 1, eos = -1, batch = ML_BATCH_MAX;
    float temp = 0.0f, topp = 0.9f;
    unsigned long long seed = 1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            printf("%sbuild ISA: %s\npack group: %d rows\n", USAGE, ML_ISA_NAME, ml_pack_group());
            return 0;
        }
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
        else if (!strcmp(a, "--batch")) ok = parse_int(v, 1, ML_BATCH_MAX, &batch);
        else if (!strcmp(a, "--dump-logits")) dump = v;
        else if (!strcmp(a, "--kv")) {
            if (!strcmp(v, "f32")) kv_type = ML_F32;
            else if (!strcmp(v, "q8")) kv_type = ML_Q8_0;
            else ok = 0;
        }
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
    /* at most one token per byte, plus BOS and SentencePiece's prefix (3 bytes) */
    const size_t pcap = prompt ? 3 * strlen(prompt) + 8 : (size_t)maxp;
    int *ptoks = malloc(sizeof(int) * pcap);
    int np = 0, rc = 0;
    if (tokens_arg) {
        np = parse_tokens(tokens_arg, ptoks, maxp, c->vocab);
        if (np < 0 || np > c->max_seq) { rc = usage("--tokens: bad id, id out of range, or longer than max_seq"); goto out_tok; }
    } else if (prompt) {
        np = ml_encode(&tok, prompt, 1, 0, ptoks, (int)pcap);
        if (np > c->max_seq) np = c->max_seq;
    }
    const int bos_id = have_tok && ml_tokenizer_bos(&tok) >= 0 ? ml_tokenizer_bos(&tok) : ML_TOK_BOS;
    if (np <= 0) { ptoks[0] = bos_id; np = 1; }

    ml_state s;
    if ((st = ml_state_init(&s, &m, nthreads, kv_type)) != ML_OK) {
        fprintf(stderr, "error: %s\n", ml_status_str(st));
        rc = 1;
        goto out_tok;
    }

    if (!no_pack && (st = ml_model_pack(&m, s.pool)) != ML_OK) {
        fprintf(stderr, "error: packing weights: %s\n", ml_status_str(st));
        rc = 1;
        goto out_state;
    }

    s.batch = batch;
    s.profile = profile;
    if (dump) {
        FILE *f = fopen(dump, "wb");
        float *all = malloc(sizeof(float) * (size_t)np * c->vocab);
        if (!f || !all) { fprintf(stderr, "error: cannot write %s\n", dump); if (f) fclose(f); free(all); rc = 1; goto out_state; }
        ml_prefill(&s, ptoks, np, 0, all);   /* ptoks were range-checked against vocab and max_seq above */
        fwrite(all, sizeof(float), (size_t)np * c->vocab, f);
        fclose(f);
        free(all);
        goto out_state;
    }

    ml_sampler smp;
    if (ml_sampler_init(&smp, c->vocab, temp, topp, seed) != ML_OK) { rc = 1; goto out_state; }
    if (steps > c->max_seq) steps = c->max_seq;
    /* Prompt: all but the last token go through batched prefill (no logits
     * needed); generation then continues token by token. */
    int token = ptoks[0], pos = 0;
    const int npre = np - 1 < steps - 1 ? np - 1 : steps - 1;
    double tp = 0.0;
    if (npre > 0) {
        const double t = ml_now();
        ml_prefill(&s, ptoks, npre, 0, NULL);
        tp = ml_now() - t;
        for (pos = 0; pos < npre; pos++) {
            if (have_tok) fputs(ml_decode(&tok, ptoks[pos], ptoks[pos + 1]), stdout);
            else printf("%d ", ptoks[pos + 1]);
        }
        token = ptoks[npre];
    }
    const double t0 = ml_now();
    while (pos < steps) {
        float *logits = ml_forward(&s, token, pos);
        if (!logits) { fprintf(stderr, "error: token %d at position %d out of range\n", token, pos); rc = 1; break; }
        const int next = pos + 1 < np ? ptoks[pos + 1] : ml_sample(&smp, logits, c->vocab);
        pos++;
        /* With -z the tokenizer's BOS/EOS stop generation (llama2.c: ids 1/2).
         * Without it the ids are unknown (Llama 3 uses neither 1 nor 2), so
         * only an explicit --eos stops early. */
        const int stop = eos >= 0 ? next == eos
                                  : (have_tok && (next == ml_tokenizer_bos(&tok) || next == ml_tokenizer_eos(&tok)));
        if (pos >= np && stop) break;
        if (have_tok) fputs(ml_decode(&tok, token, next), stdout);
        else printf("%d ", next);
        fflush(stdout);
        token = next;
    }
    putchar('\n');
    const double dt = ml_now() - t0 + tp;
    if (npre > 0)
        fprintf(stderr, "prefill: %d tokens in %.1f ms (%.1f tok/s)\n", npre, tp * 1e3, tp > 0 ? npre / tp : 0.0);
    fprintf(stderr, "positions: %d, tok/s: %.2f, threads: %d, isa: %s, packed: %d, kv: %s\n", pos, dt > 0 ? pos / dt : 0.0,
            ml_pool_size(s.pool), ML_ISA_NAME, m.pack_mem ? ml_pack_group() : 0, kv_type == ML_Q8_0 ? "q8" : "f32");
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
