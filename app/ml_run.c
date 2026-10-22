/* ml_run: load a .mlb model and generate tokens. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "config/ml_config.h"
#include "model/sampler.h"
#include "model/tokenizer.h"
#include "model/transformer.h"

static int usage(const char *msg) {
    if (msg) fprintf(stderr, "error: %s\n", msg);
    fprintf(stderr,
        "usage: ml_run MODEL.mlb [options]\n"
        "  -z FILE          tokenizer.bin (needed for -i and text output)\n"
        "  -i TEXT          prompt text\n"
        "  --tokens LIST    prompt as comma-separated token ids\n"
        "  -n N             positions to run, prompt included (default 64, max: model max_seq)\n"
        "  -T F             temperature, 0 = greedy (default 0)\n"
        "  -p F             top-p (default 0.9)\n"
        "  -s N             random seed (default 1)\n"
        "  -t N             threads (default 1)\n"
        "  --dump-logits F  write float32 logits for every --tokens position, then exit\n"
        "build ISA: %s\n", ML_ISA_NAME);
    return 2;
}

static double now_s(void) {
#ifdef _WIN32
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* Parses "1,2,3" into out; returns count, or -1 on a bad entry or overflow. */
static int parse_tokens(const char *s, int *out, int max) {
    int n = 0;
    while (*s) {
        char *end;
        long v = strtol(s, &end, 10);
        if (end == s || n >= max) return -1;
        out[n++] = (int)v;
        s = *end == ',' ? end + 1 : end;
        if (*end && *end != ',') return -1;
    }
    return n;
}

int ml_main(int argc, char **argv) {
    const char *model_path = NULL, *tok_path = NULL, *prompt = NULL, *tokens_arg = NULL, *dump = NULL;
    int steps = 64, nthreads = 1;
    float temp = 0.0f, topp = 0.9f;
    unsigned long long seed = 1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] != '-') { if (model_path) return usage("more than one model given"); model_path = a; continue; }
        if (i + 1 >= argc) return usage("option needs a value");
        const char *v = argv[++i];
        if (!strcmp(a, "-z")) tok_path = v;
        else if (!strcmp(a, "-i")) prompt = v;
        else if (!strcmp(a, "--tokens")) tokens_arg = v;
        else if (!strcmp(a, "-n")) steps = atoi(v);
        else if (!strcmp(a, "-T")) temp = (float)atof(v);
        else if (!strcmp(a, "-p")) topp = (float)atof(v);
        else if (!strcmp(a, "-s")) seed = strtoull(v, NULL, 10);
        else if (!strcmp(a, "-t")) nthreads = atoi(v);
        else if (!strcmp(a, "--dump-logits")) dump = v;
        else return usage("unknown option");
    }
    if (!model_path) return usage(NULL);
    if (prompt && !tok_path) return usage("-i needs -z tokenizer");
    if (prompt && tokens_arg) return usage("give -i or --tokens, not both");
    if (dump && !tokens_arg) return usage("--dump-logits needs --tokens");
    if (steps < 1 || nthreads < 1 || temp < 0.0f) return usage("bad numeric option");

    ml_model m;
    ml_status st = ml_model_load(&m, model_path);
    if (st != ML_OK) { fprintf(stderr, "error: %s: %s\n", model_path, ml_status_str(st)); return 1; }
    const ml_config *c = &m.cfg;
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
        np = parse_tokens(tokens_arg, ptoks, maxp);
        if (np < 0 || np > c->max_seq) { rc = usage("--tokens: bad list or longer than max_seq"); goto out_tok; }
        for (int i = 0; i < np; i++)
            if (ptoks[i] < 0 || ptoks[i] >= c->vocab) { rc = usage("--tokens: id out of range"); goto out_tok; }
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

    if (dump) {
        FILE *f = fopen(dump, "wb");
        if (!f) { fprintf(stderr, "error: cannot write %s\n", dump); rc = 1; goto out_state; }
        for (int p = 0; p < np; p++) fwrite(ml_forward(&s, ptoks[p], p), sizeof(float), (size_t)c->vocab, f);
        fclose(f);
        goto out_state;
    }

    ml_sampler smp;
    if (ml_sampler_init(&smp, c->vocab, temp, topp, seed) != ML_OK) { rc = 1; goto out_state; }
    if (steps > c->max_seq) steps = c->max_seq;
    const double t0 = now_s();
    int token = ptoks[0], pos = 0;
    while (pos < steps) {
        float *logits = ml_forward(&s, token, pos);
        const int next = pos + 1 < np ? ptoks[pos + 1] : ml_sample(&smp, logits, c->vocab);
        pos++;
        if (pos >= np && (next == ML_TOK_BOS || next == ML_TOK_EOS)) break;
        if (have_tok) fputs(ml_decode(&tok, token, next), stdout);
        else printf("%d ", next);
        fflush(stdout);
        token = next;
    }
    putchar('\n');
    const double dt = now_s() - t0;
    fprintf(stderr, "positions: %d, tok/s: %.2f, threads: %d, isa: %s\n", pos, dt > 0 ? pos / dt : 0.0,
            ml_pool_size(s.pool), ML_ISA_NAME);
    ml_sampler_free(&smp);
out_state:
    ml_state_free(&s);
out_tok:
    free(ptoks);
    if (have_tok) ml_tokenizer_free(&tok);
    ml_model_free(&m);
    return rc;
}
