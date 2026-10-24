/* ml_run: load a .mlb model and generate tokens. */
#include <errno.h>
#include <stdint.h>
#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config/ml_config.h"
#include "model/chat.h"
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
    "  --top-k N        sample only from the N most likely tokens (default 0: off)\n"
    "  --repeat-penalty F  > 1 discourages tokens among the last --repeat-last-n (default 1: off)\n"
    "  --repeat-last-n N   tokens the repetition penalty looks back on (default 64)\n"
    "  -s N             random seed (default 1)\n"
    "  -t N             threads (default 1)\n"
    "  --eos ID         stop when ID is generated (default: the tokenizer's BOS/EOS with -z,\n"
    "                   otherwise run all -n positions)\n"
    "  --dump-logits F  write float32 logits for every --tokens position, then exit\n"
    "  --batch N        prompt tokens per batched prefill step, 1..32 (default 32; 1 = token by token)\n"
    "  --no-pack        keep weights row-major (no repacking at load: less memory, slower)\n"
    "  --pack-cache     keep the packed weights in MODEL.pk<G> and map them from there next time\n"
    "  --kv TYPE        KV cache type: f32 (default) or q8 (3.8x smaller, faster on long contexts)\n"
    "  --profile        print time spent per stage (qkv, attention, wo, ffn, classifier, other)\n"
    "  --imatrix FILE   add the squares of every matrix input, per channel, to FILE (created if\n"
    "                   missing) for tools/convert.py --imatrix; run on text like the model's use\n"
    "  --chat TEMPLATE  chat with the model: llama3, zephyr (TinyLlama), llama2 (also Mistral),\n"
    "                   chatml, or qwen (ChatML with Qwen2.5's default system prompt);\n"
    "                   one turn per stdin line, or the single turn -i; -n caps each reply\n"
    "  --system TEXT    system prompt for --chat\n"
    "  -h, --help       show this help\n";

static int usage(const char *msg) {
    if (msg) fprintf(stderr, "error: %s\n", msg);
    fprintf(stderr, "%sisa-lm %s, build ISA: %s\n", USAGE, ML_VERSION, ML_ISA_NAME);
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

/* --imatrix file: 40-byte header "MLI1", version 1, n_layers, dim, hidden_dim,
 * 0 (u32 each), tokens and classifier tokens (u64), then the float64 sums in
 * ml_state.imat order. Adds this run's sums to the file's. */
static int imatrix_save(const char *path, const ml_state *s) {
    const ml_config *c = &s->m->cfg;
    const size_t n = ml_imatrix_size(c);
    uint32_t h[6] = { 0, 1, (uint32_t)c->n_layers, (uint32_t)c->dim, (uint32_t)c->hidden_dim, 0 };
    uint64_t cnt[2] = { s->imat_tokens, s->imat_cls_tokens };
    memcpy(h, "MLI1", 4);
    double *sum = malloc(sizeof(double) * n), *old = malloc(sizeof(double) * n);
    int ok = sum && old;
    FILE *f = ok ? fopen(path, "rb") : NULL;
    if (ok) memcpy(sum, s->imat, sizeof(double) * n);
    if (f) {
        uint32_t fh[6];
        uint64_t fc[2];
        ok = fread(fh, sizeof fh, 1, f) == 1 && !memcmp(fh, h, sizeof fh) &&
             fread(fc, sizeof fc, 1, f) == 1 && fread(old, sizeof(double), n, f) == n;
        fclose(f);
        if (!ok) fprintf(stderr, "error: %s is not an importance matrix for this model\n", path);
        if (ok) {
            for (size_t i = 0; i < n; i++) sum[i] += old[i];
            cnt[0] += fc[0];
            cnt[1] += fc[1];
        }
    }
    if (ok) {
        f = fopen(path, "wb");
        ok = f && fwrite(h, sizeof h, 1, f) == 1 && fwrite(cnt, sizeof cnt, 1, f) == 1 &&
             fwrite(sum, sizeof(double), n, f) == n;
        if (f && fclose(f) != 0) ok = 0;
        if (!ok) fprintf(stderr, "error: cannot write %s\n", path);
    }
    free(sum);
    free(old);
    return ok;
}

/* --chat: a conversation. Each turn renders the template, prefills it after
 * the previous reply in the same KV cache, and samples a reply up to the
 * template's stop token, the tokenizer's EOS, --eos, or max_reply tokens.
 * The token that ended a reply is fed at the start of the next turn. */
static int run_chat(ml_state *s, ml_tokenizer *tok, ml_sampler *smp, int tpl, const char *system,
                    const char *once, int max_reply, int eos_opt) {
    const ml_config *c = &s->m->cfg;
    const int stop_id = ml_tokenizer_find(tok, ml_chat_stop(tpl)), eos_id = ml_tokenizer_eos(tok);
    const int interactive = !once && isatty(fileno(stdin));
    int pos = 0, first = 1, pending[2] = { -1, -1 }, npend = 0, rc = 0;
    static char line[1 << 16];
    for (;;) {
        const char *user = once;
        if (once && !first) break;
        if (!once) {
            if (interactive) fputs("> ", stderr);
            if (!fgets(line, sizeof line, stdin)) break;
            line[strcspn(line, "\r\n")] = '\0';
            if (!line[0]) continue;
            user = line;
        }
        char *text = ml_chat_render(tpl, system, user, first);
        const size_t cap = text ? 3 * strlen(text) + 16 : 0;
        int *toks = text ? malloc(sizeof(int) * cap) : NULL;
        if (!toks) { free(text); rc = 1; break; }
        int n = 0;
        for (int i = 0; i < npend; i++) toks[n++] = pending[i];
        const int k = ml_encode(tok, text, first || ml_chat_bos_each_turn(tpl), 0, toks + n, (int)cap - n);
        free(text);
        if (k < 0) { free(toks); rc = 1; break; }
        n += k;
        if (pos + n + 1 > c->max_seq) {
            fprintf(stderr, "context full: %d of %d positions used\n", pos, c->max_seq);
            free(toks);
            break;
        }
        for (int i = 0; i < n; i++) ml_sampler_accept(smp, toks[i]);
        ml_prefill(s, toks, n - 1, pos, NULL);
        pos += n - 1;
        int token = toks[n - 1], prev = -1, reply = 0;
        free(toks);
        npend = 0;
        while (pos < c->max_seq) {
            float *logits = ml_forward(s, token, pos++);
            if (!logits) { rc = 1; break; }
            const int next = ml_sample(smp, logits, c->vocab);
            ml_sampler_accept(smp, next);
            if (next == stop_id || next == eos_id || next == eos_opt) { pending[npend++] = next; break; }
            fputs(ml_decode(tok, prev, next), stdout);
            fflush(stdout);
            prev = token = next;
            if (++reply >= max_reply) {   /* cut short: close the reply as the model would */
                pending[npend++] = next;
                if (stop_id >= 0) pending[npend++] = stop_id;
                break;
            }
        }
        putchar('\n');
        fflush(stdout);
        first = 0;
        if (rc || pos >= c->max_seq) break;
    }
    fprintf(stderr, "positions: %d of %d\n", pos, c->max_seq);
    return rc;
}

int ml_main(int argc, char **argv) {
    const char *model_path = NULL, *tok_path = NULL, *prompt = NULL, *tokens_arg = NULL, *dump = NULL;
    const char *chat = NULL, *system = NULL, *imat = NULL;
    int steps_given = 0;
    int profile = 0, no_pack = 0, pack_cache = 0, kv_type = ML_F32;
    int steps = 64, nthreads = 1, eos = -1, batch = ML_BATCH_MAX;
    float temp = 0.0f, topp = 0.9f, rep_pen = 1.0f;
    int topk = 0, rep_last = 64;
    unsigned long long seed = 1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            printf("%sisa-lm %s, build ISA: %s\npack group: %d rows\n", USAGE, ML_VERSION, ML_ISA_NAME, ml_pack_group());
            return 0;
        }
        if (!strcmp(a, "--profile")) { profile = 1; continue; }
        if (!strcmp(a, "--no-pack")) { no_pack = 1; continue; }
        if (!strcmp(a, "--pack-cache")) { pack_cache = 1; continue; }
        if (a[0] != '-') { if (model_path) return usage("more than one model given"); model_path = a; continue; }
        if (i + 1 >= argc) return usage("option needs a value");
        const char *v = argv[++i];
        int ok = 1;
        if (!strcmp(a, "-z")) tok_path = v;
        else if (!strcmp(a, "-i")) prompt = v;
        else if (!strcmp(a, "--tokens")) tokens_arg = v;
        else if (!strcmp(a, "-n")) ok = steps_given = parse_int(v, 1, 1 << 24, &steps);
        else if (!strcmp(a, "--chat")) ok = ml_chat_template(chat = v) >= 0;
        else if (!strcmp(a, "--system")) system = v;
        else if (!strcmp(a, "-T")) ok = parse_float(v, &temp);
        else if (!strcmp(a, "-p")) ok = parse_float(v, &topp);
        else if (!strcmp(a, "-s")) ok = parse_u64(v, &seed);
        else if (!strcmp(a, "--top-k")) ok = parse_int(v, 0, 1 << 24, &topk);
        else if (!strcmp(a, "--repeat-penalty")) ok = parse_float(v, &rep_pen) && rep_pen > 0.0f && rep_pen <= 1e3f;
        else if (!strcmp(a, "--repeat-last-n")) ok = parse_int(v, 0, 1 << 24, &rep_last);
        else if (!strcmp(a, "-t")) ok = parse_int(v, 1, 64, &nthreads);
        else if (!strcmp(a, "--eos")) ok = parse_int(v, 0, 1 << 24, &eos);
        else if (!strcmp(a, "--batch")) ok = parse_int(v, 1, ML_BATCH_MAX, &batch);
        else if (!strcmp(a, "--dump-logits")) dump = v;
        else if (!strcmp(a, "--imatrix")) imat = v;
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
    if (pack_cache && no_pack) return usage("give --pack-cache or --no-pack, not both");
    if (dump && !tokens_arg) return usage("--dump-logits needs --tokens");
    if (chat && !tok_path) return usage("--chat needs -z tokenizer");
    if (chat && (tokens_arg || dump)) return usage("--chat takes -i or stdin, not --tokens or --dump-logits");
    if (system && !chat) return usage("--system needs --chat");
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
    } else if (prompt && !chat) {
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

    if (!no_pack && pack_cache && ml_pack_group()) {
        char cache[4096];
        const char *how;
        snprintf(cache, sizeof cache, "%s.pk%d", model_path, ml_pack_group());
        st = ml_model_pack_cached(&m, s.pool, cache, &how);
        if (st == ML_OK) fprintf(stderr, "pack cache %s: %s\n", cache, how);
    } else if (!no_pack) {
        st = ml_model_pack(&m, s.pool);
    }
    if (st != ML_OK) {
        fprintf(stderr, "error: packing weights: %s\n", ml_status_str(st));
        rc = 1;
        goto out_state;
    }

    s.batch = batch;
    s.profile = profile;
    if (imat && ml_imatrix_enable(&s) != ML_OK) { fprintf(stderr, "error: out of memory\n"); rc = 1; goto out_state; }
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
    const ml_sampler_opts so = { temp, topp, topk, rep_pen, rep_last, seed };
    if (ml_sampler_init(&smp, c->vocab, &so) != ML_OK) { rc = 1; goto out_state; }
    if (chat) {
        rc = run_chat(&s, &tok, &smp, ml_chat_template(chat), system, prompt, steps_given ? steps : c->max_seq, eos);
        ml_sampler_free(&smp);
        goto out_state;
    }
    for (int i = 0; i < np; i++) ml_sampler_accept(&smp, ptoks[i]);
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
        const int sampled = pos + 1 >= np;
        const int next = sampled ? ml_sample(&smp, logits, c->vocab) : ptoks[pos + 1];
        if (sampled) ml_sampler_accept(&smp, next);
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
            ml_pool_size(s.pool), ML_ISA_NAME, m.packed ? ml_pack_group() : 0, kv_type == ML_Q8_0 ? "q8" : "f32");
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
    if (imat && s.imat && rc == 0 && !imatrix_save(imat, &s)) rc = 1;
    ml_state_free(&s);
out_tok:
    free(ptoks);
    if (have_tok) ml_tokenizer_free(&tok);
    ml_model_free(&m);
    return rc;
}
