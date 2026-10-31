#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config/ml_config.h"
#include "core/pack.h"
#include "model/pack_model.h"

#define RUP(n) (((size_t)(n) + ML_ALIGN - 1) & ~(size_t)(ML_ALIGN - 1))

int ml_pack_group(void) { return ML_PACK_G; }

#if ML_PACK_G

typedef struct {
    ml_qtensor **mats;
    int n;
} pack_job;

static void pack_task(void *p, int tid, int nt) {
    pack_job *j = p;
    for (int i = 0; i < j->n; i++) {
        ml_qtensor *W = j->mats[i];
        const int ng = (W->rows + ML_PACK_G - 1) / ML_PACK_G;
        ml_pack_groups(W, (uint8_t *)W->pq, (void *)W->pd, ML_PACK_G, ML_PACK_Q8_OFFSET, ML_PACK_Q4_XOR, ML_PACK_D16,
                       (int)((long long)ng * tid / nt), (int)((long long)ng * (tid + 1) / nt));
    }
}

/* The matrices to pack, in a fixed order: every layer's, then the classifier
 * (Q8 ones only where the build has Q8 packed kernels). */
static ml_status collect(ml_model *m, pack_job *job) {
    job->mats = malloc(sizeof(ml_qtensor *) * ((size_t)m->cfg.n_layers * 7 + 1));
    job->n = 0;
    if (!job->mats) return ML_E_NOMEM;
#define ADD(W) do { if ((W)->type == ML_Q4_0 || ML_PACK_Q8) job->mats[job->n++] = (W); } while (0)
    for (int l = 0; l < m->cfg.n_layers; l++) {
        ml_layer *L = &m->layers[l];
        ml_qtensor *ws[7] = { &L->wq, &L->wk, &L->wv, &L->wo, &L->w1, &L->w2, &L->w3 };
        for (int k = 0; k < 7; k++) ADD(ws[k]);
    }
    ADD(&m->wcls);
#undef ADD
    return ML_OK;
}

/* Points the matrices at consecutive packed arrays from base; returns the
 * total size. base == NULL only measures. */
static size_t place(pack_job *job, const uint8_t *base) {
    size_t off = 0;
    for (int i = 0; i < job->n; i++) {
        ml_qtensor *W = job->mats[i];
        if (base) W->pq = base + off;
        off += RUP(ml_pack_quant_bytes(W, ML_PACK_G));
        if (base) W->pd = (const float *)(const void *)(base + off);
        if (base) W->pd16 = (uint8_t)ml_pack_f16(W, ML_PACK_D16);
        off += RUP(ml_pack_scale_bytes(W, ML_PACK_G, ML_PACK_D16));
    }
    return off;
}

/* The originals are never read again, except the embedding table, which
 * the classifier may share. */
static void release_originals(ml_model *m, const pack_job *job) {
    for (int i = 0; i < job->n; i++) {
        const ml_qtensor *W = job->mats[i];
        if (W->q == m->tok_emb.q) continue;
        const size_t nq = W->type == ML_Q4_0 ? (size_t)W->rows * W->cols / 2 : (size_t)W->rows * W->cols;
        ml_file_release(&m->file, W->q, nq);
        if (W->dh) ml_file_release(&m->file, W->dh, (size_t)W->rows * (W->cols / QK) * sizeof(uint16_t));
        else ml_file_release(&m->file, W->d, (size_t)W->rows * (W->cols / QK) * sizeof(float));
    }
    (void)ml_model_trim_scales(m);   /* on failure the full float copy stays */
}

static ml_status pack_now(ml_model *m, ml_pool *pool, pack_job *job) {
    uint8_t *mem = ml_aligned_alloc(place(job, NULL));
    if (!mem) return ML_E_NOMEM;
    place(job, mem);
    ml_pool_run(pool, pack_task, job);
    m->pack_mem = mem;
    m->packed = 1;
    release_originals(m, job);
    return ML_OK;
}

ml_status ml_model_pack(ml_model *m, ml_pool *pool) {
    if (m->packed) return ML_OK;
    pack_job job;
    ml_status s = collect(m, &job);
    if (s == ML_OK && job.n > 0) s = pack_now(m, pool, &job);
    free(job.mats);
    return s;
}

/* Pack cache file: a 512-byte header, then the packed arrays exactly as
 * place() lays them out. The header ties it to this layout and this model:
 * its size, its header, and a hash of 64 bytes sampled at 16 offsets. */
#define CACHE_HDR 512
typedef struct {
    char magic[4];               /* "MLPK" */
    uint32_t version, group, q8_offset, pack_q8, q4_xor, d16;
    uint64_t total, model_size, model_hash;
    uint8_t model_header[ML_MLB_HEADER_SIZE];
} cache_header;

static uint64_t model_hash(const ml_file *f) {
    uint64_t h = 1469598103934665603ull;
    for (int k = 0; k < 16; k++) {
        const size_t at = f->size > 64 ? (f->size - 64) / 15 * (size_t)k : 0;
        for (size_t i = 0; i < 64 && at + i < f->size; i++) h = (h ^ f->data[at + i]) * 1099511628211ull;
    }
    return h;
}

static void make_header(cache_header *h, const ml_model *m, size_t total) {
    memset(h, 0, sizeof *h);
    memcpy(h->magic, "MLPK", 4);
    h->version = 1;
    h->group = ML_PACK_G;
    h->q8_offset = ML_PACK_Q8_OFFSET;
    h->pack_q8 = ML_PACK_Q8;
    h->q4_xor = ML_PACK_Q4_XOR;
    h->d16 = ML_PACK_D16;
    h->total = total;
    h->model_size = m->file.size;
    h->model_hash = model_hash(&m->file);
    memcpy(h->model_header, m->file.data, ML_MLB_HEADER_SIZE);
}

ml_status ml_model_pack_cached(ml_model *m, ml_pool *pool, const char *path, const char **how) {
    *how = "none";
    if (m->packed) return ML_OK;
    pack_job job;
    ml_status s = collect(m, &job);
    if (s != ML_OK || job.n == 0) { free(job.mats); return s; }
    const size_t total = place(&job, NULL);
    cache_header want;
    make_header(&want, m, total);

    ml_file f;
    if (ml_file_map(&f, path) == ML_OK) {
        if (f.size == CACHE_HDR + total && !memcmp(f.data, &want, sizeof want)) {
            place(&job, f.data + CACHE_HDR);
            m->pack_file = f;
            m->packed = 1;
            release_originals(m, &job);
            free(job.mats);
            *how = "loaded";
            return ML_OK;
        }
        ml_file_unmap(&f);   /* stale or for another build: repack and rewrite */
    }

    s = pack_now(m, pool, &job);
    free(job.mats);
    if (s != ML_OK) return s;
    /* write to a temporary name, then rename, so readers never see half a file */
    char tmp[4096];
    if ((size_t)snprintf(tmp, sizeof tmp, "%s.tmp", path) >= sizeof tmp) { *how = "not written"; return ML_OK; }
    FILE *out = fopen(tmp, "wb");
    uint8_t hdr[CACHE_HDR] = { 0 };
    memcpy(hdr, &want, sizeof want);
    int ok = out && fwrite(hdr, 1, CACHE_HDR, out) == CACHE_HDR && fwrite(m->pack_mem, 1, total, out) == total;
    if (out) ok = (fclose(out) == 0) && ok;
    remove(path);
    ok = ok && rename(tmp, path) == 0;
    if (!ok) remove(tmp);
    *how = ok ? "written" : "not written";
    return ML_OK;
}

#else

ml_status ml_model_pack(ml_model *m, ml_pool *pool) {
    (void)m; (void)pool;
    return ML_OK;
}

ml_status ml_model_pack_cached(ml_model *m, ml_pool *pool, const char *path, const char **how) {
    (void)m; (void)pool; (void)path;
    *how = "none";
    return ML_OK;
}

#endif
