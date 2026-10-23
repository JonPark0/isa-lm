#include <stdlib.h>
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
        ml_pack_groups(W, (uint8_t *)W->pq, (float *)W->pd, ML_PACK_G, ML_PACK_Q8_OFFSET,
                       (int)((long long)ng * tid / nt), (int)((long long)ng * (tid + 1) / nt));
    }
}

ml_status ml_model_pack(ml_model *m, ml_pool *pool) {
    if (m->pack_mem) return ML_OK;
    pack_job job = { malloc(sizeof(ml_qtensor *) * ((size_t)m->cfg.n_layers * 7 + 1)), 0 };
    if (!job.mats) return ML_E_NOMEM;
    /* Q8 matrices stay row-major where the build has no Q8 packed kernels */
#define ADD(W) do { if ((W)->type == ML_Q4_0 || ML_PACK_Q8) job.mats[job.n++] = (W); } while (0)
    for (int l = 0; l < m->cfg.n_layers; l++) {
        ml_layer *L = &m->layers[l];
        ml_qtensor *ws[7] = { &L->wq, &L->wk, &L->wv, &L->wo, &L->w1, &L->w2, &L->w3 };
        for (int k = 0; k < 7; k++) ADD(ws[k]);
    }
    ADD(&m->wcls);
#undef ADD
    if (job.n == 0) { free(job.mats); return ML_OK; }

    size_t total = 0;
    for (int i = 0; i < job.n; i++)
        total += RUP(ml_pack_quant_bytes(job.mats[i], ML_PACK_G)) + RUP(ml_pack_scale_bytes(job.mats[i], ML_PACK_G));
    uint8_t *mem = ml_aligned_alloc(total);
    if (!mem) { free(job.mats); return ML_E_NOMEM; }
    size_t off = 0;
    for (int i = 0; i < job.n; i++) {
        ml_qtensor *W = job.mats[i];
        W->pq = mem + off;
        off += RUP(ml_pack_quant_bytes(W, ML_PACK_G));
        W->pd = (const float *)(mem + off);
        off += RUP(ml_pack_scale_bytes(W, ML_PACK_G));
    }
    ml_pool_run(pool, pack_task, &job);
    m->pack_mem = mem;

    /* The originals are never read again, except the embedding table, which
     * the classifier may share. */
    for (int i = 0; i < job.n; i++) {
        const ml_qtensor *W = job.mats[i];
        if (W->q == m->tok_emb.q) continue;
        const size_t nq = W->type == ML_Q4_0 ? (size_t)W->rows * W->cols / 2 : (size_t)W->rows * W->cols;
        ml_file_release(&m->file, W->q, nq);
        ml_file_release(&m->file, W->d, (size_t)W->rows * (W->cols / QK) * sizeof(float));
    }
    free(job.mats);
    return ML_OK;
}

#else

ml_status ml_model_pack(ml_model *m, ml_pool *pool) {
    (void)m; (void)pool;
    return ML_OK;
}

#endif
