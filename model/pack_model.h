/* Repacks a loaded model's weight matrices for the build's packed kernels. */
#ifndef ML_PACK_MODEL_H
#define ML_PACK_MODEL_H
#include "core/threads.h"
#include "model/mlb.h"

/* Packs every matrix except the embedding table (which stays row-major for
 * lookups) and releases the original mapped pages. Returns ML_OK without
 * doing anything when the build has no packed kernels. */
ml_status ml_model_pack(ml_model *m, ml_pool *pool);
/* The same, through a cache file: if `path` holds the packed weights of this
 * model for this build's layout, they are mapped from it instead of packed;
 * otherwise the model is packed and the cache written (a write failure is
 * only reported). *how gets "loaded", "written", "not written" or "none". */
ml_status ml_model_pack_cached(ml_model *m, ml_pool *pool, const char *path, const char **how);
/* Rows per packed group in this build (0 = no packed kernels). */
int ml_pack_group(void);

#endif
