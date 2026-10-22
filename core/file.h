/* Read-only whole-file mapping: mmap (POSIX), CreateFileMapping (Windows), or
 * malloc + fread when built with -DML_NO_MMAP. */
#ifndef ML_FILE_H
#define ML_FILE_H
#include "core/ml.h"

typedef struct {
    const uint8_t *data;   /* 64 B aligned */
    size_t size;
    void *handle;
} ml_file;

ml_status ml_file_map(ml_file *f, const char *path);
void ml_file_unmap(ml_file *f);
/* Hints that [p, p+n) inside the mapping will not be read again (frees the
 * page cache pages it occupies in this process). No-op without mmap. */
void ml_file_release(ml_file *f, const void *p, size_t n);

#endif
