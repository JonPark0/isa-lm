#include <stdio.h>
#include <stdlib.h>
#include "core/file.h"

#if defined(ML_NO_MMAP)

ml_status ml_file_map(ml_file *f, const char *path) {
    f->data = NULL; f->size = 0; f->handle = NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) return ML_E_IO;
    long n = (fseek(fp, 0, SEEK_END) == 0) ? ftell(fp) : -1;
    if (n < 0 || fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return ML_E_IO; }
    uint8_t *buf = ml_aligned_alloc((size_t)n);
    if (!buf) { fclose(fp); return ML_E_NOMEM; }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) { ml_aligned_free(buf); fclose(fp); return ML_E_IO; }
    fclose(fp);
    f->data = buf; f->size = (size_t)n; f->handle = buf;
    return ML_OK;
}

void ml_file_unmap(ml_file *f) {
    ml_aligned_free(f->handle);
    f->data = NULL; f->size = 0; f->handle = NULL;
}

void ml_file_release(ml_file *f, const void *p, size_t n) { (void)f; (void)p; (void)n; }

#elif defined(_WIN32)

#include <windows.h>

ml_status ml_file_map(ml_file *f, const char *path) {
    f->data = NULL; f->size = 0; f->handle = NULL;
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return ML_E_IO;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(h, &sz)) { CloseHandle(h); return ML_E_IO; }
    if (sz.QuadPart == 0) { CloseHandle(h); return ML_E_FORMAT; }
    HANDLE m = CreateFileMappingA(h, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(h);
    if (!m) return ML_E_IO;
    const void *p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!p) { CloseHandle(m); return ML_E_IO; }
    f->data = p; f->size = (size_t)sz.QuadPart; f->handle = m;
    return ML_OK;
}

void ml_file_unmap(ml_file *f) {
    if (f->data) UnmapViewOfFile(f->data);
    if (f->handle) CloseHandle(f->handle);
    f->data = NULL; f->size = 0; f->handle = NULL;
}

void ml_file_release(ml_file *f, const void *p, size_t n) { (void)f; (void)p; (void)n; }

#else

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

ml_status ml_file_map(ml_file *f, const char *path) {
    f->data = NULL; f->size = 0; f->handle = NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return ML_E_IO;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return ML_E_IO; }
    if (st.st_size == 0) { close(fd); return ML_E_FORMAT; }   /* nothing to map */
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return ML_E_IO;
    f->data = p; f->size = (size_t)st.st_size;
    return ML_OK;
}

void ml_file_unmap(ml_file *f) {
    if (f->data) munmap((void *)f->data, f->size);
    f->data = NULL; f->size = 0;
}

void ml_file_release(ml_file *f, const void *p, size_t n) {
    const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
    uintptr_t a = ((uintptr_t)p + page - 1) & ~(page - 1), e = ((uintptr_t)p + n) & ~(page - 1);
    const uintptr_t lo = (uintptr_t)f->data, hi = lo + f->size;
    if (a < lo) a = lo;
    if (e > hi) e = hi;
    if (e > a) madvise((void *)a, e - a, MADV_DONTNEED);
}

#endif
