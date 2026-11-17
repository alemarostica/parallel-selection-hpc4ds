#ifndef CORE_H
#define CORE_H
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum
{
    GEN_RANDOM,
    GEN_SORTED,
    GEN_REVERSE
} GenMode;

/* tiny hot helpers: kept inline here so every file can inline them */
static inline void swap32(int32_t *x, int32_t *y)
{
    int32_t t = *x;
    *x = *y;
    *y = t;
}

/* splitmix64 finalizer */
static inline uint64_t mix64(uint64_t z)
{
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* kernels.c */
void gen_fill(int32_t *dst, int64_t start, int64_t cnt, int64_t n,
              uint64_t seed, uint64_t maxv, GenMode mode);
void insertion_sort(int32_t *a, int64_t n);
int32_t quickselect(int32_t *a, int64_t n, int64_t k);
void partition3(int32_t *data, int64_t lo, int64_t hi, int32_t pivot,
                int64_t *lt, int64_t *gt);

#endif
