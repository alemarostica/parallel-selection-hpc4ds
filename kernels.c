#include "core.h"

// DATA GENERATION

/* Values start .. start+cnt-1 of the dataset. Counter-based: value i depends
 * only on (seed, i, n, maxv, mode). */
void gen_fill(int32_t *dst, int64_t start, int64_t cnt, int64_t n,
                     uint64_t seed, uint64_t maxv, GenMode mode)
{

    if (mode == GEN_RANDOM)
    {

        for (int64_t j = 0; j < cnt; j++)
        {

            uint64_t z =
                mix64(seed + (uint64_t)(start + j + 1) * 0x9E3779B97F4A7C15ULL);
            /* multiply-shift: uniform in [0, maxv), one mul, no division */
            dst[j] = (int32_t)(((unsigned __int128)z * maxv) >> 64);
        }
    }
    else
    {

        double scale = (double)maxv / (double)n;
        for (int64_t j = 0; j < cnt; j++)
        {

            int64_t v = (int64_t)((double)(start + j) * scale);
            if (v > (int64_t)maxv - 1)
                v = (int64_t)maxv - 1;
            dst[j] = (int32_t)(mode == GEN_SORTED ? v : (int64_t)maxv - 1 - v);
        }
    }
}

// SEQUENTIAL KERNELS

// TODO: change sorting?
void insertion_sort(int32_t *a, int64_t n) {

    for (int64_t i = 1; i < n; i++) {

        int32_t v = a[i];
        int64_t j = i - 1;
        while (j >= 0 && a[j] > v) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = v;
    }
}

/* In-place quickselect (Hoare partition, median-of-3 pivot). Returns the
 * element of 0-based rank k of a[0..n-1]; the array is reordered. Equal keys
 * stop both scans, so heavy duplication still splits evenly. */
int32_t quickselect(int32_t *a, int64_t n, int64_t k) {

    int64_t lo = 0, hi = n - 1;
    while (hi - lo > 16) {

        int64_t mid = lo + (hi - lo) / 2;
        if (a[mid] < a[lo])
            swap32(&a[mid], &a[lo]);
        if (a[hi] < a[lo])
            swap32(&a[hi], &a[lo]);
        if (a[hi] < a[mid])
            swap32(&a[hi], &a[mid]);
        int32_t pivot = a[mid];
        int64_t i = lo, j = hi;
        while (i <= j) {
            while (a[i] < pivot)
                i++;
            while (a[j] > pivot)
                j--;
            if (i <= j) {

                swap32(&a[i], &a[j]);
                i++;
                j--;
            }
        }
        if (k <= j)
            hi = j;
        else if (k >= i)
            lo = i;
        else
            return a[k]; /* j < k < i: equal to the pivot */
    }
    insertion_sort(a + lo, hi - lo + 1);
    return a[k];
}

/*
 * Three way (Dutch national flag) partition of data[lo, hi) around
 * `pivot`, in place, single pass. On return, with *lt and *gt absolute indices
 * into data:
 *   data[lo, *lt) < pivot
 *   data[*lt, *gt) == pivot
 *   data[*gt, hi) > pivot
 * And empty window is valid and yields *lt == *gt == lo.
 */
void partition3(int32_t *data, int64_t lo, int64_t hi, int32_t pivot,
                       int64_t *lt, int64_t *gt) {

    int64_t i = lo, l = lo, g = hi;
    while (i < g) {

        int32_t v = data[i];
        if (v < pivot) {

            data[i] = data[l];
            data[l] = v;
            l++;
            i++;
        } else if (v > pivot) {

            g--;
            data[i] = data[g];
            data[g] = v;
        } else {

            i++;
        }
    }
    *lt = l;
    *gt = g;
}
/*
 * Three way (Dutch national flag) partition of data[lo, hi) around
 * `pivot`, in place, single pass. On return, with *lt and *gt absolute indices
 * into data:
 *   data[lo, *lt) < u
 *   data[*lt, *gt) in [u,v]
 *   data[*gt, hi) > v
 * And empty window is valid and yields *lt == *gt == lo.
 * 
 * Used by the Floyd-Rivest pivot: with u == v it behaves exactly like partition3. Elements equal to u or v stay in the middle
 */
void partition_bracket(int32_t *data, int64_t lo, int64_t hi, int32_t u, int32_t v, int64_t *lt, int64_t *gt) {

    int64_t i = lo, l = lo, g = hi;
    while (i < g) {

        int32_t x = data[i];
        if (x < u) {

            data[i] = data[l];
            data[l] = x;
            l++;
            i++;
        } else if (x > v) {

            g--;
            data[i] = data[g];
            data[g] = x;
        } else {

            i++;
        }
    }
    *lt = l;
    *gt = g;
}