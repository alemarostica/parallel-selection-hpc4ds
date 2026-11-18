#ifndef SELECTION_H
#define SELECTION_H

#include "core.h"
#include <inttypes.h>
#include <limits.h>
#include <mpi.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define HEADER_BYTES ((MPI_Offset)sizeof(int64_t))
#define READ_CHUNK (1 << 26) /* max elements per MPI_File_read_at call */
#define MAX_SWITCH (1 << 22) /* default cap on the gather size (4M ints) */
#define SAMPLES 15           /* random pivot: samples per rank */
#define CHUNK_MIN 2048       /* smallest chunk worth scheduling */
#define CHUNKS_PER_THREAD 16 /* chunks per thread, for dynamic scheduling */

// DATA STRUCTURE

typedef enum
{
    PIVOT_MOM,
    PIVOT_RANDOM,
    PIVOT_FR,
} PivotMode;

/* Each rank owns one block of the global array. */
typedef struct
{

    int rank, size;
    int64_t global_n; /* total number of elements in the file */
    int64_t local_n;  /* number of elements loaded by this rank */
    int64_t offset;   /* global index of this rank's first element */
    int32_t *data;    /* local block */
} DistArray;

typedef struct
{

    int64_t switch_size;    /* gather when N <= switch_size (0 = default) */
    double rebalance_ratio; /* 0 = never rebalance */
    int verbose;
    PivotMode pivot;
    int64_t gen_n; /* > 0: number of ints to generate */
    uint64_t seed; /* generator seed */
    uint64_t maxv; /* generated values are in [0, maxv) */
    GenMode mode;  /* random | sorted | reverse */
    int64_t fr_sample; /* FR sample size in elements (0 = N^(2/3)/2) */
    int fr_readonly;   /* FR round 1: read-only counting pass (-R) */
} Options;

typedef struct
{

    int rounds;
    int rebalances;
} Stats;

// WINDOWS

/* The local array is cut in C fixed slices (chunks). The active elements of
 * slice c are the window data[lo[c], hi[c]). Narrowing the problem only moves
 * lo/hi: nothing is copied. Threads take chunks dynamically, so uneven windows
 * are balanced across threads
 */
typedef struct
{

    int C;
    int64_t
        *start; /* C+1 slice boundaries; the other arrays follow in memory */
    int64_t *lo, *hi;
    int64_t *lt, *gt; /* partition results of the current round */
    int64_t *boff;    /* C+1 offsets of each chunk's block medians in scratch */
    int64_t active;   /* sum of window sizes */
} Windows;

/* io_gen.c */
void suicide(const char *msg);
void *xmalloc(size_t bytes);
void compute_partition(DistArray *a);
void load_array(const char *path, DistArray *a);
void free_array(DistArray *a);
void gen_fill_parallel(int32_t *dst, int64_t start, int64_t cnt, int64_t n,
                       const Options *opt);
void generate_array(DistArray *a, const Options *opt);

/* pivot.c */
uint64_t rng_next(uint64_t *state);
void windows_init(Windows *w, int64_t m, int T);
void pack_windows(int32_t *data, const Windows *w);
int block_size(int64_t N);
int32_t pivot_mom(int32_t *data, Windows *w, int B, int32_t *scratch);
int32_t pivot_random(const int32_t *data, const Windows *w, uint64_t *rng);
int32_t weighted_median(const int64_t *vw, int *order, int p, int64_t N);
void fr_bounds(const int32_t *data, const Windows *w, const int64_t *counts, int64_t N, int64_t k, int p, int me, uint64_t *rng, int64_t s_req, double *tm, int32_t *u, int32_t *v);

#endif
