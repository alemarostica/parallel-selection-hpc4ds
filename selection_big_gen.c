/*
 * selection_big_gen.c - hybrid MPI + OpenMP parallel selection (k-th smallest),
 * with in-memory data generation and low-memory verification for huge inputs.
 *
 * Merge:
 *   - every rank keeps working IN PLACE on a window of its own array (no
 *     second buffer), single-pass three-way (Italian national flag lol)
 *     partition, counts combined with MPI.
 *   - the pivot of the lecture notes ("Algorithm 1": median of block
 *     medians), OpenMP parallelism, switch to a direct solve when few
 *     elements are left ("Algorithm 2"), optional load balancing, options.
 *
 *
 *   repeat while the global problem size N > switch_size:
 *     1. pivot      every rank proposes a candidate, the p candidates are
 *                   combined into a weighted median (weight = window size).
 *                   -p mom   (default, the paper) candidate = median of the
 *                            medians of blocks of ~log2 N elements; guarantees
 *                            the problem shrinks by a constant factor.
 *                   -p rand  (friend's idea) candidate = median of 15 random
 *                            elements; cheaper per round, only expected-case.
 *     2. partition  each thread partitions its chunks of the window in place:
 *                   [ < pivot | == pivot | > pivot ]; counts via Allgather.
 *     3. recurse    keep only the < side or only the > side (window shrinks),
 *                   or stop if the pivot is the answer.
 *     4. (optional) rebalance the survivors over the ranks.
 *   then: gather the few survivors on rank 0 and solve directly.
 *
 * Input file: int64 header n, then n int32 values
 * k is 0-based: k = 0 is the minimum, k = n/2 the median.
 *
 * Build: mpicc -03 -march=native -fopenmp -o selection selection_bin_gen.c
 *
 * TODO: ragionare i launch parameters and shit
 * Run:   OMP_NUM_THREADS=4 mpirun -n 2 ./selection [options] data.bin [k]
 *        OMP_NUM_THREADS=4 mpirun -n 2 ./selection -g N [options] [k]
              (no file,: every ranks generates its own slice of N elements)
 *   -p mom|rand pivot strategy (default mom)
 *   -d          print per-round diagnostic (pivot, sizes, imbalance)
 *   -v          verify the answer against a sequential solve (rank 0 reads the
 *               whole file: only use on files that fit in one rank's memory)
 *   -r RATIO    rebalance the data when max_local / average_local > RATIO
 *               (RATIO > 1, default: never rebalance)
 *   -s SIZE     stop reducing and gather when N <= SIZE
 *               (default: min(n / log2 n, 4M elements))
 *
 *   -g N        generate N elements in memory instead of reading a file. The
 *               values are exactly those of gen_dataset with the same -S/-M/-ms
 *               so file and in-memory runs are comparable.
 *   -S SEED     generator seed (default: 42)
 *   -M MAXV     values in [0, MAXV) (default 2^31-1, at most 2^31)
 *   -m MODE     random (default) | sorted | reverse
 *   -V          verify with O(1) extra memory: every rank re-reads (or
 *               regenerates) its slice in chunks and counts elements < answer
 *               and == answer. Works at any size, unline -v.
 *
 * Memory per rank: 4 bytes x n/p for the data, plus 4/log2(n) bytes x n/p of
 * scratch with -p mom (about 0.15 bytes per element; non with -p rand), plus
 * the MPI/OpenMP runtime (~25-100 MB). Rank 0 also needs up to 16 MB to
 * gather the last survivors. Rebalancing (-r) temporarily neesa one more
 * block of N/p elements. -v needs 4 bytes x n
 * on rank 0: do not use it on huge files, use -V.
 *
 * Launch: keep (ranks per node) x (OMP_NUM_THREADS) <= cores of the node.
 */
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

typedef enum { PIVOT_MOM, PIVOT_RANDOM } PivotMode;
typedef enum { GEN_RANDOM, GEN_SORTED, GEN_REVERSE } GenMode;

/* Each rank owns one block of the global array. */
typedef struct {

    int rank, size;
    int64_t global_n; /* total number of elements in the file */
    int64_t local_n;  /* number of elements loaded by this rank */
    int64_t offset;   /* global index of this rank's first element */
    int32_t *data;    /* local block */
} DistArray;

typedef struct {

    int64_t switch_size;    /* gather when N <= switch_size (0 = default) */
    double rebalance_ratio; /* 0 = never rebalance */
    int verbose;
    PivotMode pivot;
    int64_t gen_n; /* > 0: number of ints to generate */
    uint64_t seed; /* generator seed */
    uint64_t maxv; /* generated values are in [0, maxv) */
    GenMode mode;  /* random | sorted | reverse */
} Options;

typedef struct {

    int rounds;
    int rebalances;
} Stats;

static void suicide(const char *msg) {

    fprintf(stderr, "Error: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static void *xmalloc(size_t bytes) {

    void *p = malloc(bytes > 0 ? bytes : 1);
    if (!p)
        suicide("out of memory");
    return p;
}

// DATA LOADING

/* Block distribution: the first (n % size) ranks get one extra element. */
static void compute_partition(DistArray *a) {

    int64_t base = a->global_n / a->size;
    int64_t rem = a->global_n % a->size;
    a->local_n = base + (a->rank < rem ? 1 : 0);
    a->offset = (int64_t)a->rank * base + (a->rank < rem ? a->rank : rem);
}

static void load_array(const char *path, DistArray *a) {

    MPI_File fh;
    if (MPI_File_open(MPI_COMM_WORLD, path, MPI_MODE_RDONLY, MPI_INFO_NULL,
                      &fh) != MPI_SUCCESS)
        suicide("cannot open input file");

    /* Every rank reads its own slice of the file. */
    MPI_Comm_rank(MPI_COMM_WORLD, &a->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &a->size);

    /* rank 0 reads and validates the header, then broadcasts n */
    int64_t n = 0;
    if (a->rank == 0) {

        MPI_Offset fsize;
        MPI_File_get_size(fh, &fsize);
        if (fsize < HEADER_BYTES)
            suicide("file too small to contain a header");
        MPI_File_read_at(fh, 0, &n, 1,

                         MPI_INT64_T, MPI_STATUS_IGNORE);
        if (n <= 0 ||
            fsize != HEADER_BYTES + (MPI_Offset)n * (MPI_Offset)sizeof(int32_t))
            suicide("header does not match file size (corrupt file?)");
    }
    MPI_Bcast(&n, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    a->global_n = n;

    compute_partition(a);
    a->data = xmalloc(a->local_n * sizeof(int32_t));

    /* chuncked reads: the MPI count is an int */
    int64_t done = 0;
    while (done < a->local_n) {

        int64_t left = a->local_n - done;
        int cnt = (int)(left < READ_CHUNK ? left : READ_CHUNK);
        MPI_Offset byte_off =
            HEADER_BYTES + (MPI_Offset)(a->offset + done) * sizeof(int32_t);
        MPI_File_read_at(fh, byte_off, a->data + done, cnt, MPI_INT32_T,
                         MPI_STATUS_IGNORE);
        done += cnt;
    }
    MPI_File_close(&fh);
}

static void free_array(DistArray *a) {

    free(a->data);
    a->data = NULL;
}

// IN-MEMORY DATA GENERATION

/* splitmix64 finalizer */
static inline uint64_t mix64(uint64_t z) {

    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Values start .. start+cnt-1 of the dataset. Counter-based: value i depends
 * only on (seed, i, n, maxv, mode). */
static void gen_fill(int32_t *dst, int64_t start, int64_t cnt, int64_t n,
                     uint64_t seed, uint64_t maxv, GenMode mode) {

    if (mode == GEN_RANDOM) {

        for (int64_t j = 0; j < cnt; j++) {

            uint64_t z =
                mix64(seed + (uint64_t)(start + j + 1) * 0x9E3779B97F4A7C15ULL);
            /* multiply-shift: uniform in [0, maxv), one mul, no division */
            dst[j] = (int32_t)(((unsigned __int128)z * maxv) >> 64);
        }
    } else {

        double scale = (double)maxv / (double)n;
        for (int64_t j = 0; j < cnt; j++) {

            int64_t v = (int64_t)((double)(start + j) * scale);
            if (v > (int64_t)maxv - 1)
                v = (int64_t)maxv - 1;
            dst[j] = (int32_t)(mode == GEN_SORTED ? v : (int64_t)maxv - 1 - v);
        }
    }
}

/* gen_fill spread over the OpenMP threads */
static void gen_fill_parallel(int32_t *dst, int64_t start, int64_t cnt,
                              int64_t n, const Options *opt) {

    const int64_t BLOCK = 1 << 16;
    int64_t nb = (cnt + BLOCK - 1) / BLOCK;
#pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nb; b++) {

        int64_t s = b * BLOCK, c = cnt - s < BLOCK ? cnt - s : BLOCK;
        gen_fill(dst + s, start + s, c, n, opt->seed, opt->maxv, opt->mode);
    }
}

/* Replaces load_array: every rank generates its own slice, no file, no I/O. */
static void generate_array(DistArray *a, const Options *opt) {

    MPI_Comm_rank(MPI_COMM_WORLD, &a->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &a->size);
    a->global_n = opt->gen_n;
    a->data = xmalloc(a->local_n * sizeof(int32_t));
    gen_fill_parallel(a->data, a->offset, a->local_n, a->global_n, opt);
}

// SEQUENTIAL KERNELS

static inline void swap32(int32_t *x, int32_t *y) {

    int32_t t = *x;
    *x = *y;
    *y = t;
}

// TODO: change sorting?
static void insertion_sort(int32_t *a, int64_t n) {

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
static int32_t quickselect(int32_t *a, int64_t n, int64_t k) {

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

// PIVOT SELECTION

/*
 * Three way (Dutch national flag) partition of data[lo, hi) around
 * `pivot`, in place, single pass. On return, with *lt and *gt absolute indices
 * into data:
 *   data[lo, *lt) < pivot
 *   data[*lt, *gt) == pivot
 *   data[*gt, hi) > pivot
 * And empty window is valid and yields *lt == *gt == lo.
 */
static void partition3(int32_t *data, int64_t lo, int64_t hi, int32_t pivot,
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

/* xorshift64*: small per-rank PRNG with a full 64-bit range. */
static uint64_t rng_next(uint64_t *state) {

    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

// WINDOWS

/* The local array is cut in C fixed slices (chunks). The active elements of
 * slice c are the window data[lo[c], hi[c]). Narrowing the problem only moves
 * lo/hi: nothing is copied. Threads take chunks dynamically, so uneven windows
 * are balanced across threads
 */
typedef struct {

    int C;
    int64_t
        *start; /* C+1 slice boundaries; the other arrays follow in memory */
    int64_t *lo, *hi;
    int64_t *lt, *gt; /* partition results of the current round */
    int64_t *boff;    /* C+1 offsets of each chunk's block medians in scratch */
    int64_t active;   /* sum of window sizes */
} Windows;

static void windows_init(Windows *w, int64_t m, int T) {

    free(w->start);
    int64_t C = (int64_t)CHUNKS_PER_THREAD * T;
    if (m / CHUNK_MIN < C)
        C = m / CHUNK_MIN;
    if (C < 1)
        C = 1;
    w->C = (int)C;
    w->start = xmalloc((6 * C + 2) * sizeof(int64_t));
    w->lo = w->start + (C + 1);
    w->hi = w->lo + C;
    w->lt = w->hi + C;
    w->gt = w->lt + C;
    w->boff = w->gt + C;
    for (int64_t c = 0; c <= C; c++)
        w->start[c] = m * c / C;
    for (int64_t c = 0; c < C; c++) {

        w->lo[c] = w->start[c];
        w->hi[c] = w->start[c + 1];
    }
    w->active = m;
}

/* Move all windows to the front of data[], in order: data[o, active) holds the
 * active elements contiguously. Safe in place (the destination never passes the
 * source) and needs no extra memory */
static void pack_windows(int32_t *data, const Windows *w) {

    int64_t o = 0;
    for (int c = 0; c < w->C; c++) {

        int64_t len = w->hi[c] - w->lo[c];
        if (len > 0 && w->lo[c] != o)
            memmove(data + o, data + w->lo[c], (size_t)len * sizeof(int32_t));
        o += len;
    }
}

/* Block size of the paper: about log2(N), made odd and at least 5. */
static int block_size(int64_t N) {

    int b = 0;
    for (int64_t t = N; t > 1; t >>= 1)
        b++;
    b |= 1;
    return b < 5 ? 5 : b;
}

/* Paper: local median of block medians. The blocks are processed by the OpemMP
 * threads (the paper's "for i = 1 to m/log m pardo"); the elements of a block
 * get reordered, which is harmless. Requires active > 0. */
static int32_t pivot_mom(int32_t *data, Windows *w, int B, int32_t *scratch) {

    const int C = w->C;
    w->boff[0] = 0;
    for (int c = 0; c < C; c++)
        w->boff[c + 1] = w->boff[c] + (w->hi[c] - w->lo[c] + B - 1) / B;

#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < C; c++) {

        int64_t nb = w->boff[c + 1] - w->boff[c];
        for (int64_t i = 0; i < nb; i++) {

            int64_t base = w->lo[c] + i * B;
            int64_t len = w->hi[c] - base;
            if (len > B)
                len = B;
            scratch[w->boff[c] + i] = quickselect(data + base, len, len / 2);
        }
    }
    int64_t nb = w->boff[C];
    return quickselect(scratch, nb, nb / 2);
}

/* Median of samples random elements of the window (a single random element is
 * too noisy when p is small) */
static int32_t pivot_random(const int32_t *data, const Windows *w,
                            uint64_t *rng) {

    int32_t s[SAMPLES];
    for (int j = 0; j < SAMPLES; j++) {

        int64_t r = (int64_t)(rng_next(rng) % (uint64_t)w->active);
        for (int c = 0; c < w->C; c++) {
            int64_t len = w->hi[c] - w->lo[c];
            if (r < len) {

                s[j] = data[w->lo[c] + r];
                break;
            }
            r -= len;
        }
    }
    insertion_sort(s, SAMPLES);
    return s[SAMPLES / 2];
}

/* Weighted median of the p candidates (value, weight). Ranks with weight 0
 * are ignored. The result has weight >= N/2 on each side. Weighting by window
 * size matters: a rank with 5 elements left must not outvote one with 5M. */
static int32_t weighted_median(const int64_t *vw, int *order, int p,
                               int64_t N) {

    int q = 0;
    for (int r = 0; r < p; r++)
        if (vw[2 * r + 1] > 0)
            order[q++] = r;
    for (int i = 1; i < q; i++) { /* insertion sort by candidate value */
        int v = order[i], j = i - 1;
        while (j >= 0 && vw[2 * order[j]] > vw[2 * v]) {

            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = v;
    }
    int64_t cum = 0;
    for (int i = 0; i < q; i++) {

        cum += vw[2 * order[i] + 1];
        if (2 * cum >= N)
            return (int32_t)vw[2 * order[i]];
    }
    return (int32_t)vw[2 * order[q - 1]]; /* unreachable when N > 0 */
}

// LOAD BALANCING (optional)

/* Rebalance: the N surviving elements end up
 * as N/p or N/p + 1 per rank, moving as few elements as possible. Ranks that
 * hold more than their target keep their first tgt elements and send the
 * tail; ranks that hold less append what they receive. So a rank either only
 * sends or only receives. counts[] (known on all ranks) is updated.
 * data must hold this rank's counts[me] elements contiguously; returns the
 * (possibly reallocated) buffer. *moved = total elements that crossed ranks.
 * One MPI_Alltoallv, with separate send/receive buffers (MPI forbids passing
 * the same buffer for both). */
static int32_t *redistribute(int32_t *data, int64_t *counts, int p, int me,
                             int64_t N, int64_t *moved) {

    int64_t base = N / p, rem = N % p;
    int64_t *tgt = xmalloc(2 * p * sizeof(int64_t)), *need = tgt + p;
    int *sc = xmalloc(4 * p * sizeof(int)), *sd = sc + p, *rc = sd + p,
        *rd = rc + p;

    /* the `rem` ranks holding the most get base + 1 (ties: lower rank) */
    for (int r = 0; r < p; r++) {

        int64_t pos = 0;
        for (int q = 0; q < p; q++)
            if (counts[q] > counts[r] || (counts[q] == counts[r] && q < r))
                pos++;
        tgt[r] = base + (pos < rem ? 1 : 0);
        need[r] = tgt[r] - counts[r]; /* >0 deficit, <0 surplus */
    }

    /* pair surpluses with deficits in rank order */
    for (int q = 0; q < p; q++)
        sc[q] = sd[q] = rc[q] = rd[q] = 0;
    /* greedy matching: surplus ranks in order feed deficit ranks in order */
    int64_t tot = 0, sent = 0, recvd = 0;
    int j = 0;
    for (int i = 0; i < p; i++) {

        int64_t sur = -need[i];
        while (sur > 0) {

            while (need[j] <= 0)
                j++; /* total surplus == total deficit, so j stays < p */
            int64_t amt = sur < need[j] ? sur : need[j];
            if (i == me) {

                if (amt > INT_MAX || tgt[me] + sent > INT_MAX)
                    suicide("rebalance message exceeds INT_MAX elements");
                sc[j] = (int)amt;
                sd[j] = (int)(tgt[me] + sent); /* tail of my data */
                sent += amt;
            }
            if (j == me) {

                if (amt > INT_MAX || counts[me] + recvd > INT_MAX)
                    suicide("rebalance message exceeds INT_MAX elements");
                rc[i] = (int)amt;
                rd[i] = (int)(counts[me] + recvd); /* appended after mine */
                recvd += amt;
            }
            sur -= amt;
            need[j] -= amt;
            tot += amt;
        }
    }

    if (recvd > 0) { /* only deficit ranks grow, and they send nothing */
        data = realloc(data, (size_t)tgt[me] * sizeof(int32_t));
        if (!data)
            suicide("out of memory");
    }
    /* a rank only sends or only receives: the unused side gets a dummy */
    int32_t dummy = 0;
    int32_t *sbuf = (sent > 0) ? data : &dummy;
    int32_t *rbuf = (recvd > 0) ? data : &dummy;
    MPI_Alltoallv((sbuf), sc, sd, MPI_INT32_T, rbuf, rc, rd, MPI_INT32_T,
                  MPI_COMM_WORLD);
    if (sent > 0 && tgt[me] > 0) { /* surplus rank: give the tail back*/

        int32_t *shrunk = realloc(data, (size_t)tgt[me] * sizeof(int32_t));
        if (shrunk)
            data = shrunk;
    }

    for (int r = 0; r < p; r++)
        counts[r] = tgt[r];
    *moved = tot;
    free(tgt);
    free(sc);
    return data;
}

// PARALLEL SELECTION

/* Return the k-th smallest (0-based) element of the distributed array.
 * COLLECTIVE: every rank must call it with the same k; all ranks return the
 * same value. Invariant: the answer is the k-th smallest among the union of the
 * ranks' windows; k is relative to that union. */
static int32_t select_kth(DistArray *arr, int64_t k, const Options *opt,
                          Stats *st) {

    const int p = arr->size, me = arr->rank;
    int32_t *data = arr->data;
    int64_t N = arr->global_n;
    const int T = omp_get_max_threads();

    int64_t switch_size = opt->switch_size;
    if (switch_size <= 0) {

        int lg = 1;
        for (int64_t t = N; t > 1; t >>= 1)
            lg++;
        switch_size = N / lg;
        if (switch_size > MAX_SWITCH)
            switch_size = MAX_SWITCH;
    }
    if (switch_size > (1 << 30)) /* Gatherv counts are ints */
        switch_size = 1 << 30;
    if (switch_size < 1)
        switch_size = 1;

    Windows w = {0};
    windows_init(&w, arr->local_n, T);

    /* no rank ever holds more than ceil(N/p) elements, even when rebalanced */
    int64_t cap = N / p + 1;
    int32_t *scratch = NULL;
    if (opt->pivot == PIVOT_MOM)
        scratch = xmalloc((cap / block_size(switch_size) +
                           (int64_t)CHUNKS_PER_THREAD * T + 8) *
                          sizeof(int32_t));

    int64_t *counts = xmalloc(p * sizeof(int64_t));
    int64_t *tri = xmalloc(3 * p * sizeof(int64_t));
    int64_t *vw = xmalloc(2 * p * sizeof(int64_t));
    int *order = xmalloc(p * sizeof(int));
    uint64_t rng = 0x9E3779B97F4A7C15ULL * ((uint64_t)me + 1); /* per rank */

    MPI_Allgather(&w.active, 1, MPI_INT64_T, counts, 1, MPI_INT64_T,
                  MPI_COMM_WORLD);

    int32_t result = 0;
    int found = 0;

    while (N > switch_size) {

        double t0 = MPI_Wtime();
        const int64_t N_before = N;
        const int C = w.C;

        /* pivot: everyone proposes a candidate, all pick the same weighted
         * median */
        int B = block_size(N);
        int64_t mine[2] = {0, w.active};
        if (w.active > 0)
            mine[0] = (opt->pivot == PIVOT_MOM)
                          ? pivot_mom(data, &w, B, scratch)
                          : pivot_random(data, &w, &rng);
        MPI_Allgather(mine, 2, MPI_INT64_T, vw, 2, MPI_INT64_T, MPI_COMM_WORLD);
        const int32_t pivot = weighted_median(vw, order, p, N);

        /* three-way partition counts, one chunk per thread */
#pragma omp parallel for schedule(static)
        for (int c = 0; c < C; c++)
            partition3(data, w.lo[c], w.hi[c], pivot, &w.lt[c], &w.gt[c]);

        int64_t loc[3] = {0, 0, 0};
        for (int c = 0; c < C; c++) {

            loc[0] += w.lt[c] - w.lo[c];
            loc[1] += w.gt[c] - w.lt[c];
            loc[2] += w.hi[c] - w.gt[c];
        }
        MPI_Allgather(loc, 3, MPI_INT64_T, tri, 3, MPI_INT64_T, MPI_COMM_WORLD);
        int64_t L = 0, E = 0, G = 0;
        for (int r = 0; r < p; r++) {

            L += tri[3 * r];
            E += tri[3 * r + 1];
            G += tri[3 * r + 2];
        }

        /* every rank sees the same L, E, G and k -> same branch everywhere */
        int keep_less;
        if (k < L)
            keep_less = 1;
        else if (k < L + E) {

            result = pivot;
            found = 1;
            st->rounds++;
            if (opt->verbose && me == 0)
                printf("  round %2d: N=%" PRId64 " pivot=%d L=%" PRId64
                       " E=%" PRId64 " G=%" PRId64 " -> pivot is the answer\n",
                       st->rounds, N_before, pivot, L, E, G);
            break;
        } else {

            keep_less = 0;
            k -= L + E;
        }
        for (int c = 0; c < C; c++) {

            if (keep_less)
                w.hi[c] = w.lt[c];
            else
                w.lo[c] = w.gt[c];
        }
        for (int r = 0; r < p; r++)
            counts[r] = keep_less ? tri[3 * r] : tri[3 * r + 2];
        N = keep_less ? L : G;
        w.active = counts[me];

        /* optional load balancing */
        int64_t mx = 0;
        for (int r = 0; r < p; r++)
            if (counts[r] > mx)
                mx = counts[r];
        double imbalance = N > 0 ? (double)mx * p / (double)N : 1.0;
        int did_rebalance = 0;
        int64_t moved = 0;
        if (opt->rebalance_ratio > 0 && N > switch_size &&
            imbalance > opt->rebalance_ratio) {

            pack_windows(data, &w);
            data = redistribute(data, counts, p, me, N, &moved);
            windows_init(&w, counts[me], T);
            did_rebalance = 1;
            st->rebalances++;
        }

        char rb[64] = "";
        if (did_rebalance)
            snprintf(rb, sizeof rb, " [rebalanced, moved %" PRId64 "]", moved);
        st->rounds++;
        if (opt->verbose && me == 0)
            printf("  round %2d: N=%" PRId64
                   " B=%d pivot=%d keep=%s -> N=%" PRId64
                   " (%.1f%%) imbalance=%.2f%s  %.1f ms\n",
                   st->rounds, N_before, B, pivot, keep_less ? "L" : "R", N,
                   100.0 * (double)N / (double)N_before, imbalance, rb,
                   1e3 * (MPI_Wtime() - t0));
    }

    if (!found) {

        /* few elements are left: gather them on rank 0 and solve directly
         * (quickselect instead of sorting: same answer, linear time). */
        pack_windows(data, &w);
        int *rc = xmalloc(2 * p * sizeof(int)), *rd = rc + p;
        int64_t s = 0;
        for (int r = 0; r < p; r++) {

            rc[r] = (int)counts[r];
            rd[r] = (int)s;
            s += counts[r];
        }
        int32_t *all = (me == 0) ? xmalloc(N * sizeof(int32_t)) : NULL;
        MPI_Gatherv(data, (int)w.active, MPI_INT32_T, all, rc, rd, MPI_INT32_T,
                    0, MPI_COMM_WORLD);
        if (me == 0) {

            result = quickselect(all, N, k);
            free(all);
        }
        MPI_Bcast(&result, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
        free(rc);
        if (opt->verbose && me == 0)
            printf("  gathered %" PRId64 " elements on rank 0, k=%" PRId64 "\n",
                   N, k);
    }

    arr->data = data;
    free(w.start);
    free(scratch);
    free(counts);
    free(tri);
    free(vw);
    free(order);
    return result;
}

// VERIFICATION (rank 0 only, reads the whole file)

static int verify(const char *path, int64_t k, int32_t got) {

    FILE *f = fopen(path, "rb");
    if (!f)
        suicide("verify: cannot open file");
    int64_t n;
    if (fread(&n, sizeof n, 1, f) != 1)
        suicide("verify: bad header");
    int32_t *v = xmalloc(n * sizeof(int32_t));
    if (fread(v, sizeof(int32_t), n, f) != (size_t)n)
        suicide("verify: short read");
    fclose(f);
    int32_t want = quickselect(v, n, k);
    free(v);
    printf("verify: expected %d, got %d -> %s\n", want, got,
           want == got ? "OK" : "MISMATCH");
    return want == got;
}

/*
 * Distribued verification with O(1) extra memory. Every ranks streams its
 * ORIGINAL slice again (re-read from the file, or regenerated), in chunks, and
 * counts the elements < got and == got. The answer is the k-th smallest
 * (0-based) iff less <= k < less + equal.  COLLECTIVE call on all ranks.
 */
static int verify_stream(const char *path, const DistArray *arr,
                         const Options *opt, int64_t k, int32_t got) {

    const int64_t CH = 1 << 22;
    int32_t *buf = xmalloc(CH * sizeof(int32_t));
    MPI_File fh = MPI_FILE_NULL;
    if (!opt->gen_n && MPI_File_open(MPI_COMM_WORLD, path, MPI_MODE_RDONLY,
                                     MPI_INFO_NULL, &fh) != MPI_SUCCESS)
        suicide("verify: cannot reopen the file");

    int64_t less = 0, equal = 0;
    for (int64_t done = 0; done < arr->local_n;) {

        int64_t c = arr->local_n - done < CH ? arr->local_n - done : CH;
        if (opt->gen_n)
            gen_fill_parallel(buf, arr->offset + done, c, arr->global_n, opt);
        else
            MPI_File_read_at(fh,
                             HEADER_BYTES + (MPI_Offset)(arr->offset + done) *
                                                (MPI_Offset)sizeof(int32_t),
                             buf, (int)c, MPI_INT32_T, MPI_STATUS_IGNORE);
#pragma omp parallel for reduction(+ : less, equal) schedule(static)
        for (int64_t i = 0; i < c; i++) {

            less += (buf[i] < got);
            equal += (buf[i] == got);
        }
        done += c;
    }
    if (fh != MPI_FILE_NULL)
        MPI_File_close(&fh);
    free(buf);

    int64_t loc[2] = {less, equal}, glob[2];
    MPI_Allreduce(loc, glob, 2, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    int ok = glob[0] <= k && k < glob[0] + glob[1];
    if (arr->rank == 0)

        printf("verify (distributed): %" PRId64 " elements < %d, %" PRId64
               " equal, k=%" PRId64 "  -> %s\n",
               glob[0], got, glob[1], k, ok ? "OK" : "MISMATCH");
    return ok;
}

// MAIN

int main(int argc, char **argv) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
        suicide("MPI lacks MPI_THREAD_FUNNELED");
    omp_set_dynamic(0);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Options opt = {.pivot = PIVOT_MOM,
                   .seed = 42,
                   .maxv = 2147483647ULL,
                   .mode = GEN_RANDOM};
    int do_verify = 0, do_verify_stream = 0, opterr_flag = 0, c;
    while ((c = getopt(argc, argv, "dvVr:s:p:g:S:M:m")) != -1) {
        switch (c) {
        case 'd':
            opt.verbose = 1;
            break;
        case 'v':
            do_verify = 1;
            break;
        case 'V':
            do_verify_stream = 1;
            break;
        case 'r':
            opt.rebalance_ratio = atof(optarg);
            break;
        case 's':
            opt.switch_size = strtoll(optarg, NULL, 10);
            break;
        case 'p':
            if (strcmp(optarg, "mom") == 0)
                opt.pivot = PIVOT_MOM;
            else if (strcmp(optarg, "rand") == 0)
                opt.pivot = PIVOT_RANDOM;
            else
                opterr_flag = 1;
            break;
        case 'g':
            opt.gen_n = strtoll(optarg, NULL, 10);
            if (opt.gen_n <= 0)
                opterr_flag = 1;
            break;
        case 'S':
            opt.seed = strtoull(optarg, NULL, 10);
            break;
        case 'M':
            opt.maxv = strtoull(optarg, NULL, 10);
            break;
        case 'm':
            if (strcmp(optarg, "random") == 0)
                opt.mode = GEN_RANDOM;
            else if (strcmp(optarg, "sorted") == 0)
                opt.mode = GEN_SORTED;
            else if (strcmp(optarg, "reverse") == 0)
                opt.mode = GEN_REVERSE;
            else
                opterr_flag = 1;
        default:
            opterr_flag = 1;
        }
    }

    const int gen = opt.gen_n > 0;
    const int nfiles = gen ? 0 : 1; /* positional args: [file] [k] */
    if (opterr_flag || optind + nfiles > argc ||
        (opt.rebalance_ratio != 0 && opt.rebalance_ratio <= 1.0) ||
        opt.maxv == 0 || opt.maxv > 2147483648ULL || (gen && do_verify)) {

        if (rank == 0)

            fprintf(stderr,
                    "Usage: %s [-p mom|rand] [-d] [-v] [-r ratio>1] "
                    "[-s switch_size] <data_file> [k]\n"
                    "       %s -g N [-S seed] [-M maxv] [-m ]"
                    "random|sorted|reverse] [other options] [k]\n"
                    "       (-v needs a file; with -g use -V)\n",
                    argv[0], argv[0]);
        MPI_Finalize();
        return 1;
    }
    const char *path = gen ? NULL : argv[optind];
    const int kpos = optind + nfiles;

    DistArray arr;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    if (gen)
        generate_array(&arr, &opt);
    else
        load_array(path, &arr);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_load = MPI_Wtime() - t0;

    /* k-th smallest element to find (0-based); default: median */
    int64_t k = (kpos < argc) ? strtoll(argv[kpos], NULL, 10)
                              : arr.global_n / 2;
    if (k < 0 || k >= arr.global_n)
        suicide("k out of range");

    if (rank == 0) {

        double data_mb = (double)arr.local_n * 4 / 104856.0;
        double scratch_mb =
            opt.pivot == PIVOT_MOM
                ? (double)arr.local_n / block_size(arr.global_n) * 4 / 1048576.0
                : 0.0;
        printf("n=%" PRId64 "  ranks=%d  threads/rank=%d  pivot=%s  k=%" PRId64
               "  %s=%.3f s\n",
               arr.global_n, arr.size, omp_get_max_threads(),
               opt.pivot == PIVOT_MOM ? "mom" : "rand", k,
               gen ? "generate" : "load", t_load);
        printf(
            "memory per rank: ~%.0f MB data + ~%.0f MB scratch (+ runtime)\n",
            data_mb, scratch_mb);
    }

    Stats st = {0, 0};
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    int32_t answer = select_kth(&arr, k, &opt, &st);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_sel = MPI_Wtime() - t0;

    int ok = 1;
    if (rank == 0) {

        printf("k-th smallest = %d   (%d rounds, %d rebalances, %.3f s)\n",
               answer, st.rounds, st.rebalances, t_sel);
        if (do_verify)
            ok = verify(path, k, answer);
    }
    if (do_verify_stream)
        ok = verify_stream(path, &arr, &opt, k, answer);

    free_array(&arr);
    MPI_Finalize();
    return ok ? 0 : 2;
}
