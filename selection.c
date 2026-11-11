/*
 * selection.c - MPI parallel selection algorithm.
 *
 * Loads a binary file produced by gen_dataset (int64 header n + n int32 values)
 * and distributes it in contiguous blocks over the MPI ranks.
 */
#include <inttypes.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#define HEADER_BYTES ((MPI_Offset)sizeof(int64_t))
/* max elements per MPI_File_read_at call (count is an int) */
#define READ_CHUNK (1 << 26)

/* Each rank owns one block of the global array. */
typedef struct {

    int rank, size;
    int64_t global_n; /* total number of elements in the dataset */
    int64_t local_n;  /* number of elements held by this rank */
    int64_t offset;   /* global index of this rank's first element */
    int32_t *data;    /* local block, length local_n */
} DistArray;

static void suicide(const char *msg) {

    fprintf(stderr, "Error: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

/* Block distribution: the first (n % size) ranks get one extra element. */
static void compute_partition(DistArray *a) {

    int64_t base = a->global_n / a->size;
    int64_t rem = a->global_n % a->size;
    a->local_n = base + (a->rank < rem ? 1 : 0);
    a->offset = (int64_t)a->rank * base + (a->rank < rem ? a->rank : rem);
}

/* File read is distributed too, read the header, and load this rank's block. */
static void load_array(const char *path, DistArray *a) {

    MPI_File fh;
    if (MPI_File_open(MPI_COMM_WORLD, path, MPI_MODE_RDONLY, MPI_INFO_NULL,
                      &fh) != MPI_SUCCESS)
        suicide("cannot open input file");

    MPI_Comm_rank(MPI_COMM_WORLD, &a->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &a->size);

    /* rank 0 reads the header, then broadcasts it */
    int64_t n = 0;
    if (a->rank == 0) {

        MPI_Offset fsize;
        MPI_File_get_size(fh, &fsize);
        if (fsize < HEADER_BYTES)
            suicide("file too small to contain a header");
        MPI_File_read_at(fh, 0, &n, 1, MPI_INT64_T, MPI_STATUS_IGNORE);
        if (n <= 0 ||
            fsize != HEADER_BYTES + (MPI_Offset)n * (MPI_Offset)sizeof(int32_t))
            suicide("header does not match file size (corrupt file?)");
    }
    MPI_Bcast(&n, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    a->global_n = n;

    compute_partition(a);

    a->data = malloc((a->local_n > 0 ? a->local_n : 1) * sizeof(int32_t));
    if (!a->data)
        suicide("out of memory");

    /* Each rank reads its own slice, in chunks to respect the int count limit.
     */
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

/*
 * Three-way (Dutch national flag) partition of the window data[lo, hi)
 * around `pivot`, in place, single pass, no communication.
 *
 * On return, with *lt and *gt as absolute indices into data:
 *   data[lo, *lt)   <  pivot
 *   data[*lt, *gt)  == pivot
 *   data[*gt, hi)   >  pivot
 * so the local group sizes are (*lt - lo), (*gt - *lt) and (hi - *gt).
 * An empty window (lo == hi) is valid and yields *lt == *gt == lo.
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

/* xorshift64*: small per-rank PRNG with a full 64-bit range, so indexing
 * works even if a rank holds more than RAND_MAX elements. */
static uint64_t rng_next(uint64_t *state) {

    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static int cmp_i32(const void *x, const void *y) {

    int32_t a = *(const int32_t *)x, b = *(const int32_t *)y;
    return (a > b) - (a < b);
}

/*
 * Returns the k-th smallest (0-based) element of the distributed array.
 * COLLECTIVE: every rank must call it with the same k. All ranks return
 * the same value.
 *
 * Invariant: the answer is the k-th smallest among the union of every
 * rank's window data[lo, hi); k is relative to that union.
 */
static int32_t select_kth(DistArray *a, int64_t k) {

    int64_t lo = 0, hi = a->local_n;
    int64_t lt, gt;

    /* per rank: {candidate, valid flag}; gathered: 2 ints from each rank */
    int32_t *gathered = malloc(2 * (size_t)a->size * sizeof(int32_t));
    int32_t *valid_c = malloc((size_t)a->size * sizeof(int32_t));
    if (!gathered || !valid_c)
        suicide("out of memory");

    /* different stream on every rank */
    uint64_t rng = 0x9E3779B97F4A7C15ULL * ((uint64_t)a->rank + 1);

    int32_t result;
    while (true) {

        /* 1. pivot agreement: everyone shares a candidate, all pick the same */
        int32_t mine[2];
        /* random element of my window; value ignored if window is empty */
        mine[0] = lo < hi ? a->data[lo + (int64_t)(rng_next(&rng) % (uint64_t)(hi - lo))] : 0;
        mine[1] = lo < hi;
        MPI_Allgather(mine, 2, MPI_INT32_T, gathered, 2, MPI_INT32_T,
                      MPI_COMM_WORLD);

        int nvalid = 0;
        for (int r = 0; r < a->size; r++)
            if (gathered[2 * r + 1])
                valid_c[nvalid++] = gathered[2 * r];
        /* k < total remaining elements, so at least one window is non-empty */
        qsort(valid_c, nvalid, sizeof(int32_t), cmp_i32);
        int32_t pivot = valid_c[nvalid / 2];

        /* 2. local three-way partition of my window */
        partition3(a->data, lo, hi, pivot, &lt, &gt);

        /* 3. global sizes of the <, ==, > groups */
        int64_t loc[3] = {lt - lo, gt - lt, hi - gt};
        int64_t glob[3];
        MPI_Allreduce(loc, glob, 3, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

        /* 4. every rank sees the same glob and k -> same branch everywhere */
        if (k < glob[0]) {
            hi = lt; /* answer is among the smaller ones */
        } else if (k < glob[0] + glob[1]) {
            result = pivot; /* answer falls in the == block */
            break;
        } else {
            k -= glob[0] + glob[1]; /* skip < and == groups */
            lo = gt;
        }
    }

    free(gathered);
    free(valid_c);
    return result;
}
int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank;
    int world_size;
    int32_t value;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <data_file> [k]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    DistArray arr;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    load_array(argv[1], &arr);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_load = MPI_Wtime() - t0;
    /* k-th smallest element to find (0-based); default: median */
    int64_t k = (argc > 2) ? strtoll(argv[2], NULL, 10) : arr.global_n / 2;
    if (k < 0 || k >= arr.global_n)
        suicide("k oout of range");

    if (rank == 0){
        printf("Loaded %" PRId64
               " integers on %d ranks in %.3f s; looking for k=%" PRId64 "\n",
               arr.global_n, arr.size, t_load, k);
        printf("  rank %d: local_n=%" PRId64 ", global offset=%" PRId64 "\n",
               arr.rank, arr.local_n, arr.offset);
    
    
    }

    value = select_kth(&arr,k);
    if(rank == 0) printf("The k-th value in the collection is: %d\n",value);	
    free_array(&arr);
    MPI_Finalize();
    return 0;
}
