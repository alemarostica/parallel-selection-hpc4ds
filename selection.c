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

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    if (rank == 0)
        printf("Loaded %" PRId64
               " integers on %d ranks in %.3f s; looking for k=%" PRId64 "\n",
               arr.global_n, arr.size, t_load, k);
    printf("  rank %d: local_n=%" PRId64 ", global offset=%" PRId64 "\n",
           arr.rank, arr.local_n, arr.offset);

    /* TODO: parallel selection algorithm lmao
     * Input: arr.data (local block), arr.local_n, arr.global_n, k
     * Output: the k-th smallest element, known on rank 0 (or all ranks)
     */

    free_array(&arr);
    MPI_Finalize();
    return 0;
}
