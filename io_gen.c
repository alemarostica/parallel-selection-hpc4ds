#include "selection.h"

void suicide(const char *msg)
{

    fprintf(stderr, "Error: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

void *xmalloc(size_t bytes)
{

    void *p = malloc(bytes > 0 ? bytes : 1);
    if (!p)
        suicide("out of memory");
    return p;
}

// DATA LOADING

/* Block distribution: the first (n % size) ranks get one extra element. */
void compute_partition(DistArray *a)
{

    int64_t base = a->global_n / a->size;
    int64_t rem = a->global_n % a->size;
    a->local_n = base + (a->rank < rem ? 1 : 0);
    a->offset = (int64_t)a->rank * base + (a->rank < rem ? a->rank : rem);
}

void load_array(const char *path, DistArray *a)
{

    MPI_File fh;
    if (MPI_File_open(MPI_COMM_WORLD, path, MPI_MODE_RDONLY, MPI_INFO_NULL,
                      &fh) != MPI_SUCCESS)
        suicide("cannot open input file");

    /* Every rank reads its own slice of the file. */
    MPI_Comm_rank(MPI_COMM_WORLD, &a->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &a->size);

    /* rank 0 reads and validates the header, then broadcasts n */
    int64_t n = 0;
    if (a->rank == 0)
    {

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
    while (done < a->local_n)
    {

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

void free_array(DistArray *a)
{

    free(a->data);
    a->data = NULL;
}
// IN-MEMORY DATA GENERATION

/* gen_fill spread over the OpenMP threads */
void gen_fill_parallel(int32_t *dst, int64_t start, int64_t cnt,
                              int64_t n, const Options *opt)
{

    const int64_t BLOCK = 1 << 16;
    int64_t nb = (cnt + BLOCK - 1) / BLOCK;
#pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nb; b++)
    {

        int64_t s = b * BLOCK, c = cnt - s < BLOCK ? cnt - s : BLOCK;
        gen_fill(dst + s, start + s, c, n, opt->seed, opt->maxv, opt->mode);
    }
}

/* Replaces load_array: every rank generates its own slice, no file, no I/O. */
void generate_array(DistArray *a, const Options *opt)
{

    MPI_Comm_rank(MPI_COMM_WORLD, &a->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &a->size);
    a->global_n = opt->gen_n;
    compute_partition(a);
    a->data = xmalloc(a->local_n * sizeof(int32_t));
    gen_fill_parallel(a->data, a->offset, a->local_n, a->global_n, opt);
}
