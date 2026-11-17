#define _POSIX_C_SOURCE 200809L
#include "core.h"
#include <time.h>
#include <unistd.h>

static void suicide(const char *msg)
{
    fprintf(stderr, "Error: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t bytes)
{
    void *p = malloc(bytes > 0 ? bytes : 1);
    if (!p)
        suicide("out of memory");
    return p;
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int cmp32(const void *x, const void *y)
{
    int32_t a = *(const int32_t *)x, b = *(const int32_t *)y;
    return (a > b) - (a < b);
}

/* file format: int64 header n, then n int32 values */
static int32_t *load_file(const char *path, int64_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        suicide("cannot open input file");
    if (fread(n, sizeof *n, 1, f) != 1 || *n <= 0)
        suicide("bad header");
    int32_t *a = xmalloc((size_t)*n * sizeof *a);
    if (fread(a, sizeof *a, (size_t)*n, f) != (size_t)*n)
        suicide("header does not match file size (corrupt file?)");
    fclose(f);
    return a;
}

int main(int argc, char **argv)
{
    int64_t gen_n = 0;
    uint64_t seed = 42, maxv = 2147483647ULL;
    GenMode mode = GEN_RANDOM;
    int verify = 0, bad = 0, c;
    while ((c = getopt(argc, argv, "vg:S:M:m:")) != -1)
    {
        switch (c)
        {
        case 'v':
            verify = 1;
            break;
        case 'g':
            gen_n = strtoll(optarg, NULL, 10);
            bad |= gen_n <= 0;
            break;
        case 'S':
            seed = strtoull(optarg, NULL, 10);
            break;
        case 'M':
            maxv = strtoull(optarg, NULL, 10);
            break;
        case 'm':
            if (!strcmp(optarg, "random"))
                mode = GEN_RANDOM;
            else if (!strcmp(optarg, "sorted"))
                mode = GEN_SORTED;
            else if (!strcmp(optarg, "reverse"))
                mode = GEN_REVERSE;
            else
                bad = 1;
            break;
        default:
            bad = 1;
        }
    }
    int nfiles = gen_n > 0 ? 0 : 1;
    if (bad || optind + nfiles > argc || maxv == 0 || maxv > 2147483648ULL)
    {
        fprintf(stderr, "Usage: %s [-v] <data_file> [k]\n"
                        "       %s -g N [-S seed] [-M maxv] [-m random|sorted|reverse] [-v] [k]\n",
                argv[0], argv[0]);
        return 1;
    }
    double t0 = now();
    int64_t n = gen_n;
    int32_t *a;
    if (gen_n > 0)
    {
        a = xmalloc((size_t)n * sizeof *a);
        gen_fill(a, 0, n, n, seed, maxv, mode);
    }
    else
    {
        a = load_file(argv[optind], &n);
    }
    double t_load = now() - t0;
    int kpos = optind + nfiles;
    int64_t k = kpos < argc ? strtoll(argv[kpos], NULL, 10) : n / 2;
    if (k < 0 || k >= n)
        suicide("Out of range");
    int32_t *copy = NULL;
    if (verify)
    {
        copy = malloc((size_t)n * sizeof *copy);
        if (!copy)
            suicide("Out of memory");
        memcpy(copy, a, (size_t)n * sizeof *a);
    }
    t0 = now();
    int32_t answer = quickselect(a, n, k);
    double t_sel = now() - t0;
    printf("n=%" PRId64 "   k=%" PRId64 "   %s=%.3f s\n", n, k, gen_n > 0 ? "generate" : "load", t_load);
    printf("k-th smallest = %d  (%.3f s)\n", answer, t_sel);

    int ok = 1;
    if (verify)
    {
        qsort(copy, (size_t)n, sizeof *copy, cmp32);
        ok = copy[k] == answer;
        printf("verify: %s (qsort says %d)\n", ok ? "OK" : "MISMATCH", copy[k]);
        free(copy);
    }
    free(a);
    return ok ? 0 : 2;
}