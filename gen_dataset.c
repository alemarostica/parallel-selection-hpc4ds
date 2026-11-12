/*
 * gen_dataset.c - fast parallel generator of binary dataset
 *
 * File layout (little endian): int64 n, then n int32 values in [0, max_value)
 *
 * Counter-based: value i depends only on (seed, i, n, max_value, mode), never
 * on the previous values. So
 *   - the file is byte-identical for any number of threads, and
 *   - `selection_big -g n -S seed -M max_value -m mode` regenerates exactly
 * these values in memory, without any file.
 * Threads generate blocks indipendently and write them with pwrite() at their
 * own offsets, so the speed is limited by the disk, not by the generator.
 *
 * Build: gcc -O3 -fopenmp -o gen_dataset gen_dataset.c
 * Usage: OMP_NUM_THREADS=8 ./gen_dataset <n> <output_file> [seed] [max_value]
 *                          [--mode random|sorted|reverse]
 *   seed      default 42
 *   max_value values in [0, max_value), default 2^31-1, at most 2^31
 *   --mode    random (default): uniform random values
 *             sorted / reverse: globally sorted ramp (makes the ranks lose
 *             elements unevenly during selection: useful to test balancing)
 */
#include <asm-generic/errno-base.h>
#include <stddef.h>
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef enum { GEN_RANDOM, GEN_SORTED, GEN_REVERSE } GenMode;

/* splitmix64 finalizer */
static inline uint64_t mix64(uint64_t z) {

    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 32);
}

/* Values start .. start+cnt-1 of the dataset. SAME as in selection file, must
 * be indentical. */
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

static int write_all(int fd, const void *buf, size_t len, off_t off) {

    const char *p = buf;
    while (len > 0) {

        ssize_t w = pwrite(fd, p, len, off);
        if (w < 0) {

            if (errno == EINTR)
                continue;
            return -1;
        }
        p += w;
        off += w;
        len -= (size_t)w;
    }
    return 0;
}

static void usage(const char *prog) {

    fprintf(stderr,
            "Usage: %s <n> <output_file> [seed] [max_value] "
            "[--mode random|sorted|reverse]\n",
            prog);
    exit(1);
}

int main(int argc, char **argv) {

    const uint16_t one = 1;
    if (*(const uint8_t *)&one != 1) {

        fprintf(stderr, "this generator assumes a little-endian host\n");
        return 1;
    }

    const char *pos[4] = {0};
    int npos = 0;
    GenMode mode = GEN_RANDOM;
    for (int i = 1; i < argc; i++) {

        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {

            i++;
            if (strcmp(argv[i], "random") == 0)
                mode = GEN_RANDOM;
            else if (strcmp(argv[i], "sorted"))
                mode = GEN_SORTED;
            else if (strcmp(argv[i], "reverse"))
                mode = GEN_REVERSE;
            else
                usage(argv[0]);
        } else if (npos < 4) {
            pos[npos++] = argv[i];
        } else {
            usage(argv[0]);
        }
    }
    if (npos < 2)
        usage(argv[0]);

    char *end;
    int64_t n = strtoll(pos[0], &end, 10);
    if (*end || n <= 0)
        usage(argv[0]);
    const char *path = pos[1];
    uint64_t seed = npos > 2 ? strtoull(pos[2], NULL, 10) : 42;
    uint64_t maxv = npos > 3 ? strtoull(pos[3], NULL, 10) : 2147483647ULL;
    if (maxv == 0 || maxv > 2147483647ULL) {
        fprintf(stderr, "Invalid arguments (0 < max_value <= 2^31)\n");
        return 1;
    }

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {

        perror("open");
        return 1;
    }
    if (write_all(fd, &n, sizeof n, 0) != 0) {

        perror("write header");
        return 1;
    }

    const int64_t BLOCK = 1 << 20; /* 4 MB per write */
    int64_t nblocks = (n + BLOCK - 1) / BLOCK;
    volatile int failed = 0;
    double t0 = omp_get_wtime();

#pragma omp parallel
    {

        int32_t *buf = malloc(BLOCK * sizeof(int32_t));
        if (!buf) {

#pragma omp atomic write
            failed = 1;
        }
#pragma omp for schedule(dynamic)
        for (int64_t b = 0; b < nblocks; b++) {

            if (failed)
                continue;
            int64_t start = b * BLOCK;
            int64_t cnt = n - start < BLOCK ? n - start : BLOCK;
            gen_fill(buf, start, cnt, n, seed, maxv, mode);
            if (write_all(fd, buf, (size_t)cnt * sizeof(int32_t),
                          (off_t)(8 + start * 4)) != 0) {

#pragma omp atomic write
                failed = 1;
            }
        }
        free(buf);
    }
    if (failed) {

        fprintf(stderr, "write failed (disk full or out of memory?)\n");
        return 1;
    }
    if (close(fd) != 0) {

        perror("close");
        return 1;
    }

    double dt = omp_get_wtime() - t0;
    double gb = (8.0 + 4.0 * (double)n) / 1e9;
    printf("Wrote %" PRId64 " integers to %s (%.2f GB) in %.2f s = %.2f GB/s "
           "[%s, %d threads]\n",
           n, path, gb, dt, gb / (dt > 0 ? dt : 1e-9),
           mode == GEN_RANDOM ? "random"
                              : (mode == GEN_SORTED ? "sorted" : "reverse"),
           omp_get_max_threads());
    return 0;
}
