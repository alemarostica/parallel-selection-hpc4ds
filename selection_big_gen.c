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
 * Build: mpicc -O3 -march=native -fopenmp -o selection selection_big_gen.c \
 *              io_gen.c kernels.c pivot.c
 *
 * TODO: ragionare i launch parameters and shit
 * Run:   OMP_NUM_THREADS=4 mpirun -n 2 ./selection [options] data.bin [k]
 *        OMP_NUM_THREADS=4 mpirun -n 2 ./selection -g N [options] [k]
              (no file,: every ranks generates its own slice of N elements)
 *   -p mom|rand|fr pivot strategy (default mom)
 *   -f n        with -p fr: sample size in elements (default: N^(2/3)/2)
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
 *               and == answer. Works at any size, unlike -v.
 *
 * Memory per rank: 4 bytes x n/p for the data, plus 4/log2(n) bytes x n/p of
 * scratch with -p mom (about 0.15 bytes per element; non with -p rand), plus
 * the MPI/OpenMP runtime (~25-100 MB). Rank 0 also needs up to 16 MB to
 * gather the last survivors. Rebalancing (-r) temporarily needs one more
 * block of N/p elements. -v needs 4 bytes x n
 * on rank 0: do not use it on huge files, use -V.
 *
 * Launch: keep (ranks per node) x (OMP_NUM_THREADS) <= cores of the node.
 */
#include "selection.h"

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
                             int64_t N, int64_t *moved)
{

    int64_t base = N / p, rem = N % p;
    int64_t *tgt = xmalloc(2 * p * sizeof(int64_t)), *need = tgt + p;
    int *sc = xmalloc(4 * p * sizeof(int)), *sd = sc + p, *rc = sd + p,
        *rd = rc + p;

    /* the `rem` ranks holding the most get base + 1 (ties: lower rank) */
    for (int r = 0; r < p; r++)
    {

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
    for (int i = 0; i < p; i++)
    {

        int64_t sur = -need[i];
        while (sur > 0)
        {

            while (need[j] <= 0)
                j++; /* total surplus == total deficit, so j stays < p */
            int64_t amt = sur < need[j] ? sur : need[j];
            if (i == me)
            {

                if (amt > INT_MAX || tgt[me] + sent > INT_MAX)
                    suicide("rebalance message exceeds INT_MAX elements");
                sc[j] = (int)amt;
                sd[j] = (int)(tgt[me] + sent); /* tail of my data */
                sent += amt;
            }
            if (j == me)
            {

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

    if (recvd > 0)
    { /* only deficit ranks grow, and they send nothing */
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
    if (sent > 0 && tgt[me] > 0)
    { /* surplus rank: give the tail back*/

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
                          Stats *st)
{

    const int p = arr->size, me = arr->rank;
    int32_t *data = arr->data;
    int64_t N = arr->global_n;
    const int T = omp_get_max_threads();

    int64_t switch_size = opt->switch_size;
    if (switch_size <= 0)
    {

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
    if (opt->pivot != PIVOT_RANDOM)
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
    /* FR is switched off for good if a round does not shrink the problem
     * (e.g. many duplicates); MoM then guarantees progress. */
    int fr_active = (opt->pivot == PIVOT_FR);
    while (N > switch_size)
    {

        double t0 = MPI_Wtime();
        const int64_t N_before = N;
        const int C = w.C;

        /* pivot: FR picks two bounds u <= v from a sample; the other modes
         * pick one pivot (everyone proposes a candidate, all pick the same
         * weighted median) and use u = v = pivot, so the code below handles
         * all modes. */
        int B = block_size(N);
        int32_t u, v;
        double tm[4] = {0, 0, 0, 0}; /* FR timings, see fr_bounds */
        const int fr_round = fr_active; /* the guard below may switch it off */
        if (fr_active)
            fr_bounds(data, &w, counts, N, k, p, me, &rng, opt->fr_sample, tm, &u, &v);
        else
        {
            int64_t mine[2] = {0, w.active};
            if (w.active > 0)
                mine[0] = (opt->pivot != PIVOT_RANDOM)
                              ? pivot_mom(data, &w, B, scratch)
                              : pivot_random(data, &w, &rng);
            MPI_Allgather(mine, 2, MPI_INT64_T, vw, 2, MPI_INT64_T, MPI_COMM_WORLD);
            u = v = weighted_median(vw, order, p, N);
        }
        const int32_t pivot = u; /* shown in the diagnostics */

        /* three-way partition: below u / inside [u, v] / above v
         * (for u == v: below / equal / above the pivot), one chunk per thread */
        double t_part = MPI_Wtime();
#pragma omp parallel for schedule(static)
        for (int c = 0; c < C; c++)
        {
            if (fr_active)
                partition_bracket(data, w.lo[c], w.hi[c], u, v, &w.lt[c], &w.gt[c]);
            else
                partition3(data, w.lo[c], w.hi[c], pivot, &w.lt[c], &w.gt[c]);
        }
        t_part = MPI_Wtime() - t_part;

        int64_t loc[3] = {0, 0, 0};
        for (int c = 0; c < C; c++)
        {

            loc[0] += w.lt[c] - w.lo[c];
            loc[1] += w.gt[c] - w.lt[c];
            loc[2] += w.hi[c] - w.gt[c];
        }
        MPI_Allgather(loc, 3, MPI_INT64_T, tri, 3, MPI_INT64_T, MPI_COMM_WORLD);
        int64_t L = 0, E = 0, G = 0;
        for (int r = 0; r < p; r++)
        {

            L += tri[3 * r];
            E += tri[3 * r + 1];
            G += tri[3 * r + 2];
        }
        /* everything fell inside [u, v] and u < v: FR cannot make progress on
         * this input, so switch to MoM from the next round on */
        if (fr_active && u < v && E == N_before)
            fr_active = 0;

        /* three outcomes: k in the part below u, inside [u, v], or above v.
         * L, E, G are the sizes of those parts (as before when u == v).
         * Every rank sees the same L, E, G and k -> same branch everywhere */
        int keep_less = 0, keep_mid = 0;
        if (k < L)
            keep_less = 1;
        else if (k < L + E)
        {
            if (u == v)
            {
                /* the middle part is all equal to u, so u is the answer */
                result = pivot;
                found = 1;
                st->rounds++;
                if (opt->verbose && me == 0)
                    printf(" round %2d: N=%" PRId64 " pivot=%d L=%" PRId64 " E=%" PRId64 " G=%" PRId64 " -> pivot is the answer\n", st->rounds, N_before, pivot, L, E, G);
                break;
            }
            /* keep only [u, v]; k becomes relative to this part */
            keep_mid = 1;
            k -= L;
        }
        else
        {
            k -= L + E;
        }
        /* narrow each chunk's window to the kept part: no data is moved */
        for (int c = 0; c < C; c++)
        {
            if (keep_mid)
            {
                w.lo[c] = w.lt[c];
                w.hi[c] = w.gt[c];
            }
            else if (keep_less)
                w.hi[c] = w.lt[c];
            else
                w.lo[c] = w.gt[c];
        }
        for (int r = 0; r < p; r++)
            counts[r] = keep_mid ? tri[3 * r + 1] : keep_less ? tri[3 * r]
                                                              : tri[3 * r + 2];
        N = keep_mid ? E : keep_less ? L
                                     : G;
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
            imbalance > opt->rebalance_ratio)
        {

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
                   st->rounds, N_before, B, pivot, keep_mid ? "M" : keep_less ? "L"
                                                                              : "R",
                   N,
                   100.0 * (double)N / (double)N_before, imbalance, rb,
                   1e3 * (MPI_Wtime() - t0));
        if (opt->verbose && me == 0 && fr_round)
            printf("           fr: s=%.0f draw %.0f ms, allgather %.0f ms, "
                   "select %.0f ms, partition %.0f ms\n",
                   tm[3], 1e3 * tm[0], 1e3 * tm[1], 1e3 * tm[2], 1e3 * t_part);
    }

    if (!found)
    {

        /* few elements are left: gather them on rank 0 and solve directly
         * (quickselect instead of sorting: same answer, linear time). */
        pack_windows(data, &w);
        int *rc = xmalloc(2 * p * sizeof(int)), *rd = rc + p;
        int64_t s = 0;
        for (int r = 0; r < p; r++)
        {

            rc[r] = (int)counts[r];
            rd[r] = (int)s;
            s += counts[r];
        }
        int32_t *all = (me == 0) ? xmalloc(N * sizeof(int32_t)) : NULL;
        MPI_Gatherv(data, (int)w.active, MPI_INT32_T, all, rc, rd, MPI_INT32_T,
                    0, MPI_COMM_WORLD);
        if (me == 0)
        {

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

static int verify(const char *path, int64_t k, int32_t got)
{

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
                         const Options *opt, int64_t k, int32_t got)
{

    const int64_t CH = 1 << 22;
    int32_t *buf = xmalloc(CH * sizeof(int32_t));
    MPI_File fh = MPI_FILE_NULL;
    if (!opt->gen_n && MPI_File_open(MPI_COMM_WORLD, path, MPI_MODE_RDONLY,
                                     MPI_INFO_NULL, &fh) != MPI_SUCCESS)
        suicide("verify: cannot reopen the file");

    int64_t less = 0, equal = 0;
    for (int64_t done = 0; done < arr->local_n;)
    {

        int64_t c = arr->local_n - done < CH ? arr->local_n - done : CH;
        if (opt->gen_n)
            gen_fill_parallel(buf, arr->offset + done, c, arr->global_n, opt);
        else
            MPI_File_read_at(fh,
                             HEADER_BYTES + (MPI_Offset)(arr->offset + done) *
                                                (MPI_Offset)sizeof(int32_t),
                             buf, (int)c, MPI_INT32_T, MPI_STATUS_IGNORE);
#pragma omp parallel for reduction(+ : less, equal) schedule(static)
        for (int64_t i = 0; i < c; i++)
        {

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

int main(int argc, char **argv)
{

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
    while ((c = getopt(argc, argv, "dvVr:s:p:g:S:M:m:f:")) != -1)
    {
        switch (c)
        {
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
            else if (strcmp(optarg, "fr") == 0)
                opt.pivot = PIVOT_FR;
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
        case 'f':
            opt.fr_sample = strtoll(optarg, NULL, 10);
            if (opt.fr_sample <= 0)
                opterr_flag = 1;
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
            break;
        default:
            opterr_flag = 1;
        }
    }

    const int gen = opt.gen_n > 0;
    const int nfiles = gen ? 0 : 1; /* positional args: [file] [k] */
    if (opterr_flag || optind + nfiles > argc ||
        (opt.rebalance_ratio != 0 && opt.rebalance_ratio <= 1.0) ||
        opt.maxv == 0 || opt.maxv > 2147483648ULL || (gen && do_verify))
    {

        if (rank == 0)

            fprintf(stderr,
                    "Usage: %s [-p mom|rand|fr] [-d] [-v] [-r ratio>1] "
                    "[-s switch_size] [-f fr_sample] <data_file> [k]\n"
                    "       %s -g N [-S seed] [-M maxv] [-m "
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

    if (rank == 0)
    {

        double data_mb = (double)arr.local_n * 4 / 1048576.0;
        double scratch_mb =
            opt.pivot != PIVOT_RANDOM
                ? (double)arr.local_n / block_size(arr.global_n) * 4 / 1048576.0
                : 0.0;
        printf("n=%" PRId64 "  ranks=%d  threads/rank=%d  pivot=%s  k=%" PRId64
               "  %s=%.3f s\n",
               arr.global_n, arr.size, omp_get_max_threads(),
               (const char *[]){"mom", "rand", "fr"}[opt.pivot], k,
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
    if (rank == 0)
    {

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
