# parallel-selection-hpc4ds

Distributed k-th smallest element (selection) of a large array of 32-bit
integers, with MPI between ranks and OpenMP inside each rank. The data can be
read from a binary file or generated in memory.

`k` is 0-based: `k = 0` is the minimum, `k = n/2` (the default) is the median.

## Build

```bash
mpicc -O3 -march=native -fopenmp -o selection \
    selection_big_gen.c io_gen.c kernels.c pivot.c -lm
```

On the cluster, `submit.sh <job.sh>` loads MPICH, builds
`selection` and submits the job.

`-march=native` builds for the CPU of the machine where you compile (the login
node). If a job ever dies with an illegal instruction (exit code 132), build
without it.

## Run

```bash
# data from a file
OMP_NUM_THREADS=4 mpirun -n 16 ./selection [options] data.bin [k]

# data generated in memory (no file)
OMP_NUM_THREADS=4 mpirun -n 16 ./selection -g N [options] [k]
```

Output: the k-th smallest value, the number of rounds and the time of the
selection (generation or loading is timed separately).

### Input file

`int64` n, followed by n `int32` values (little endian). Create one with:

```bash
gcc -O3 -fopenmp -o gen_dataset gen_dataset.c
OMP_NUM_THREADS=8 ./gen_dataset <n> data.bin [seed] [max_value] [--mode random|sorted|reverse]
```

`-g N` with the same `-S`, `-M`, `-m` produces exactly the same values as
`gen_dataset`, so file and in-memory runs are comparable.

## Options

| Option | Meaning |
|---|---|
| `-p mom\|rand\|fr` | Pivot strategy (default `mom`), see below |
| `-f n` | With `-p fr`: sample size in elements (default `N^(2/3)/2`) |
| `-R` | With `-p fr`: read-only counting pass instead of the in-place swap partition (see below) |
| `-d` | Print one line per round (pivot, remaining size, imbalance, time); with `fr` also the sampling and partition timings |
| `-r RATIO` | Redistribute the survivors when `max_local / average_local > RATIO` (RATIO > 1; default: never) |
| `-s SIZE` | Stop reducing and gather on rank 0 when N <= SIZE (default: `min(n / log2 n, 4M)`) |
| `-g N` | Generate N elements instead of reading a file |
| `-S SEED` | Generator seed (default 42) |
| `-M MAXV` | Generated values are in `[0, MAXV)` (default 2^31-1, at most 2^31) |
| `-m MODE` | Generated data: `random` (default), `sorted`, `reverse` |
| `-v` | Verify against a sequential solve (file input only; rank 0 reads the whole file, so only for files that fit in one rank's memory) |
| `-V` | Verify with O(1) extra memory by counting elements below and equal to the answer; works at any size and with `-g` |

### Which `-p` to use

- `fr` (Floyd-Rivest): each round picks two bounds from a random sample so that
  the k-th element almost surely lies between them, and keeps only that part.
  Usually 2 rounds. Fastest in our benchmarks on all inputs tested except one
  with very few distinct values (10 distinct values: about as fast as `rand`).
  Use this by default.
- `rand`: pivot is the median of 15 random elements per rank. Cheap per round,
  about 12 rounds, only expected-case guarantees.
- `mom`: median of block medians. Deterministic and robust, but the slowest.

Results are identical in all modes; only the time changes.

### `-R`: read-only counting pass (with `-p fr`)

By default `fr` partitions the data in place: below `u` / inside `[u, v]` /
above `v`. With `-R` the first pass over each chunk only reads the data: it
counts the elements below `u` and copies the few elements inside `[u, v]` into
a side buffer; if k falls in that middle part (almost always) the buffer
becomes the new data. A chunk that would need more than `len/32 + 4096` slots
is partitioned in place instead, and if k misses the middle part the remaining
chunks are partitioned in place, so the result is always the same.

Measured on 16 ranks x 4 threads, N = 2e10 unless noted (medians of 3 runs unless noted):

| Case | in-place | `-R` |
|---|---|---|
| median (k = N/2) | 2.45 s (1 run) | 0.87 and 1.72 s (2 runs) |
| k = N/10 | 1.48 s | 1.18 s |
| many duplicates (1e6 distinct values) | 2.03 s | 0.91 s |
| small N (2e8, 2e9, median) | 0.041, 0.22 s | 0.021, 0.10 s |
| sorted, reverse | 0.63, 0.59 s | 0.76, 0.80 s |
| k = 0 | 0.88 s | 1.18 s |
| k = N-1 | 1.07 s | 0.94 s (about equal) |
| 10 distinct values | 3.2 s | 3.4 s |

`-R` costs about the same in every case (round 1 is a flat ~0.5 s of partition
time at N = 2e10), so it wins where the in-place pass is slow (middle k,
duplicates) and loses a little where the in-place pass is fast (ordered data,
k near 0). Cost: a side buffer of about 3% of the local data (see Memory).

## Examples

```bash
# median of 2e10 generated values on 16 ranks x 4 threads, verified
OMP_NUM_THREADS=4 mpirun -n 16 ./selection -g 20000000000 -p fr -d -V

# 10th percentile of a file
OMP_NUM_THREADS=4 mpirun -n 4 ./selection -p fr data.bin 100000000

# same, with the read-only counting pass
OMP_NUM_THREADS=4 mpirun -n 16 ./selection -g 20000000000 -p fr -R -d -V

# sorted data with rebalancing when a rank holds 1.5x the average
OMP_NUM_THREADS=4 mpirun -n 16 ./selection -g 20000000000 -m sorted -r 1.5 -p fr
```

## Memory

Per rank: 4 bytes x n/p for the data, plus about 0.15 bytes per element of
scratch with `mom` or `fr` (none with `rand`), plus about 3% of the data for
the side buffer with `-R`. Example: n = 2e10 on 16 ranks is about 4.8 GB of
data per rank. Choose the PBS `mem` request accordingly (two ranks per node
need about 10 GB; with 8 ranks, n = 2e10 does not fit in `mem=16gb` per node).

## Benchmarks on the cluster

Job scripts in `bash-scripts/` run `mom`, `rand`, `fr` and `fr -R` (labelled
`frR` in the logs) three times each on one family of configurations; results
are written to `bash-scripts/logs/` and copied to `bash-scripts/results/`.
The shared helper is `cfg_lib.sh` (`bench "<label>" <options>`).

| Script | What it varies |
|---|---|
| `cfg_k.sh` | position of k |
| `cfg_dist.sh` | data distribution (sorted, reverse, duplicates, rebalancing) |
| `cfg_size.sh` | problem size |
| `cfg_scale.sh` | number of ranks, see below |
| `cfg_sample.sh` | FR sample size (`-f`), with `-R`, plus `rand` as reference |
| `cfg_threads.sh` | split between ranks and OpenMP threads, `-f`, `-s` (see below) |
| `sel_cmp.sh` | N = 4e10 against N = 2.4e11 on 128 ranks, with `-V` |

Build once with `./submit.sh cfg_k.sh`, submit the others with `qsub`, one job
at a time (jobs running together can share nodes and disturb the timings).
Read a result log with:

```bash
grep -E "^###|k-th smallest" logs/cfg_k.txt
```

The scaling job takes the rank count from the command line; give each run its
own log, and set `select` to ranks / 2 (two ranks per chunk):

```bash
qsub -o logs/cfg_scale_8.txt  -e logs/cfg_scale_8_err.txt  -v RANKS=8  -l select=4:ncpus=8:mpiprocs=2:mem=16gb  cfg_scale.sh
qsub -o logs/cfg_scale_16.txt -e logs/cfg_scale_16_err.txt -v RANKS=16 -l select=8:ncpus=8:mpiprocs=2:mem=16gb  cfg_scale.sh
qsub -o logs/cfg_scale_32.txt -e logs/cfg_scale_32_err.txt -v RANKS=32 -l select=16:ncpus=8:mpiprocs=2:mem=16gb cfg_scale.sh
```

### Results so far (selection time, N = 2e10 unless noted, 16 ranks x 4 threads)

All problem sizes tried, selection time in seconds, median k (k = N/2), uniform
random data generated in memory. Medians of 3 runs unless marked with `*`
(one run) or noted; `-` means the mode was not run at that size.

| N | Data | Ranks x threads | mom | rand | fr | fr -R | Notes |
|---|---|---|---|---|---|---|---|
| 2e8 | 0.8 GB | 16 x 4 | 0.127 | 0.089 | 0.041 | **0.021** | `cfg_size.sh` |
| 2e9 | 8 GB | 16 x 4 | 1.125 | 0.482 | 0.223 | **0.096** | `cfg_size.sh` |
| 1e10 | 40 GB | 8 x 4 | 10.95 | 4.80 | 1.91 | **0.60** | `cfg_scale.sh` |
| 1e10 | 40 GB | 16 x 4 | 5.57 | 2.25 | 1.02 | **0.37** | `cfg_scale.sh` |
| 1e10 | 40 GB | 32 x 4 | 3.24 | 1.09 | 0.69 | **0.25** | `cfg_scale.sh` |
| 1e10 | 40 GB | 64 x 2 | 2.76 | 1.08 | 0.64 | **0.28** | `cfg_threads.sh`; 0.24 with `-f 250000` |
| 2e10 | 80 GB | 16 x 4 | 10.8 | 3.7 | 2.45* | **0.88** | 0.55 with `-f 250000` |
| 4e10 | 160 GB | 64 x 2 | - | - | - | **1.72*** | generated; from a file: 2.11* (loading it took 377 s) |
| 4e10 | 160 GB | 128 x 2 | - | - | - | **0.42** | 6 runs, 8 ranks per host |
| 2.4e11 | 960 GB | 128 x 2 | - | - | - | **4.79** | 6 runs, verified with `-V` |

On a single laptop-class machine (4 ranks x 2 threads, 800M integers, 3.2 GB),
with the serial `quickselect` as reference:

| N | serial | mom | rand | fr | fr -R |
|---|---|---|---|---|---|
| 8e8 | 18.0 | 9.74 | 3.08 | 1.54 | **0.60** |

The serial reference took 2.3 s at 1e8 and 8.1 s at 4e8 on the same machine
(about 20-23 ns per element). Ranks and thread layouts differ between rows, so
compare the modes within a row, not the rows with each other; the scaling
tables below compare layouts at a fixed size.

Mean over 9 configurations (4 values of k, 5 data distributions), 3 runs each:

| mom | rand | fr | fr -R |
|---|---|---|---|
| 12.0 s | 5.8 s | 1.36 s | 1.27 s |

Scaling, N = 1e10, k = N/2 (median of 3 runs, 2 ranks per node x 4 threads):

| Ranks | mom | rand | fr | fr -R |
|---|---|---|---|---|
| 8 | 10.95 s | 4.80 s | 1.91 s | 0.60 s |
| 16 | 5.57 s | 2.25 s | 1.02 s | 0.37 s |
| 32 | 3.24 s | 1.09 s | 0.69 s | 0.25 s |

From 8 to 32 ranks (ideal 4x): `mom` 3.4x, `rand` 4.4x, `fr` 2.8x,
`fr -R` 2.4x. The `fr` modes flatten because some costs do not shrink with the
rank count: the two quickselects on the sample (about 20 ms, repeated on every
rank), the sample `Allgatherv` and the final gather. At 32 ranks they are
about a third of the 0.25 s of `fr -R`; a smaller sample (`-f`) reduces this
(see the threads experiments below).

Ranks x threads, N = 1e10, k = N/2, same 128 cores (16 nodes x 8 cores), median
of 3 runs (`cfg_threads.sh`, no thread pinning, `place=scatter`):

| Ranks x threads | mom | rand | fr | fr -R | `-R` round 1 partition | `-R` sampling |
|---|---|---|---|---|---|---|
| 32 x 4 | 2.99 s | 1.11 s | 0.645 s | 0.318 s | 127 ms | 74 ms |
| 16 x 8 | 3.05 s | 1.00 s | 0.696 s | 0.339 s | 132 ms | 122 ms |
| 64 x 2 | 2.76 s | 1.08 s | 0.638 s | **0.283 s** | 125 ms | 66 ms |

More threads per rank did not help: the partition pass takes the same time in
all three layouts (125-132 ms with `-R`, about 510 ms for the in-place pass), so
its speed is set by the total number of cores (or by memory bandwidth), not by
how they are split. With fewer, fatter ranks the sampling gets slower, because
each rank draws a larger share of the sample on one thread (the draw is 20 ms
at 16 ranks and 3-4 ms at 64 ranks); the two quickselects stay at 20-26 ms per
rank. The best layout, 64 x 2, is about 11% faster than 32 x 4, which is within
what 3 runs can resolve.

Sample size and final gather, `fr -R`, 64 x 2, N = 1e10 (selection time):

| Variant | Runs | Median |
|---|---|---|
| default sample (2.3M) | 0.268, 0.307, 0.283 s | 0.283 s |
| `-f 250000`, first job | 1.494, 0.287, 0.223 s | 0.287 s |
| `-f 250000`, second job | 0.257, 0.214, 0.243 s | **0.243 s** |
| `-s 500000` (gather below 500k) | 0.456, 1.251, 0.259 s | 0.456 s |

`-f 250000` cuts the sampling cost from about 66 ms to about 16 ms (quickselect
25 -> 3 ms, `Allgatherv` 36 -> 4-12 ms) at the price of a slightly slower
round 2 and a larger final gather (3.6M instead of 1.9M elements). In the
second job it is about 14% faster than the default; in the first job one run
was disturbed by a slow collective (`Allgatherv` 486 ms), so treat the gain as
small but likely. `-s 500000` adds a third round and does not help.

Some runs show collectives that take 0.2-0.7 s instead of 5-40 ms. That jitter
is larger than anything the sample or gather tuning removes, and it did not
appear in a single-node run (4 ranks, spread under 3%), so it most likely comes
from the shared network. `place=pack` was not tested: in PBS it puts all
chunks on a single host, so it cannot work for a job that needs more cores than
one node has (the nodes of this queue have 28 to 96 cores).

Large run, N = 2.4e11 integers (960 GB generated in memory), `fr -R -f 250000`,
k = N/2, 128 ranks x 2 threads on 16 hosts with 8 ranks per host
(`select=16:ncpus=16:mpiprocs=8:mem=72gb`, `sel_cmp.sh`), compared with N = 4e10
on the same layout and in the same job (6 runs per size over two jobs, all with
the distributed check `-V` passing in the second job):

| N | Data per rank | Selection time (runs) | Median | Rounds | Round 1 partition |
|---|---|---|---|---|---|
| 4e10 (160 GB) | 312M elements | 0.339, 0.379, 0.395, 0.449, 0.476, 0.477 s | **0.42 s** | 3 | 350-359 ms |
| 2.4e11 (960 GB) | 1.875 billion elements | 4.10, 4.60, 4.68, 4.90, 4.98, 5.01 s | **4.79 s** | 3 | 2389-2401 ms |

The median found is 1073743833; the check counts 119,999,999,955 elements below
it and 96 equal to it, so rank k = 120,000,000,000 is inside. Generating the data
in memory takes 4.5-4.9 s at this size (0.7 s at 4e10). Memory per rank is about
7.2 GB of data plus 0.2 GB of scratch, about 66 GB per host.

6x more data costs about 10-11x more time: the partition pass is 6.8x slower
(about 25 GB/s per host in both sizes, so it looks memory-bound per host), and
the first sample `Allgatherv` takes 640-760 ms at 2.4e11 against 3-4 ms at
4e10, which is not explained. The medians of all runs are within one standard
deviation of the expected median of a uniform sample (about 1073741824 +- 2200
at this size). This run times the selection only; reading a file of this size
would take much longer than the selection.

Queue limits seen while preparing it: `shortHPC4DS` serves only 24 hosts
(six `c02`, `c03-n03`, `c11-n11..n18`, `c12-n01..n08` and `c12-n14`), so a job
with `place=scatter` and more than 24 chunks never starts (`Insufficient amount
of resource: queuelist`); use fewer hosts with more ranks per host.

Run-to-run noise on the cluster is large in some cells (up to a factor of two),
so compare modes within the same job.
