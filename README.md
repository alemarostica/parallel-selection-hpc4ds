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
  with very few distinct values. Use this by default.
- `rand`: pivot is the median of 15 random elements per rank. Cheap per round,
  about 12 rounds, only expected-case guarantees.
- `mom`: median of block medians. Deterministic and robust, but the slowest.

Results are identical in all modes; only the time changes.

## Examples

```bash
# median of 2e10 generated values on 16 ranks x 4 threads, verified
OMP_NUM_THREADS=4 mpirun -n 16 ./selection -g 20000000000 -p fr -d -V

# 10th percentile of a file
OMP_NUM_THREADS=4 mpirun -n 4 ./selection -p fr data.bin 100000000

# sorted data with rebalancing when a rank holds 1.5x the average
OMP_NUM_THREADS=4 mpirun -n 16 ./selection -g 20000000000 -m sorted -r 1.5 -p fr
```

## Memory

Per rank: 4 bytes x n/p for the data, plus about 0.15 bytes per element of
scratch with `mom` or `fr` (none with `rand`). Example: n = 2e10 on 16 ranks is
about 4.8 GB of data per rank. Choose the PBS `mem` request accordingly
(two ranks per node need about 10 GB).

## Benchmarks on the cluster

Job scripts in `bash-scripts/` run `mom`, `rand` and `fr` three times each on one
family of configurations; results are written to `bash-scripts/logs/` and
copied to `bash-scripts/results/`.

| Script | What it varies |
|---|---|
| `cfg_k.sh` | position of k |
| `cfg_dist.sh` | data distribution (sorted, reverse, duplicates, rebalancing) |
| `cfg_size.sh` | problem size |
| `cfg_scale.sh` | number of ranks (submit with `qsub -v RANKS=8 -l select=4:ncpus=8:mpiprocs=2:mem=16gb cfg_scale.sh`, then 16 and 32) |
| `cfg_sample.sh` | FR sample size (`-f`) |

Build once with `./submit.sh cfg_k.sh`, submit the others with `qsub`. Read a
result log with:

```bash
grep -E "^###|k-th smallest" logs/cfg_k.txt
```
