#!/bin/bash
# Usage: ./submit.sh <job.sh>
# Builds selection from the sources in the repo root, then submits the job.
set -e
[[ -r "$1" ]] || { echo "Usage: $0 <job.sh>" >&2; exit 1; }
SRC="$(cd "$(dirname "$0")/.." && pwd)"
module unload MPICH/3.4.2-GCC-10.3.0
module load MPICH/3.4.2-GCC-10.3.0
mpicc -O3 -march=native -fopenmp -I"$SRC" \
    "$SRC"/selection_big_gen.c "$SRC"/io_gen.c "$SRC"/kernels.c "$SRC"/pivot.c \
    -o selection
chmod +x mpi_wrapper.sh
qsub "$1"
