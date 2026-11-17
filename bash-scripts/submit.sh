#!/bin/bash
# Usage: ./submit.sh <source.c> <job.sh>
#
set -e 
[[ -r "$1" && -r "$2" ]] || { echo "Usage: $0 <source.c> <job.sh>" >&2; exit 1; }
module unload MPICH/3.4.2-GCC-10.3.0
module load MPICH/3.4.2-GCC-10.3.0
mpicc -O3 -march=native -fopenmp "$1" -o selection 
chmod +x mpi_wrapper.sh
qsub "$2"

