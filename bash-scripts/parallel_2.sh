#!/bin/bash

#PBS -N parallel_selection
#PBS -l select=8:ncpus=8:mpiprocs=2:mem=16gb
#PBS -l walltime=0:40:00
#PBS -l place=scatter
#PBS -q shortHPC4DS
#PBS -o logs/out.txt
#PBS -e logs/err.txt

cd $PBS_O_WORKDIR
mkdir -p logs
export OMP_NUM_THREADS=4
module load MPICH/3.4.2-GCC-10.3.0

mpirun -n 16 ./mpi_wrapper.sh ./selection -g 20000000000 -p mom -d -V

