#!/bin/bash

#PBS -N parallel_selection
#PBS -l select=2:ncpus=2:mpiprocs=2:mem=4gb
#PBS -l walltime=0:10:00
#PBS -l place=scatter
#PBS -q shortHPC4DS
#PBS -o out.txt
#PBS -e err.txt

cd $PBS_O_WORKDIR

export OMP_NUM_THREADS=1
module load MPICH/3.4.2-GCC-10.3.0

mpirun -n 4 ./mpi_wrapper.sh ./parallel.out ./data.bin 

