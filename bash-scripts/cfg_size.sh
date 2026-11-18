#!/bin/bash
#PBS -N cfg_size
#PBS -l select=8:ncpus=8:mpiprocs=2:mem=16gb
#PBS -l walltime=00:15:00
#PBS -l place=scatter
#PBS -q shortHPC4DS
#PBS -o logs/cfg_size.txt
#PBS -e logs/cfg_size_err.txt

cd $PBS_O_WORKDIR
mkdir -p logs
export OMP_NUM_THREADS=4
module load MPICH/3.4.2-GCC-10.3.0
source ./cfg_lib.sh

bench "N=2e8" -g 200000000
bench "N=2e9" -g 2000000000
