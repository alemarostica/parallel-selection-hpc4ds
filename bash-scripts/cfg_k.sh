#!/bin/bash
#PBS -N cfg_k
#PBS -l select=8:ncpus=8:mpiprocs=2:mem=16gb
#PBS -l walltime=00:40:00
#PBS -l place=scatter
#PBS -q shortHPC4DS
#PBS -o logs/cfg_k.txt
#PBS -e logs/cfg_k_err.txt

cd $PBS_O_WORKDIR
mkdir -p logs
export OMP_NUM_THREADS=4
module load MPICH/3.4.2-GCC-10.3.0
source ./cfg_lib.sh

N=20000000000
bench "k=N/10"  -g $N 2000000000
bench "k=N/100" -g $N 200000000
bench "k=0"     -g $N 0
bench "k=N-1"   -g $N 19999999999
