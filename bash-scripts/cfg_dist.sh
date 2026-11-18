#!/bin/bash
#PBS -N cfg_dist
#PBS -l select=8:ncpus=8:mpiprocs=2:mem=16gb
#PBS -l walltime=00:40:00
#PBS -l place=scatter
#PBS -q shortHPC4DS
#PBS -o logs/cfg_dist.txt
#PBS -e logs/cfg_dist_err.txt

cd $PBS_O_WORKDIR
mkdir -p logs
export OMP_NUM_THREADS=4
module load MPICH/3.4.2-GCC-10.3.0
source ./cfg_lib.sh

N=20000000000
bench "sorted"            -g $N -m sorted
bench "reverse"           -g $N -m reverse
bench "sorted+rebalance"  -g $N -m sorted -r 1.5
bench "dup-1e6"           -g $N -M 1000000
bench "dup-10"            -g $N -M 10
