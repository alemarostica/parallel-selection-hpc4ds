#!/bin/bash
#PBS -N cfg_scale
#PBS -l select=8:ncpus=8:mpiprocs=2:mem=16gb
#PBS -l walltime=00:15:00
#PBS -l place=scatter
#PBS -q shortHPC4DS
#PBS -o logs/cfg_scale.txt
#PBS -e logs/cfg_scale_err.txt

cd $PBS_O_WORKDIR
mkdir -p logs
export OMP_NUM_THREADS=4
module load MPICH/3.4.2-GCC-10.3.0
source ./cfg_lib.sh

bench "ranks=$RANKS N=1e10" -g 10000000000
