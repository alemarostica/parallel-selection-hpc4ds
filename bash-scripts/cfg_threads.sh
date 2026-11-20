#!/bin/bash
#PBS -N cfg_threads
#PBS -l select=16:ncpus=8:mpiprocs=2:mem=16gb
#PBS -l walltime=00:15:00
#PBS -l place=scatter
#PBS -q shortHPC4DS
#PBS -o logs/cfg_threads.txt
#PBS -e logs/cfg_threads_err.txt

cd $PBS_O_WORKDIR
mkdir -p logs
export OMP_NUM_THREADS=${THREADS:-4}
[ -n "$BIND" ] && export OMP_PROC_BIND=close OMP_PLACES=cores
module load MPICH/3.4.2-GCC-10.3.0
export RANKS=${RANKS:-32}
source ./cfg_lib.sh

EXTRA=""
[ -n "$FSAMPLE" ] && EXTRA="$EXTRA -f $FSAMPLE"
[ -n "$SWITCH" ]  && EXTRA="$EXTRA -s $SWITCH"
bench "ranks=$RANKS threads=$OMP_NUM_THREADS extra=$EXTRA N=1e10" -g 10000000000 $EXTRA
