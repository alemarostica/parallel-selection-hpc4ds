#!/bin/bash
#PBS -N sel_cmp
#PBS -l select=16:ncpus=16:mpiprocs=8:mem=72gb
#PBS -l walltime=00:30:00
#PBS -l place=scatter
#PBS -q shortHPC4DS
#PBS -o logs/sel_cmp.txt
#PBS -e logs/sel_cmp_err.txt

cd $PBS_O_WORKDIR
mkdir -p logs
export OMP_NUM_THREADS=2
module load MPICH/3.4.2-GCC-10.3.0

for rep in 1 2 3; do
    echo "### N=4e10 ranks=128 | rep=$rep"
	    mpirun -n 128 ./mpi_wrapper.sh ./selection -g 40000000000 -p fr -R -f 250000 -d -V
done

for rep in 1 2 3; do
    echo "### N=2.4e11 ranks=128 | rep=$rep"
    mpirun -n 128 ./mpi_wrapper.sh ./selection -g 240000000000 -p fr -R -f 250000 -d -V
done

