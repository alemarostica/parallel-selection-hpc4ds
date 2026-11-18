#!/bin/bash
#PBS -N cfg_sample
#PBS -l select=8:ncpus=8:mpiprocs=2:mem=16gb
#PBS -l walltime=00:30:00
#PBS -l place=scatter
#PBS -q shortHPC4DS
#PBS -o logs/cfg_sample.txt
#PBS -e logs/cfg_sample_err.txt

cd $PBS_O_WORKDIR
mkdir -p logs
export OMP_NUM_THREADS=4
module load MPICH/3.4.2-GCC-10.3.0

# FR sample-size sweep at N=2e10, k=N/2 (-f 0 is not allowed: "default" = no -f)
N=20000000000
for rep in 1 2 3; do
    echo "### sample=default | pivot=fr | rep=$rep"
    mpirun -n 16 ./mpi_wrapper.sh ./selection -p fr -d -V -g $N
    for f in 250000 500000 1000000 2000000 4000000 8000000; do
        echo "### sample=$f | pivot=fr | rep=$rep"
        mpirun -n 16 ./mpi_wrapper.sh ./selection -p fr -d -V -g $N -f $f
    done
    echo "### sample=- | pivot=rand | rep=$rep"
    mpirun -n 16 ./mpi_wrapper.sh ./selection -p rand -d -V -g $N
done
