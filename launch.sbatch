#!/bin/bash
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 10
#SBATCH -C gpu
#SBATCH -t 00:30:00
#Precise source dir
module load esslurm cuda
srcdir=/global/homes/h/hbrunie/benchmarks/CuSolver
#Move to srcdir to be able to use relative path
cd $srcdir
srun -n 1 -c 10 ./getrf
