#!/bin/bash --login

#SBATCH --time=02:15:00   # walltime
#SBATCH --ntasks=15   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=2048M   # memory per CPU core
#SBATCH -J "Torch_Reservoir"   # job name
#SBATCH --mail-user=dallin.seyfried@mathematics.byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

cd /nobackup/autodelete/usr/seyfdall/classes/513R/Nudging-Reservoirs/
conda activate torch_reservoir

export MPICC=$(which mpicc)
mpirun -np 15 python3 -c "from visuals import nudging_reservoir_visuals_mpi_helper; nudging_reservoir_visuals_mpi_helper()"