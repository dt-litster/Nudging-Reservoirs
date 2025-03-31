#!/bin/bash --login

#SBATCH --time=00:10:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=2048M   # memory per CPU core
#SBATCH -J "Torch_Reservoir"   # job name
#SBATCH --qos=test
#SBATCH --mail-user=dallin.seyfried@mathematics.byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load cuda
cd /nobackup/autodelete/usr/seyfdall/classes/513R/Nudging-Reservoirs/
conda activate torch_reservoir

python ./train_node.py $SAVE_EXT -w $WIDTH --num_epochs=$NUM_EPOCHS --save_every=$SAVE_EVERY --val_every=$NUM_EPOCHS -ws $WFN -lr $LR --weight_decay $WD --cuda