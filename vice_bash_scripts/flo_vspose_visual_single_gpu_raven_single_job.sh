#!/bin/bash -l

#SBATCH -o ./job_%j.out
#SBATCH -e ./job_%j.err
#SBATCH -D ./
#SBATCH -J vspose_dnn_similarities
#
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#
# NOTE: change *ntasks-per-core* and *OMP_NUM_THREADS* to 2, iff HT is enabled on compute node
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --mail-type=none
#SBATCH --mail-user=fmahner@rzg.mpg.de

module purge
module load intel/21.2.0
module load impi/2021.2
module load cuda/11.2
module load anaconda/3/2020.02
module load pytorch/gpu-cuda-11.2/1.8.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

TR_DIR='/u/fmahner/triplets6'
DIM=100
BS=256
WINDOW=200
DEVICE='cuda'
RND_SEED=42
PRIOR='gaussian'
K=5
BURNIN=500
MCSAMPLES=5
STEPS=30
ETA=0.001
EPOCHS=2000

echo "Started VSPoSE $SLURM_ARRAY_TASK_ID optimization at $(date)"

srun python3 ./main.py --triplets_dir $TR_DIR --k $K --eta $ETA --spike 0.25 --slab 1 --pi 0.5 --init_dim $DIM --batch_size $BS --epochs $EPOCHS --ws $WINDOW --mc_samples $MCSAMPLES --steps $STEPS --device $DEVICE --rnd_seed $RND_SEED --burnin $BURNIN >> vspose_vision_dnn.out

echo "Finished VSPoSE $SLURM_ARRAY_TASK_ID optimization at $(date)"