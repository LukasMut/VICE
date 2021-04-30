#!/bin/bash -l

#SBATCH -a 0-19
#SBATCH -o ./job_%A_%a.out
#SBATCH -e ./job_%A_%a.err
#SBATCH -D ./
#SBATCH -J vspose_behavioral_data
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# NOTE: change *ntasks-per-core to 2, iff you want to enable HT

#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=230000M
#SBATCH --mail-type=none
#SBATCH --mail-user=lmutt@rzg.mpg.de

module purge
module load gcc/10
module load impi/2021.2
module load anaconda/3/2020.02
module load pytorch/cpu/1.7.0

export OMP_NUM_THREADS=1
#export SLURM_HINT=multithread

TASK='odd_one_out'
MODALITY='behavioral/'
TR_DIR='./triplets/behavioral/'
LR="0.001"
DIM=100
T=1000
BS=128
WS=900
SAMPLES=20
DEVICE='cpu'
RND_SEEDS=(0 1 2 3 4 5 6 7 8 9 10 21 22 23 24 25 26 27 29 42)

echo "Started VSPoSE $SLURM_ARRAY_TASK_ID optimization at $(date)"

srun python3 ./train.py --task $TASK --modality $MODALITY --triplets_dir $TR_DIR --learning_rate $LR --embed_dim $DIM --batch_size $BS --epochs $T --window_size $WS --lambdas 3 4 5 6 7 8 9 10 11 --weight_decays 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 --k_samples $SAMPLES --device $DEVICE --rnd_seed ${RND_SEEDS[$SLURM_ARRAY_TASK_ID]}  >> vspose_behavioral_parallel.out

echo "Finished VSPoSE $SLURM_ARRAY_TASK_ID optimization at $(date)"
