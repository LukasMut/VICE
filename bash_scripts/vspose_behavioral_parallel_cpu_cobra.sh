#!/bin/bash -l
#SBATCH -a 0-19
#SBATCH -o ./job_%A_%a.out
#SBATCH -e ./job_%A_%a.err
#SBATCH -D ./
#SBATCH -J vspose_opt_behav_cpu
#SBATCH --time=24:00:00

#SBATCH --partition="medium"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=2
#SBATCH --cpus-per-task=80
#SBATCH --mem=100000

#SBATCH --mail-type=none
#SBATCH --mail-user=lmutt@rzg.mpg.de

module purge
module load gcc/8
module load impi/2019.9
module load anaconda/3/2020.02
module load pytorch/cpu/1.6.0

export OMP_NUM_THREADS=2
#export OMP_NUM_THREADS=1
export SLURM_HINT=multithread

TASK='odd_one_out'
MODALITY='behavioral/'
TR_DIR='./triplets/behavioral/'
LR="0.001"
DIM=100
T=500
BS=128
WS=300
SAMPLES=20
DEVICE='cpu'
RND_SEEDS=(0 1 2 3 4 5 6 7 8 9 10 21 22 23 24 25 26 27 29 42) 


echo "Started VSPoSE $SLURM_ARRAY_TASK_ID optimization at $(date)"

srun python ./train.py --task $TASK --modality $MODALITY --triplets_dir $TR_DIR --learning_rate $LR --embed_dim $DIM --batch_size $BS --epochs $T --window_size $WS --lambdas 4 5 6 7 8 9 10 --weight_decays 0.005 0.01 0.05 0.1 0.5 1.0 --k_samples $SAMPLES --device $DEVICE --rnd_seed ${RND_SEEDS[$SLURM_ARRAY_TASK_ID]}  >> vspose_opt_behav_parallel_cpu.out

echo "Finished VSPoSE $SLURM_ARRAY_TASK_ID optimization at $(date)"
