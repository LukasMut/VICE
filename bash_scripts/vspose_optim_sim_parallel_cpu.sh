#!/bin/bash -l

#SBATCH -a 0-19
#SBATCH -o ./job_%A_%a.out
#SBATCH -e ./job_%A_%a.err
#SBATCH -D ./
#SBATCH -J vspose_opt_sim_cpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=114000M
#SBATCH --mail-type=none
#SBATCH --mail-user=lmutt@rzg.mpg.de

export OMP_NUM_THREADS=1

module purge
module load gcc/8
module load impi/2019.9
module load anaconda/3/2020.02
module load pytorch/cpu/1.6.0

VERSION='variational'
TASK='odd_one_out'
MODALITY='synthetic/'
TR_DIR='./triplets/synthetic/sample_01/'
LR="0.001"
DIMS=(100 200)
T=300
BS=128
WS=100
W_DECAY="0.05"
SAMPLES=20
DEVICE='cpu'
RND_SEEDS=(0 1 2 3 4 5 6 7 8 9 10 21 22 23 24 25 26 27 29 42) 

for d in "${DIMS[@]}"; do

	echo "Started VSPoSE $d optimization at $(date)"
	srun python3 ./train.py --version $VERSION --task $TASK --modality $MODALITY --triplets_dir $TR_DIR --learning_rate $LR --embed_dim $d --batch_size $BS --n_models $SLURM_CPUS_PER_TASK --epochs $T --window_size $WS --k_samples $SAMPLES --plot_dims --device $DEVICE --rnd_seed ${RND_SEEDS[$SLURM_ARRAY_TASK_ID]}  >> vspose_opt_sim_parallel_cpu.out
	echo "Finished VSPoSE $d optimization at $(date)"
done
