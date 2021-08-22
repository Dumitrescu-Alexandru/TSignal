#!/bin/bash -l
#SBATCH --time=5:00:00
#SBATCH --mem=35G
#SBATCH --array=0-35
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module load anaconda

python main.py --param_set_search_number $SLURM_ARRAY_TASK_ID --batch_size 32 --run_name param_search --train_cs_predictor --add_lg_info