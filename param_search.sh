#!/bin/bash -l
#SBATCH -t 5:00:00
#SBATCH --mem=20G
#SBATCH --array=0-35
#SBATCH --partition=gpu
#SBTACH --gres=gpu:1:teslap100
#SBATCH -c 4
module load anaconda

python main.py --param_set_search_number $SLURM_ARRAY_TASK_ID --batch_size 32 --run_name param_search --train_cs_predictor --add_lg_info