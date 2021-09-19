#!/bin/bash -l
#SBATCH --time=8:00:00
#SBATCH --mem=35G
#SBATCH --array=0-14
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='volta'

module load anaconda

python main.py --param_set_search_number $SLURM_ARRAY_TASK_ID --dropout 0 --batch_size 32 --run_name beam_search_ --train_cs_predictor --add_lg_info --patience 30 --test_beam