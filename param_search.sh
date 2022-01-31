#!/bin/bash -l
#SBATCH --time=8:00:00
#SBATCH --mem=35G
#SBATCH --array=0-53
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='volta'

module load anaconda

python main.py --param_set_search_number $SLURM_ARRAY_TASK_ID --batch_size 32 --run_name data_perc_runs --train_cs_predictor --add_lg_info --patience 50 --simplified --very_simplified --validate_on_mcc  --tune_bert --frozen_epochs 15