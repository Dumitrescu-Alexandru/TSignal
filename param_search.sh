#!/bin/bash -l
#SBATCH --time=8:00:00
#SBATCH --mem=35G
#SBATCH --array=0-35
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='volta'

module load anaconda/2020-05-tf2

python main.py --train_cs_predictor --param_set_search_number $SLURM_ARRAY_TASK_ID --run_name random_folds_run --lr 0.00001 --batch_size 32 --simplified --train_folds 0 1 --nlayers 3 --nheads 16  --add_lg_info --very_simplified --patience 50 --validate_on_mcc --tune_bert --frozen_epochs 3 --dropout 0.1 --train_only_decoder --random_folds
