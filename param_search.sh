#!/bin/bash -l
#SBATCH --time=5:00:00
#SBATCH --mem=35G
#SBATCH --array=0-71
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out_folder/p_search.out

module load anaconda

python main.py --param_set_search_number $SLURM_ARRAY_TASK_ID --batch_size 32 --run_name param_search_w_nl_nh --train_cs_predictor --add_lg_info