#!/bin/bash -l
#SBATCH -t 24:00:00
#SBATCH --mem=20G
#SBATCH --array=0-9
#SBTACH --gres=gpu:1 --constraint='pascal|volta'
#SBATCH -o sp_data/results/param_search_results.out

module load anaconda/2020-05-tf2

python main.py --param_set_search_number $SLURM_ARRAY_TASK_ID