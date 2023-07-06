#!/bin/bash
#SBATCH --time=01:59:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

source /home/gra595/miniconda3/etc/profile.d/conda.sh # Replace with path to your conda
conda activate minkenv # Replace with name of your environment

_ROOT='/home/gra595/Documents/uncertainty-internship/uncertainty_placerecognition/MinkLoc3D' # Replace with root of your MinkLoc3D 
_SAVEDIR="${_ROOT}/batch_jobs/oxford_baseline" # Replace with your save root 


python $_ROOT/training/train_STUN.py  \
    --config $_ROOT/config/eval_datasets/oxford_inhouse.yaml \
    data.dataset_folder "${_ROOT}/dataset_folder/incrementalPointClouds" \
    data.train_file "${_ROOT}/pickles/oxford/training_queries_baseline.pickle" \
    save_path $_SAVEDIR \
    # See config/default.yaml for a list of tunable hyperparameters!
    



