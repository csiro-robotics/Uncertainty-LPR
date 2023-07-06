#!/bin/bash
#SBATCH --time=01:59:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

source /datasets/work/d61-eif/work/kni101/miniconda3/etc/profile.d/conda.sh # Replace with path to your conda
conda activate MinkLoc3D # Replace with name of your environment

_ROOT='/datasets/work/d61-eif/work/kni101/uncertainty_placerecognition/MinkLoc3D' # Replace with root of your MinkLoc3D 
_SAVEDIR="${_ROOT}/batch_jobs/oxford_baseline" # Replace with your save root 


python training/train.py  \
    --config config/eval_datasets/oxford_inhouse.yaml \
    data.dataset_folder '/datasets/work/d61-eif/source/incrementalPointClouds' \
    data.train_file "${_ROOT}/pickles/oxford/training_queries_baseline.pickle" \
    save_path $_SAVEDIR \
    # See config/default.yaml for a list of tunable hyperparameters!
    



