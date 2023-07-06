#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --account=OD-227199

source /scratch2/gra595/miniconda/etc/profile.d/conda.sh # Replace with path to your conda
conda activate minklocenv # Replace with name of your environment

echo "Trained On: $1";

_ROOT='/datasets/work/d61-eif/work/gra595/uncertainty_placerecognition/MinkLoc3D' # Replace with root of your MinkLoc3D 
# _WEIGHTS="${_ROOT}/weights/pfe_stun_dropout/dropout_minkloc_$1.pth" # Eval provided models
_WEIGHTS="${_ROOT}/weights/batch_jobs/batch_jobs_pfe_stun_dropout/dropout_minkloc_$1/checkpoint_final.pth" # Eval trained models
_RESULTS="${_ROOT}/results/results_pfe_stun_dropout/dropout_minkloc_$1.csv" 

export PYTHONPATH=$PYTHONPATH:$_ROOT

python $_ROOT/eval/eval_wrapper.py \
    --config $_ROOT/config/eval_datasets/minkloc3d_mcdropout.yaml \
    --uncertainty_method dropout \
    --dropout_passes 5 \
    --weights $_WEIGHTS \
    save_results $_RESULTS \
