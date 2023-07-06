#!/bin/bash
#SBATCH --time=01:59:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --account=OD-227199

source /scratch2/gra595/miniconda/etc/profile.d/conda.sh # Replace with path to your conda
conda activate minklocenv # Replace with name of your environment

echo "Architecture: $1, Trained On: $2, Model no.: $3";

_ROOT='/datasets/work/d61-eif/work/gra595/uncertainty_placerecognition/MinkLoc3D' # Replace with root of your MinkLoc3D 
# _WEIGHTS="${_ROOT}/weights/$1_$2.pth" # Eval provided models
_WEIGHTS="${_ROOT}/weights/batch_jobs/$1_$2_$3/checkpoint_final.pth" # Eval trained models
_RESULTS="${_ROOT}/results/results_standard/$1_$2_$3.csv" 

export PYTHONPATH=$PYTHONPATH:$_ROOT

python $_ROOT/eval/eval_wrapper.py \
    --config $_ROOT/config/eval_datasets/$1.yaml \
    --weights $_WEIGHTS \
    --save_results $_RESULTS \