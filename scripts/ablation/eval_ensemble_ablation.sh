#!/bin/bash
#SBATCH --time=05:59:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

source <path>/miniconda/etc/profile.d/conda.sh # Replace with path to your conda
conda activate minklocenv # Replace with name of your environment

echo "Architecture: $1, Trained On: $2, No. Models: $3";

_ROOT='<path>/Uncertainty-LPR' # Replace with root of your Uncertainty-LPR 
_WEIGHTS="${_ROOT}/weights/batch_jobs/$1_$2_1/checkpoint_final.pth" 
_RESULTS="${_ROOT}/results/results_ablation/ens_$1_$2_$3.csv" 

export PYTHONPATH=$PYTHONPATH:$_ROOT

python $_ROOT/eval/eval_wrapper.py \
    --config $_ROOT/config/eval_datasets/$1.yaml \
    --uncertainty_method ensemble \
    --ensemble_models $3 \
    --weights $_WEIGHTS \
    --save_results $_RESULTS \

