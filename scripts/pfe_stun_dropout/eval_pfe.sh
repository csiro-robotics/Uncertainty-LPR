#!/bin/bash
#SBATCH --time=01:59:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --account=OD-227199

source <path>/miniconda/etc/profile.d/conda.sh # Replace with path to your conda
conda activate minklocenv # Replace with name of your environment

echo "Trained On: $1";

_ROOT='<path>/Uncertainty-LPR' # Replace with root of your Uncertainty-LPR 
# _WEIGHTS="${_ROOT}/weights/pfe_stun_dropout/pfe_minkloc_$1.pth" # Eval provided models
_WEIGHTS="${_ROOT}/weights/batch_jobs/batch_jobs_pfe_stun_dropout/pfe_minkloc_$1/checkpoint_final.pth" # Eval trained models
_RESULTS="${_ROOT}/results/results_pfe_stun_dropout/pfe_minkloc_$1.csv" 

export PYTHONPATH=$PYTHONPATH:$_ROOT

python $_ROOT/eval/eval_wrapper.py \
    --config $_ROOT/config/eval_datasets/minkloc3d_pfe.yaml \
    --uncertainty_method pfe \
    --weights $_WEIGHTS \
    save_results $_RESULTS \

