#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

source <path>/miniconda/etc/profile.d/conda.sh # Replace with path to your conda
conda activate minklocenv # Replace with name of your environment

echo "Trained On: $1";

_ROOT='<path>/Uncertainty-LPR' # Replace with root of your Uncertainty-LPR 
_SAVEDIR="${_ROOT}/weights/batch_jobs/batch_jobs_pfe_stun_dropout/dropout_minkloc_$1"

if [ "$1" == "oxford" ]; then
    trainfile="oxford/training_queries_oxford"
elif [ "$1" == "dcc" ]; then
    trainfile="MulRan/training_queries_dcc"
elif [ "$1" == "riverside" ]; then
    trainfile="MulRan/training_queries_riverside"
fi

export PYTHONPATH=$PYTHONPATH:$_ROOT

python $_ROOT/training/train.py  \
    --config $_ROOT/config/eval_datasets/minkloc3d_mcdropout.yaml \
    --uncertainty_method dropout \
    --teacher_net "${_ROOT}/weights/minkloc_$1.pth" \
    data.train_file "${_ROOT}/pickles/${trainfile}.pickle" \
    save_path $_SAVEDIR \

