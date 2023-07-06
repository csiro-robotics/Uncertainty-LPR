#!/bin/bash
#SBATCH --time=08:00:00
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
_SAVEDIR="${_ROOT}/weights/batch_jobs/$1_$2_$3" 
_RESULTS="${_ROOT}/results/results_standard/$1_$2_$3.csv" 

if [ "$2" == "oxford" ]; then
    trainfile="oxford/training_queries_oxford"
elif [ "$2" == "dcc" ]; then
    trainfile="MulRan/training_queries_dcc"
elif [ "$2" == "riverside" ]; then
    trainfile="MulRan/training_queries_riverside"
fi

export PYTHONPATH=$PYTHONPATH:$_ROOT

python $_ROOT/training/train.py  \
    --config $_ROOT/config/eval_datasets/$1.yaml \
    data.train_file "${_ROOT}/pickles/${trainfile}.pickle" \
    save_path $_SAVEDIR \
    save_results $_RESULTS \
    

