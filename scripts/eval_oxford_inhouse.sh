source /datasets/work/d61-eif/work/kni101/miniconda3/etc/profile.d/conda.sh # Replace with path to your conda
conda activate MinkLoc3D # Replace with name of your environment

_ROOT='/datasets/work/d61-eif/work/kni101/uncertainty_placerecognition/MinkLoc3D' # Replace with root of your MinkLoc3D 
_WEIGHTS="${_ROOT}/weights/minkloc3d_baseline.pth" # Replace with your checkpoint

python eval/eval_wrapper.py \
    --config config/eval_datasets/oxford_inhouse.yaml \
    --weights $_WEIGHTS \
    --save_results None # Replace with path to csv you want to save results to if so desired

