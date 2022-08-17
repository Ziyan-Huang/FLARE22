# !/bin/bash -e
export RESULTS_FOLDER="$PWD/RESULTS_FOLDER/"
export nnUNet_raw_data_base="$PWD/nnUNet_raw_data_base/"
export nnUNet_preprocessed="$PWD/nnUNet_preprocessed/"

nnUNet_predict -i /workspace/inputs/  -o /workspace/outputs  -t 126  -p nnUNetPlansFLARE22Small   -m 3d_fullres \
 -tr nnUNetTrainerV2_S5_D2_W16_FLARE  -f all   --step_size 0.5 --mode fastest --disable_tta

