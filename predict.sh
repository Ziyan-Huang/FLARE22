# !/bin/bash -e
export RESULTS_FOLDER="$PWD/RESULTS_FOLDER/"
export nnUNet_raw_data_base="$PWD/nnUNet_raw_data_base/"
export nnUNet_preprocessed="$PWD/nnUNet_preprocessed/"

nnUNet_predict -i .//inputs/  -o ./outputs  -t 26  -p nnUNetPlansFLARE22Small   -m 3d_fullres \
 -tr nnUNetTrainerV2_FLARE_Small  -f all  --mode fastest --disable_tta

