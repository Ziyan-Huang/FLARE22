# Solution of Team Blackbean for FLARE22 Challenge
**Revisiting nnU-Net for Iterative Pseudo Labeling and Efficient Sliding Window Inference** \
*Ziyan Huang, Haoyu Wang, Jin Ye, Jingqi Niu, Can Tu, Yuncheng Yang, Shiyi Du, Zhongying Deng, Lixu Gu, and Junjun He* \

Built upon [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet), this repository provides the solution of team blackbean for [MICCAI FLARE22](https://flare22.grand-challenge.org/) Challenge. The details of our method are described in our [paper](https://openreview.net/forum?id=FNMbe2vLvev). 

You can reproduce our method as follows step by step:

## Environments and Requirements:
Install nnU-Net [1] as below. You should meet the requirements of nnUNet, our method does not need any additional requirements. For more details, please refer to https://github.com/MIC-DKFZ/nnUNet
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

## 1. Training Big nnUNet for Pseudo Labeling
### 1.1. Copy the following files in this repo to your nnUNet environment.
```
FLARE22/nnunet/training/network_training/nnUNetTrainerV2_FLARE.py
FLARE22/nnunet/experiment_planning/experiment_planner_FLARE22Big.py
```
### 1.2. Prepare 50 Labeled Data of FLARE
Following nnUNet, give a TaskID (e.g. Task022) to the 50 labeled data and organize them folowing the requirement of nnUNet.

    nnUNet_raw_data_base/nnUNet_raw_data/Task022_FLARE22/
    ├── dataset.json
    ├── imagesTr
    ├── imagesTs
    └── labelsTr
### 1.3. Conduct automatic preprocessing using nnUNet.
Here we do not use the default setting.
```
nnUNet_plan_and_preprocess -t 22 -pl3d ExperimentPlanner3D_FLARE22Big -pl2d None
```
### 1.4. Training Big nnUNet by 5-fold Cross Validation
```
for FOLD in 0 1 2 3 4
do
nnUNet_train 3d_fullres nnUNetTrainerV2_FLARE_Big 22 $FOLD -p nnUNetPlansFLARE22Big
done
```
### 1.5. Generate Pseudo Labels for 2000 Unlabeled Data
```
nnUNet_predict -i INPUTS_FOLDER -o OUTPUTS_FOLDER  -t 22  -tr nnUNetTrainerV2_FLARE_Big  -m 3d_fullres  -p nnUNetPlansFLARE22Big  --all_in_gpu True 
```

### 1.6. Iteratively Train Models and Generate Pseudo Labels
- Give a new TaskID (e.g. Task023) and organize the 50 Labeled Data and 2000 Pseudo Labeled Data as above.
- Conduct automatic preprocessing using nnUNet as above.
  ```
  nnUNet_plan_and_preprocess -t 23 -pl3d ExperimentPlanner3D_FLARE22Big -pl2d None
  ```
- Training new big nnUNet by all training data instead of 5-fold.
  ```
  nnUNet_train 3d_fullres nnUNetTrainerV2_FLARE_Big 23 all -p nnUNetPlansFLARE22Big
  ```
- Generate new pseudo labels for 2000 unlabeled data.

## 2. Filter Low-quality Pseudo Labels
We compare Pseudo Labels in different rounds and filter out the labels with high variants.
```
select.ipynb
```

## 3. Train Small nnUNet 
### 3.1. Copy the following files in this repo to your nnUNet environment.
```
FLARE22/nnunet/training/network_training/nnUNetTrainerV2_FLARE.py
FLARE22/nnunet/experiment_planning/experiment_planner_FLARE22Small.py
```
### 3.2. Prepare 50 Labeled Data and 1924 Selected Pseudo Labeled Data of FLARE
Give a new TaskID (e.g. Task026) and organize the 50 Labeled Data and 1924 Pseudo Labeled Data as above.

### 3.3. Conduct automatic preprocessing using nnUNet
Here we use the plan designed for small nnUNet.
```
nnUNet_plan_and_preprocess -t 26 -pl3d ExperimentPlanner3D_FLARE22Small -pl2d None
```
### 3.4. Train small nnUNet on all training data
```
nnUNet_train 3d_fullres nnUNetTrainerV2_FLARE_Small 26 all -p nnUNetPlansFLARE22Small
```

## 4. Do Efficient Inference with Small nnUNet
We modify a lot of parts of nnunet source code for efficiency. Please make sure the code backup is done and then copy the whole repo to your nnunet environment.
```
nnUNet_predict -i INPUT_FOLDER  -o OUTPUT_FOLDER  -t 26  -p nnUNetPlansFLARE22Small   -m 3d_fullres \
 -tr nnUNetTrainerV2_FLARE_Small  -f all  --mode fastest --disable_tta
```

# Citations
If you find this repository useful, please consider citing our paper:
```
@incollection{huang2023revisiting,
  title={Revisiting nnU-Net for Iterative Pseudo Labeling and Efficient Sliding Window Inference},
  author={Huang, Ziyan and Wang, Haoyu and Ye, Jin and Niu, Jingqi and Tu, Can and Yang, Yuncheng and Du, Shiyi and Deng, Zhongying and Gu, Lixu and He, Junjun},
  booktitle={Fast and Low-Resource Semi-supervised Abdominal Organ Segmentation: MICCAI 2022 Challenge, FLARE 2022, Held in Conjunction with MICCAI 2022, Singapore, September 22, 2022, Proceedings},
  pages={178--189},
  year={2023},
  publisher={Springer}
}
```
