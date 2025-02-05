#!/bin/bash

#SBATCH --job-name=C5
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu

#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8G
#SBATCH -c10
#SBATCH -p gpu --qos=day
#SBATCH -p gpu --qos=short

###############SBATCH -p gpu --qos=short
#############SBATCH -C "gmem12&infiniband"

################SBATCH -C gmemT48
#############SBATCH --gres=gpu:turing:2
##############SBATCH --exclude=c4-2

# srun --pty -p gpu --qos short --gres=gpu:2 -c16 -C gmem48 bash
# srun --pty -p gpu --qos short --gres=gpu:1 -c16 bash

nvidia-smi
nvidia-smi -q |grep -i serial

source ~/.bashrc
CONDA_BASE=$(conda info --base) ; 
source $CONDA_BASE/etc/profile.d/conda.sh

echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}
echo $SLURM_JOB_ID $SLURM_JOB_NODELIST
echo $CONDA_DEFAULT_ENV

scontrol write batch_script $SLURM_JOB_ID
rsync -a slurm-$SLURM_JOB_ID.sh ucf2:~/RLQ-CGAL-UBD/outputs/

cd ~/RLQ-CGAL-UBD/
# conda activate pathak 
conda activate ~/mambaforge/envs/rql 
# conda activate rlq

celeb=/home/c3-0/datasets/ID-Dataset/Celeb-reID/

last=/home/c3-0/datasets/LaST/
deepchange=/home/c3-0/datasets/DeepChange/

prcc=/home/c3-0/datasets/PRCC/prcc/
prcc_sil=/home/c3-0/datasets/ID-Dataset/masks/PRCC/jpgs


ltcc=/home/c3-0/datasets/LTCC/
ltcc_sil=/home/c3-0/datasets/ID-Dataset/ltcc/
ltcc_bkd=/home/c3-0/datasets/ID-Dataset/ltcc/background/
ltcc_pose=/home/c3-0/datasets/ID-Dataset/AlphaPose/ltcc/2DPose/
ltcc_gender=Scripts/Helper/LTCC_Gender.csv

Celeb_Wt_KL=logs/celeb/B=40_KL_4/best_model.pth.tar
R_LA_15_2_ABS_GID=logs/celeb_cc_colors/R_LA_15_2_ABS_GID/best_model.pth.tar

ROOT=$ltcc
DATASET=ltcc_cc_gender
SIL=$ltcc_sil
POSE=$ltcc_pose
GENDER=$ltcc_gender
DATASET_COLORS=ltcc_colors
DATASET_ORIG=ltcc


ntu=/home/c3-0/datasets/ID-Dataset/ntu/RGB/
NTU_Wt_KL=logs/BM_NTU/best_model.pth.tar
NTU_Wt_KL=logs/BM_NTU_Tight/best_model.pth.tar


# ROOT=$prcc
# DATASET=prcc_cc_gender
# SIL=$prcc_sil
# POSE=$prcc_pose
# GENDER=$prcc_gender
# DATASET_COLORS=prcc_colors
# DATASET_ORIG=prcc


GPUS=0,1
NUM_GPU=2
RUN_NO=1
PORT=12352



# # BaseModel 
# PORT=12371
# BATCH_SIZE=28
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
#     --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" --additional_loss 'kl_o_oid' --seed=$RUN_NO
# # ==> Best Rank-1 42.1%, achieved at epoch 40. Best MaP 20.9%

# BaseModel 
# PORT=12371
# BATCH_SIZE=28
# RUN_NO=4
# checkpoint=ltcc_colors/scratch_image/best_model.pth.tar
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
#     --tag BM_28_4_LTCC --resume $checkpoint --eval 
# rsync -a ucf0:~/RLQ-CGAL-UBD/BM_28_4_LTCC.pkl ./




################### RLQ (Celeb ReID Base Model)  [NTU Teacher]
# R_LA_15 # 15 clusters
PORT=12348
BATCH_SIZE=32
RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $NTU_Wt_KL --teacher_dataset ntu_colors --teacher_dir $ntu --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save >> outputs/$DATASET_ORIG'_'RLQ-R_LA_15-NTU-UCF.txt

# R_LA_25 # 25 clusters
# BATCH_SIZE=32
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_25_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $NTU_Wt_KL --teacher_dataset ntu_colors --teacher_dir $ntu --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=26 --Pose=$POSE --pose-mode="R_LA_25" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save >> outputs/$DATASET_ORIG'_'RLQ-R_LA_25-NTU-UCF.txt

# ##################### Only UBD    
# BATCH_SIZE=40
# RUN_NO=1
# PORT=12353
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
#     --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
#     --teacher_wt $NTU_Wt_KL --teacher_dataset ntu_colors --teacher_dir $ntu \
#     --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific >> outputs/$DATASET_ORIG'_'UBD-NTU-UCF-$BATCH_SIZE-$RUN_NO.txt


# ##################### UBD + Dataset Sampling 
# BATCH_SIZE=28
# RUN_NO=1
# PORT=12347
# SAMPLING=2
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
#     --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb \
#     --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific --dataset_sampling $SAMPLING >> outputs/BM_$DATASET_ORIG'_'UBD-DS-$SAMPLING'-'$BATCH_SIZE-$RUN_NO-UCF.txt
    



# ROOT=/home/c3-0/datasets/LaST/
# DATASET=last_cc_gender
# SIL=/home/c3-0/datasets/ID-Dataset/masks/LaST/jpgs
# POSE=....
# GENDER=Scripts/Helper/LaST_Gender.csv
# DATASET_COLORS=last_colors
# DATASET_ORIG=last

# ROOT=/home/c3-0/datasets/DeepChange/
# DATASET=deepchange_cc_gender
# SIL=/home/c3-0/datasets/ID-Dataset/masks/DeepChangeDataset/jpgs 
# POSE=....
# GENDER=Scripts/Helper/DeepChange_Gender.csv
# DATASET_COLORS=deepchange_colors
# DATASET_ORIG=deepchange

PORT=12355
BATCH_SIZE=40
RUN_NO=4
SAMPLING=10


################### RLQ (Celeb ReID Base Model)
############ POSE Cluster Variations ############


# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=36 --Pose=$POSE --pose-mode="R_LA_35" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save --sampling $SAMPLING  >> outputs/$DATASET_ORIG-RLQ_R_LA_35-$BATCH_SIZE-$SAMPLING-$RUN_NO.txt


# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=11 --Pose=$POSE --pose-mode="R_LAC_10" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save --sampling $SAMPLING  >> outputs/$DATASET_ORIG-RLQ_R_LAC_10-$BATCH_SIZE-$SAMPLING-$RUN_NO.txt



# R_LA_15 # 15 clusters
# BATCH_SIZE=32
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=11 --Pose=$POSE --pose-mode="R_A_10" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_A_10-$RUN_NO.txt

# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=11 --Pose=$POSE --pose-mode="R_A_10" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_A_10-$RUN_NO.txt


# BATCH_SIZE=32
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=16 --Pose=$POSE --pose-mode="R_A_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_A_15-$RUN_NO.txt

# BATCH_SIZE=32
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=16 --Pose=$POSE --pose-mode="R_A_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_A_15-$RUN_NO.txt

# BATCH_SIZE=32
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=21 --Pose=$POSE --pose-mode="R_A_20" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_A_20-$RUN_NO.txt

# BATCH_SIZE=32
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=21 --Pose=$POSE --pose-mode="R_A_20" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_A_20-$RUN_NO.txt


# BATCH_SIZE=32
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=26 --Pose=$POSE --pose-mode="R_A_25" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_LAC_25-$RUN_NO.txt

# BATCH_SIZE=32
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=26 --Pose=$POSE --pose-mode="R_A_25" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_LAC_25-$RUN_NO.txt

# BATCH_SIZE=32
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=31 --Pose=$POSE --pose-mode="R_LAC_30" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_LAC_30-$RUN_NO.txt

# BATCH_SIZE=32
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=31 --Pose=$POSE --pose-mode="R_LAC_30" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_LAC_30-$RUN_NO.txt
# BATCH_SIZE=32
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=31 --Pose=$POSE --pose-mode="R_LAC_30" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_LAC_30-$RUN_NO.txt


# BATCH_SIZE=

# BATCH_SIZE=32
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=41 --Pose=$POSE --pose-mode="R_LAC_40" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_LAC_40-$RUN_NO.txt

# BATCH_SIZE=32
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=41 --Pose=$POSE --pose-mode="R_LAC_40" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_LAC_40-$RUN_NO.txt





# ROOT=/home/c3-0/datasets/LaST/
# DATASET=last_cc_gender
# SIL=/home/c3-0/datasets/ID-Dataset/masks/LaST/jpgs
# POSE=....
# GENDER=Scripts/Helper/LaST_Gender.csv
# DATASET_COLORS=last_colors
# DATASET_ORIG=last
# ################### RLQ (Celeb ReID Base Model) - Silhouttes 
# ############ POSE Cluster Variations ############
# BATCH_SIZE=40
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG'_'RLQ-FAug_Newton.txt




##################### MARKET 
market=/home/c3-0/datasets/Market1501/Market-1501-v15.09.15/
ROOT=$market
DATASET_ORIG=market
DATASET=market_gender
POSE=Scripts/Helper/MARKET_Pose_Cluster.csv
GENDER=Scripts/Helper/market_Gender.csv



##### BaseModel -- Foreground Aug 
# PORT=12351
# BATCH_SIZE=28
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_ORIG \
#     --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" --additional_loss 'kl_o_oid' >> outputs/BM-$DATASET_ORIG-$RUN_NO.txt

# checkpoint=market/scratch_image/best_model.pth.tar
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_ORIG \
#     --gpu $GPUS --output ./ --tag image-$RUN_NO --root $ROOT --image --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --resume $checkpoint --eval 
    
# ##### Only UBD  
# PORT=12357
# BATCH_SIZE=40
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_ORIG \
#     --gpu $GPUS --output ./ --tag UBD-$RUN_NO --root $ROOT --image --max_epochs 300 --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb \
#     --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific >> outputs/UBD-$BATCH_SIZE-$DATASET_ORIG-$RUN_NO-300.txt

# # srun --pty -p gpu --gres=gpu:1 -c10 -C gmem48 bash
# PORT=12354
# BATCH_SIZE=28
# RUN_NO=4
# NUM_GPU=1
# for ((i=10; i<=200; i+=10))
# do
#     # checkpoint=market/UBD-$RUN_NO/best_model.pth.tar
#     checkpoint=market/UBD-$RUN_NO/checkpoint_ep"$i".pth.tar 
#     CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_ORIG \
#         --gpu $GPUS --output ./ --tag DUMP-$RUN_NO --root $ROOT --image --backbone="resnet50_joint2" --batch_size $BATCH_SIZE \
#         --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb \
#         --resume $checkpoint --eval --dataset-specific --no-classifier >> outputs/EVAL-UBD-$DATASET_ORIG-$RUN_NO.txt
# done


# # R_LA_25 # 25 clusters
# PORT=12366
# BATCH_SIZE=32
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_25_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=26 --Pose=$POSE --pose-mode="R_LA_25" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save --max_epochs 300  >> outputs/RLQ-$DATASET-$RUN_NO-300.txt

# # R_LA_15 # 25 clusters
# PORT=12366
# BATCH_SIZE=32
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_25_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save --max_epochs 400  >> outputs/RLQ-RLA15-$BATCH_SIZE-$DATASET-$RUN_NO-400.txt

# # R_LAC_15 # 25 clusters
# PORT=12345
# BATCH_SIZE=40
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_25_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=16 --Pose=$POSE --pose-mode="R_LAC_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save --max_epochs 400  >> outputs/RLQ-RLAC15-$BATCH_SIZE-$DATASET-$RUN_NO-400.txt


# # R_LAC_25 # 25 clusters
# PORT=12358
# BATCH_SIZE=28
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_25_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=26 --Pose=$POSE --pose-mode="R_LAC_25" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save --max_epochs 400  >> outputs/RLQ-RLAC25-$BATCH_SIZE-$DATASET-$RUN_NO-400.txt


# PORT=12372
# BATCH_SIZE=40
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_25_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=36 --Pose=$POSE --pose-mode="N_A_35" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save --max_epochs 400  >> outputs/RLQ-NA35-$BATCH_SIZE-$DATASET-$RUN_NO-400.txt


# PORT=12351
# BATCH_SIZE=28
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_25_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=21 --Pose=$POSE --pose-mode="R_LAC_20" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save --max_epochs 400  >> outputs/RLQ-RLAC20-$BATCH_SIZE-$DATASET-$RUN_NO-400.txt

# PORT=12372
# BATCH_SIZE=32
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_25_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=21 --Pose=$POSE --pose-mode="N_A_20" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save --max_epochs 400  >> outputs/RLQ-NA20-$BATCH_SIZE-$DATASET-$RUN_NO-400.txt

# PORT=12373
# BATCH_SIZE=32
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_25_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=21 --Pose=$POSE --pose-mode="R_A_20" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save --max_epochs 400  >> outputs/RLQ-RA20-$BATCH_SIZE-$DATASET-$RUN_NO-400.txt




# outputs/UCF/RLQ-market_gender-4-300.txt
# outputs/UCF/RLQ-RA20-32-market_gender-1-400.txt
# outputs/UCF/RLQ-RLAC25-40-market_gender-1-400.txt
# outputs/UCF/RLQ-RLAC25-market_gender-1-400.txt


















# rsync -r ~/RLQ-CGAL-UBD/logs ucf0:~/RLQ-CGAL-UBD/
# rsync -a outputs/* ucf2:~/RLQ-CGAL-UBD/outputs/
rsync -a outputs/* ucf2:~/RLQ-CGAL-UBD/outputs/UCF/
# rsync -a ucf0:~/RLQ-CGAL-UBD/outputs/* ~/RLQ-CGAL-UBD/outputs/UCF/

# cd ~/RLQ-CGAL-UBD
# sbatch Scripts/sbatch.sh
