cd ~/RLQ-CGAL-UBD/
conda activate bert2
conda activate pathak3

Celeb_Wt_KL=logs/celeb/B=40_KL_4/best_model.pth.tar
R_LA_15_2_ABS_GID=logs/celeb_cc_colors/R_LA_15_2_ABS_GID/best_model.pth.tar
celeb=/groups/yrawat/Celeb-reID/
NTU_Wt_KL=logs/BM_NTU/best_model.pth.tar

deepchange=/groups/yrawat/DeepChange/


ltcc=/groups/yrawat/LTCC/
ltcc_pose=/data/priyank/synthetic/LTCC/AlphaPose/2DPose/
ltcc_gender=Scripts/Helper/LTCC_Gender.csv
ltcc_sil="/groups/yrawat/LTCC/masks/ltcc/" 

prcc=/groups/yrawat/PRCC
prcc_sil="/groups/yrawat/PRCC/masks/jpgs/" 
prcc_gender=Scripts/Helper/PRCC_Gender.csv
prcc_pose=...

ntu=/groups/yrawat/NTU/RGB/
NTU_Wt_KL=logs/BM_NTU/best_model.pth.tar
NTU_Wt_KL=logs/BM_NTU_Tight/best_model.pth.tar

ROOT=$prcc
DATASET=prcc_cc_gender
SIL=$prcc_sil
POSE=$prcc_pose
GENDER=$prcc_gender
DATASET_COLORS=prcc_colors
DATASET_ORIG=prcc


# ROOT=$ltcc
# DATASET=ltcc_cc_gender
# SIL=$ltcc_sil
# POSE=$ltcc_pose
# GENDER=$ltcc_gender
# DATASET_COLORS=ltcc_colors
# DATASET_ORIG=ltcc

PORT=12345
GPUS=0,1
NUM_GPU=2
RUN_NO=1


# rsync -a outputs/* ucf2:~/RLQ-CGAL-UBD/outputs/
# rsync -a outputs/* ucf2:~/RLQ-CGAL-UBD/outputs/

# ##################### Only UBD    
# BATCH_SIZE=40
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
#     --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
#     --teacher_wt $NTU_Wt_KL --teacher_dataset ntu_colors --teacher_dir $ntu \
#     --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific >> outputs/$DATASET_ORIG'_'UBD-NTU-UCF-$BATCH_SIZE-$RUN_NO.txt



##################### UBD + Dataset Sampling  + SAMPLING  
BATCH_SIZE=28
RUN_NO=1
SAMPLING=4
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
    --teacher_wt $NTU_Wt_KL --teacher_dataset ntu_colors --teacher_dir $ntu \
    --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific --dataset_sampling $SAMPLING >> outputs/$DATASET_ORIG'_'UBD-NTU-DS-$SAMPLING-$BATCH_SIZE-$RUN_NO.txt

BATCH_SIZE=28
RUN_NO=4
SAMPLING=4
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
    --teacher_wt $NTU_Wt_KL --teacher_dataset ntu_colors --teacher_dir $ntu \
    --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific --dataset_sampling $SAMPLING >> outputs/$DATASET_ORIG'_'UBD-NTU-DS-$SAMPLING-$BATCH_SIZE-$RUN_NO.txt





    

# ################### RLQ (Celeb ReID Base Model) + SS
# BATCH_SIZE=32
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=26 --Pose=$POSE --pose-mode="R_LAC_25" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save --mse-ss >> outputs/$DATASET_ORIG-RLQ_R_LAC_25-SS-$RUN_NO.txt


# # RLQ_R_LAC_10 + DS SAMPLING
# BATCH_SIZE=40
# RUN_NO=1
# SAMPLING=30
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=11 --Pose=$POSE --pose-mode="R_LAC_10" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save --dataset_sampling $SAMPLING >> outputs/$DATASET_ORIG-RLQ_R_LAC_10-$SAMPLING-$RUN_NO.txt




##################### UBD + Dataset Sampling 
BATCH_SIZE=32
RUN_NO=1
PORT=12345
SAMPLING=8
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
    --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb \
    --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific --dataset_sampling $SAMPLING >> outputs/BM_$DATASET_ORIG'_'UBD-DS-$SAMPLING'-'$BATCH_SIZE-$RUN_NO-UCF4.txt
    

BATCH_SIZE=32
RUN_NO=2
PORT=12347
SAMPLING=8
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
    --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb \
    --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific --dataset_sampling $SAMPLING >> outputs/BM_$DATASET_ORIG'_'UBD-DS-$SAMPLING'-'$BATCH_SIZE-$RUN_NO-UCF4.txt
    





cd ~/RLQ-CGAL-UBD/
conda activate pathak3
PORT=12345
GPUS=0,1
NUM_GPU=2
RUN_NO=1

Celeb_Wt_KL=logs/celeb/B=40_KL_4/best_model.pth.tar
R_LA_15_2_ABS_GID=logs/celeb_cc_colors/R_LA_15_2_ABS_GID/best_model.pth.tar
celeb=/groups/yrawat/Celeb-reID/

ROOT=$prcc
DATASET=prcc_cc_gender
SIL=$prcc_sil
POSE=$prcc_pose
GENDER=$prcc_gender
DATASET_COLORS=prcc_colors
DATASET_ORIG=prcc 

################### RLQ (Celeb ReID Base Model)
############ POSE Cluster Variations ############
# # R_LA_15 # 15 clusters
# BATCH_SIZE=32
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
#     --class_2=16 --Pose=$POSE --pose-mode="R_LAC_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_LAC_15-$RUN_NO.txt

# BATCH_SIZE=32
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=21 --Pose=$POSE --pose-mode="R_LAC_20" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_LAC_20-$RUN_NO.txt

# BATCH_SIZE=32
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=26 --Pose=$POSE --pose-mode="R_LA_25" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_LA_25-$RUN_NO.txt

# BATCH_SIZE=32
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=31 --Pose=$POSE --pose-mode="R_LA_30" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_LA_30-$RUN_NO.txt

# BATCH_SIZE=32
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=36 --Pose=$POSE --pose-mode="R_LA_35" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_LA_35-$RUN_NO.txt


# BATCH_SIZE=32
# RUN_NO=1
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=41 --Pose=$POSE --pose-mode="R_LAC_40" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG-RLQ_R_LAC_40-$RUN_NO.txt









# srun --pty -t200:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu80 bash

# rsync -a outputs/* ucf2:~/RLQ-CGAL-UBD/outputs/ 
