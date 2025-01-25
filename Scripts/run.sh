cd ~/RLQ-CGAL-UBD/
conda activate bert2
ROOT=/data/priyank/synthetic

ltcc=/data/priyank/synthetic/LTCC/
ltcc_sil="/data/priyank/synthetic/LTCC/masks/ltcc/" 
ltcc_pose=/data/priyank/synthetic/LTCC/AlphaPose/2DPose/
ltcc_gender=Scripts/Helper/LTCC_Gender.csv

Celeb_Wt_KL=logs/celeb/B=40_KL_4/best_model.pth.tar
R_LA_15_2_ABS_GID=logs/celeb_cc_colors/R_LA_15_2_ABS_GID/best_model.pth.tar


ntu=/data/priyank/synthetic/NTU/RGB/
ntu_sil=/data/priyank/synthetic/NTU/Mask
NTU_Wt_KL=logs/BM_NTU/best_model.pth.tar
NTU_Wt_KL=logs/BM_NTU_Tight/best_model.pth.tar

celeb=/data/priyank/synthetic/Celeb-reID/
celeb_sil=/data/priyank/synthetic/Celeb-reID/masks/jpgs 
celeb_pose=Scripts/Helper/Celeb-reID_Pose_Cluster.csv
celeb_gender=Scripts/Helper/Celeb-reID_Gender.csv

prcc=/data/priyank/synthetic/PRCC/
prcc_sil="/data/priyank/synthetic/PRCC/masks/jpgs/" 
prcc_gender=Scripts/Helper/PRCC_Gender.csv
prcc_pose=...

last=/data/priyank/synthetic/LaST/
last_sil=/data/priyank/synthetic/LaST/masks/jpgs
last_pose=....
last_gender=Scripts/Helper/LaST_Gender.csv

deepchange=/data/priyank/synthetic/DeepChangeDataset
deepchange_sil=/data/priyank/synthetic/DeepChangeDataset/masks/jpgs 
deepchange_pose=....
deepchange_gender=Scripts/Helper/DeepChange_Gender.csv


market=/data/priyank/synthetic/Market-1501-v15.09.15/

ROOT=$deepchange
DATASET=deepchange_cc_gender
SIL=$deepchange_sil
POSE=$deepchange_pose
GENDER=$deepchange_gender
DATASET_COLORS=deepchange_colors
DATASET_ORIG=deepchange

ROOT=$last
DATASET=last_cc_gender
SIL=$last_sil
POSE=$last_pose
GENDER=$last_gender
DATASET_COLORS=last_colors
DATASET_ORIG=last

ROOT=$ltcc
DATASET=ltcc_cc_gender
SIL=$ltcc_sil
POSE=$ltcc_pose
GENDER=$ltcc_gender
DATASET_COLORS=ltcc_colors
DATASET_ORIG=ltcc


GPUS=0,1
NUM_GPU=2
RUN_NO=1



############################################################
######## Vanilla CAL  ############
# PORT=12345
# BATCH_SIZE=32
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal.yaml --dataset $DATASET_ORIG \
#     --gpu $GPUS --output ./ --tag VANILLA_CAL --root $ROOT --image --max_epochs 200 --backbone="resnet50" --batch_size $BATCH_SIZE --only-CAL 
# # ==> Best Rank-1 41.3%, achieved at epoch 90. Best MaP 18.6%

# PORT=12365
# NUM_GPU=2
# BATCH_SIZE=64
# checkpoint=ltcc/VANILLA_CAL/best_model.pth.tar
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_ORIG --gpu $GPUS --output ./ --root $ROOT --image \
#     --tag CAL_LTCC --resume $checkpoint --eval --no-classifier

############################################################
######## CAL (Celebrity) ############
# ROOT=$celeb
# DATASET=celeb_cc_gender
# SIL=$celeb_sil
# POSE=$celeb_pose
# GENDER=$celeb_gender
# DATASET_COLORS=celeb_colors
# DATASET_ORIG=celeb
# # Vanilla CAL 
# PORT=12345
# BATCH_SIZE=32
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal.yaml --dataset $DATASET_ORIG \
#     --gpu $GPUS --output ./ --tag VANILLA_CAL --root $ROOT --image --max_epochs 200 --backbone="resnet50" --batch_size $BATCH_SIZE --only-CAL >> outputs/celeb_cal.txt
# ==> Best Rank-1 41.3%, achieved at epoch 90. Best MaP 18.6%


############################################################
######## Base Model ############
# BaseModel 
# PORT=12348
# BATCH_SIZE=28
# RUN_NO=4
# PERCENT=100
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
#     --gpu $GPUS --output ./ --tag BM --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" --additional_loss 'kl_o_oid' --seed=$RUN_NO >> outputs/BM.txt
# # ==> Best Rank-1 42.1%, achieved at epoch 40. Best MaP 20.9%


##################### CAL + Only UBD    
# BATCH_SIZE=28
# RUN_NO=1
# PORT=12353
# Celeb_CAL_Wt=logs/celeb/VANILLA_CAL/best_model.pth.tar
# # rsync -r logs/celeb/VANILLA_CAL/ ucf0:~/RLQ-CGAL-UBD/logs/celeb/VANILLA_CAL/
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal.yaml --dataset $DATASET_ORIG \
#     --gpu $GPUS --output ./ --tag UBD_CAL --root $ROOT --image --max_epochs 200 --backbone="resnet50" --batch_size $BATCH_SIZE --only-CAL \
#     --teacher_wt $Celeb_CAL_Wt --teacher_dataset celeb --teacher_dir $celeb \
#     --unused_param --seed=$RUN_NO --dataset-specific >> outputs/$DATASET_ORIG'_'UBD-100.txt
# # ==> Best Rank-1 44.6%, achieved at epoch 100. Best MaP 21.8%

# ############################################################
# ##################### Only UBD    
# BATCH_SIZE=28
# RUN_NO=1
# PORT=12353
# SAMPLING=70
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
#     --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb \
#     --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific --sampling $SAMPLING >> outputs/BM_$DATASET_ORIG'_'UBD-100.txt

# ##################### Only UBD  - Teacher 
# BATCH_SIZE=28
# RUN_NO=1
# PORT=12355
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
#     --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb \
#     --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific --no-teacher --tag UBD-BM-NoTeacher >> outputs/BM_$DATASET_ORIG'_'UBD-Teacher+Int.MSE.txt

# ##################### UBD  - Teacher w/ ltcc 
# BATCH_SIZE=28
# RUN_NO=1
# PORT=12356
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
#     --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset ltcc --teacher_dir $ltcc \
#     --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific --no-teacher --tag UBD-BM-NoTeacher-LTCC >> outputs/BM_$DATASET_ORIG'_'UBD-Teacher+LTCC+Int.MSE.txt

# ##################### ltcc as augmentation  
# BATCH_SIZE=28
# RUN_NO=3
# PORT=12357
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
#     --gpu $GPUS --output ./ --tag BM_lr_aug --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
#     --additional_loss="kl_o_oid" --seed=$RUN_NO --lr-aug >> outputs/BM_$DATASET_ORIG'_'LR_AUG-$RUN_NO"_2.txt"


# ##################### Only UBD - MSE (ONLY KL)
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
#     --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb \
#     --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific --no-teacher --tag UBD-BM-NoTeacher-NoMSE --no-mse >> outputs/BM_$DATASET_ORIG'_'UBD-Teacher-MSE-$RUN_NO.txt 

# ##################### Only UBD + MSE (SS)   
# BATCH_SIZE=28
# RUN_NO=1
# PORT=12348
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
#     --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb \
#     --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific --mse-ss >> outputs/BM_$DATASET_ORIG'_'UBD-100-MSE-SS-$RUN_NO.txt

# ##################### UBD + Sampling 
# BATCH_SIZE=32
# RUN_NO=1
# PORT=12359
# SAMPLING=100
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
#     --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb \
#     --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific --sampling $SAMPLING >> outputs/BM_$DATASET_ORIG'_'UBD-$SAMPLING'-'$BATCH_SIZE.txt

################### RLQ (Celeb ReID Base Model) - Silhouttes 
############ POSE Cluster Variations ############
# BATCH_SIZE=32
# RUN_NO=4
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
#     --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
#     --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
#     --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
#     --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save  >> outputs/$DATASET_ORIG'_'RLQ-FAug_Newton.txt



##################### UBD + Dataset Sampling 
BATCH_SIZE=28
RUN_NO=3
PORT=12345
SAMPLING=6
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
    --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb \
    --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific --dataset_sampling $SAMPLING >> outputs/BM_$DATASET_ORIG'_'UBD-DS-$SAMPLING'-'$BATCH_SIZE-$RUN_NO-UCF2.txt
    

BATCH_SIZE=28
RUN_NO=2
PORT=12347
SAMPLING=2
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
    --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb \
    --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific --dataset_sampling $SAMPLING >> outputs/BM_$DATASET_ORIG'_'UBD-DS-$SAMPLING'-'$BATCH_SIZE-$RUN_NO-UCF2.txt
    




ROOT=$prcc
DATASET=prcc_cc_gender
SIL=$prcc_sil
POSE=$prcc_pose
GENDER=$prcc_gender
DATASET_COLORS=prcc_colors
DATASET_ORIG=prcc




ntu=/data/priyank/synthetic/NTU/RGB/
ntu_sil=/data/priyank/synthetic/NTU/Mask
ROOT=$ntu
SIL=$ntu_sil
DATASET_COLORS=ntu_colors
DATASET_ORIG=ntu
NTU_Wt_KL=logs/BM_NTU/best_model.pth.tar
######## Base Model ############
PORT=12348
BATCH_SIZE=28
RUN_NO=4
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
    --gpu $GPUS --output ./ --tag BM_NTU --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" --additional_loss 'kl_o_oid' --seed=$RUN_NO >> outputs/BM_$DATASET_ORIG.txt
# ==> Best Rank-1 42.1%, achieved at epoch 40. Best MaP 20.9%



ROOT=$ltcc
DATASET=ltcc_cc_gender
SIL=$ltcc_sil
POSE=$ltcc_pose
GENDER=$ltcc_gender
DATASET_COLORS=ltcc_colors
DATASET_ORIG=ltcc
# NTU_Wt_KL=logs/BM_NTU/best_model.pth.tar
NTU_Wt_KL=logs/BM_NTU_Tight/best_model.pth.tar
##################### Only UBD    
BATCH_SIZE=28
RUN_NO=3
PORT=12353
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
    --teacher_wt $NTU_Wt_KL --teacher_dataset ntu_colors --teacher_dir $ntu \
    --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific >> outputs/$DATASET_ORIG'_'UBD-NTU-UCF-$BATCH_SIZE-$RUN_NO.txt


rsync -r ~/RLQ-CGAL-UBD/logs/ ucf4:~/RLQ-CGAL-UBD/logs/
rsync -r ~/RLQ-CGAL-UBD/logs/ ucf:~/RLQ-CGAL-UBD/logs/


################### RLQ (Celeb ReID Base Model)
# R_LA_15 # 15 clusters
PORT=12348
BATCH_SIZE=32
RUN_NO=4
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
    --gpu $GPUS --output ./ --tag RLQ_15_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
    --teacher_wt $NTU_Wt_KL --teacher_dataset ntu_colors --teacher_dir $ntu --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
    --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
    --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save 

# R_LA_25 # 25 clusters
BATCH_SIZE=32
RUN_NO=1
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
    --gpu $GPUS --output ./ --tag RLQ_25_"B=$BATCH_SIZE"_"$RUN_NO" --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair23" \
    --teacher_wt $NTU_Wt_KL --teacher_dataset ntu_colors --teacher_dir $ntu --KL_weights "[0,0,1,1,1]" --MSE_weights "[1,1,1,1]" --CA_weight 0 \
    --class_2=26 --Pose=$POSE --pose-mode="R_LA_25" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
    --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO --dataset-specific --no-save 




##################### MARKET 
ROOT=$market
DATASET_ORIG=market

# BaseModel -- Foreground Aug 
PORT=12347
BATCH_SIZE=28
RUN_NO=2
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_ORIG \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" --additional_loss 'kl_o_oid' >> outputs/BM-$DATASET_ORIG-$RUN_NO.txt


##################### Only UBD    
BATCH_SIZE=28
RUN_NO=2
PORT=12345
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_ORIG \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
    --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb \
    --additional_loss="kl_o_oid" --unused_param --seed=$RUN_NO --dataset-specific >> outputs/$DATASET_ORIG'_'UBD-NTU-UCF-$BATCH_SIZE-$RUN_NO.txt




    
     










# error analysis of EVA : show pose and gender 
rsync -r /data/priyank/synthetic/NTU/ ucf0:/home/c3-0/datasets/ID-Dataset/ntu/
rsync -r /data/priyank/synthetic/NTU/ ucf4:/groups/yrawat/NTU/

rsync -r /data/priyank/synthetic/NTU/ ucf0:/home/c3-0/datasets/ID-Dataset/ntu/




rsync -r logs/ ucf4:~/RLQ-CGAL-UBD/logs/
rsync -r logs/ ucf0:~/RLQ-CGAL-UBD/logs/

rsync -a ~/RLQ-CGAL-UBD/hr_lr.png ucf2:~/RLQ-CGAL-UBD/

# rsync -r ucf0:/home/c3-0/datasets/ID-Dataset/masks/DeepChangeDataset/ /data/priyank/synthetic/DeepChangeDataset/
rsync -r /data/priyank/synthetic/DeepChangeDataset/ ucf4:/groups/yrawat/DeepChangeDataset/
rsync -r /home/c3-0/datasets/ID-Dataset/masks/DeepChangeDataset/ ucf4:/groups/yrawat/DeepChangeDataset/