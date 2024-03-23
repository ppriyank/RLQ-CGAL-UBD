cd ~/RLQ-CGAL-UBD/
conda activate bert2
ROOT=/data/priyank/synthetic

ltcc=/data/priyank/synthetic/LTCC/
ltcc_sil="/data/priyank/synthetic/LTCC/masks/ltcc/" 
ltcc_bkd=/data/priyank/synthetic/LTCC/masks/ltcc/background/
ltcc_enchanced=/data/priyank/synthetic/LTCC/Enhanced
ltcc_pose=/data/priyank/synthetic/LTCC/AlphaPose/2DPose/
ltcc_gender=Scripts/Helper/LTCC_Gender.csv

ltcc_v1=Helper/LTCC_Validation_1.csv
ltcc_v2=Helper/LTCC_Validation_2.csv
ltcc_v3=Helper/LTCC_Validation_3.csv


Celeb_Wt_KL=logs/celeb/B=40_KL_4/checkpoint_ep200.pth.tar
R_LA_15_2_ABS_GID=logs/celeb_cc_colors/R_LA_15_2_ABS_GID/best_model.pth.tar

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
####################### Only Pose # R_LA_15 
BATCH_SIZE=28
RUN_NO=3
PORT=12354
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair16" \
    --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --use_gender $GENDER --seed=$RUN_NO 
# ==> Best Rank-1 43.4%, achieved at epoch 30. Best MaP 20.9%

###################### POSE VARIETY  
###### POSE CGAL 
# R_LA_15 + Abs 
BATCH_SIZE=28
RUN_NO=3
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair16" \
    --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
    --use_gender $GENDER --pose_id --seed=$RUN_NO --no-save 
# ==> Best Rank-1 43.4%, achieved at epoch 30. Best MaP 20.9%    
###### POSE CAL 
# R_LA_15 + Abs 
BATCH_SIZE=28
RUN_NO=3
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair16" \
    --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
    --use_gender $GENDER --pose_cl --seed=$RUN_NO --no-save 
# ==> Best Rank-1 41.6%, achieved at epoch 30. Best MaP 21.5%
######## POSE BOTH 
# R_LA_15 + Abs 
BATCH_SIZE=28
RUN_NO=2
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair16" \
    --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
    --use_gender $GENDER --pose_both --seed=$RUN_NO --no-save 
# ==> Best Rank-1 42.1%, achieved at epoch 30. Best MaP 20.1%    

###################### Distillation + Pose  
# R_LA_15 + Abs 
BATCH_SIZE=28
RUN_NO=1
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
    --gpu $GPUS --output ./ --tag teacher_student --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size $BATCH_SIZE --train_fn="2feats_pair16" \
    --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --additional_loss="Pose3_kl_o_oid" --unused_param \
    --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --use_gender $GENDER --seed=$RUN_NO --dataset-specific >> outputs/UBD_POS-3-R_LA_15_"B=$BATCH_SIZE"_"$RUN_NO".txt 
# top1:46.2% top5:59.2% top10:66.1% top20:70.9% mAP:22.2%
        





