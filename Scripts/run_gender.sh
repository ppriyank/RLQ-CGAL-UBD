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
###################### Only Gender 
BATCH_SIZE=28
RUN_NO=2
PORT=12349
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair14" \
    --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --additional_loss="kl_o_oid" --seed=$RUN_NO 
# ==> Best Rank-1 43.4%, achieved at epoch 60. Best MaP 20.9%

###################### Gender Variants
# Gender CGAL Branch + CAL Branch 
BATCH_SIZE=28
RUN_NO=2
PORT=12350
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair14" \
    --use_gender $GENDER --extra_class_embed 8192 --extra_class_no 2 --gender_overall --additional_loss="kl_o_oid" --seed=$RUN_NO 
# ==> Best Rank-1 42.1%, achieved at epoch 30. Best MaP 20.8%    

# Gender CAL Branch 
BATCH_SIZE=28
RUN_NO=4
PORT=12351
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair14" \
    --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_clothes --additional_loss="kl_o_oid" --seed=$RUN_NO 
# ==> Best Rank-1 41.6%, achieved at epoch 30. Best MaP 20.8%

# Gender CGAL Branch 
BATCH_SIZE=28
RUN_NO=2
PORT=12352
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair14" \
    --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --additional_loss="kl_o_oid" --seed=$RUN_NO 
# ==> Best Rank-1 43.4%, achieved at epoch 60. Best MaP 20.9%

############################################################
###################### Distillation + Gender 
BATCH_SIZE=28
RUN_NO=1
PORT=12353
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair14" \
    --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id \
    --additional_loss 'kl_o_oid' --unused_param --seed=$RUN_NO --dataset-specific 
