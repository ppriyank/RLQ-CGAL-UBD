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
######## Base Model ############
# SEP Branches 
BATCH_SIZE=28
RUN_NO=2
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
    --gpu $GPUS --output ./ --tag scratch --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_separate2" --batch_size $BATCH_SIZE \
    --seed=$RUN_NO --train_fn="2feats_pair3" --additional_loss 'kl_o_oid' >> outputs/LT_SEP_KL_$BATCH_SIZE"_"$RUN_NO"_Debug.txt"
# Base Model : Joint at Block 3
BATCH_SIZE=28
RUN_NO=2
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
    --gpu $GPUS --output ./ --tag scratch --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE \
    --seed=$RUN_NO --train_fn="2feats_pair3" --additional_loss 'kl_o_oid' --overlap_1 -1 >> outputs/LT_J-3_KL_$BATCH_SIZE"_"$RUN_NO"_Debug.txt"
# Base Model :  Joint at Block 1
BATCH_SIZE=28
RUN_NO=2
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
    --gpu $GPUS --output ./ --tag scratch --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE \
    --seed=$RUN_NO --train_fn="2feats_pair3" --additional_loss 'kl_o_oid' --overlap_1 -3 >> outputs/LT_J-1_KL_$BATCH_SIZE"_"$RUN_NO"_Debug.txt"

# BaseModel -- Foreground Aug 
PORT=12347
BATCH_SIZE=28
RUN_NO=2
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_ORIG \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" --additional_loss 'kl_o_oid' 
# ==> Best Rank-1 39.3%, achieved at epoch 40. Best MaP 17.9%

# BaseModel 
PORT=12348
BATCH_SIZE=28
RUN_NO=4
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET_COLORS \
    --gpu $GPUS --output ./ --tag scratch_image --root $ROOT --image --max_epochs 200 --silhouettes=$SIL --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" --additional_loss 'kl_o_oid' --seed=$RUN_NO
# ==> Best Rank-1 42.1%, achieved at epoch 40. Best MaP 20.9%
