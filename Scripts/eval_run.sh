cd ~/RLQ-CGAL-UBD/
conda activate bert2

ltcc=/data/priyank/synthetic/LTCC/
ltcc_sil="/data/priyank/synthetic/LTCC/masks/ltcc/" 
ltcc_pose=/data/priyank/synthetic/LTCC/AlphaPose/2DPose/
ltcc_gender=Scripts/Helper/LTCC_Gender.csv

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
NUM_GPU=1
RUN_NO=1




ROOT=$celeb
DATASET=celeb_cc_gender
SIL=$celeb_sil
POSE=...
GENDER=$celeb_gender
DATASET_COLORS=celeb_colors
DATASET_ORIG=celeb
GPUS=0,1
RUN_NO=1

# Celeb HQ
NUM_GPU=1
checkpoint=Dump/UBD-BM-NoTeacher/best_model.pth.tar
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --silhouettes=$celeb_sil --root $celeb --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
    --tag BM_NT_MSE_Celeb_HD --resume $checkpoint --eval --no-classifier
# top1:29.8% top5:37.3% top10:41.5% top20:47.9% mAP:3.0%
# Celeb LQ
checkpoint=Dump/UBD-BM-NoTeacher/best_model.pth.tar
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --silhouettes=$celeb_sil --root $celeb --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
    --tag BM_NT_MSE_Celeb_LR --resume $checkpoint --eval --LR-MODE --no-classifier --dataset-specific
# top1:21.0% top5:29.9% top10:34.9% top20:40.8% mAP:2.1%    

# Celeb HQ
NUM_GPU=1
checkpoint=Dump/UBD-BM-NoTeacher-NoMSE/best_model.pth.tar
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --silhouettes=$celeb_sil --root $celeb --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
    --tag BM_NT_NoMSE_Celeb_HD --resume $checkpoint --eval --no-classifier
# Celeb LQ
checkpoint=Dump/UBD-BM-NoTeacher-NoMSE/best_model.pth.tar
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --silhouettes=$celeb_sil --root $celeb --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
    --tag BM_NT_NoMSE_Celeb_LR --resume $checkpoint --eval --LR-MODE --no-classifier --dataset-specific
# top1:21.0% top5:29.9% top10:34.9% top20:40.8% mAP:2.1%    


# RQL-1 
checkpoint=logs/RLQ_25_B=32_1/best_model.pth.tar
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
    --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint2" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $Celeb_Wt_KL --teacher_dataset celeb --teacher_dir $celeb --class_2=26 --Pose=$POSE --pose-mode="R_LA_25" --overlap_2=-3 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --seed=$RUN_NO \
    --eval --resume $checkpoint --tag RLQ_25_B=32_1 --no-classifier 



cd ~/CCReID 
python Scripts/analysis/low_res_analysis_Celeb.py ~/RLQ-CGAL-UBD/BM_NT_MSE_Celeb_HD ~/RLQ-CGAL-UBD/BM_NT_MSE_Celeb_LR
python Scripts/analysis/low_res_analysis_Celeb.py ~/RLQ-CGAL-UBD/BM_NT_NoMSE_Celeb_HD ~/RLQ-CGAL-UBD/BM_NT_NoMSE_Celeb_LR




