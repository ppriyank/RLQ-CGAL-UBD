cd ~/CCReID/
conda activate bert2
BKD=/data/priyank/synthetic/dataset-backgrounds/
GPUS=0,1
ltcc=/data/priyank/synthetic/LTCC/
ltcc_sil="/data/priyank/synthetic/LTCC/masks/ltcc/" 
ltcc_bkd=/data/priyank/synthetic/LTCC/masks/ltcc/background/

ltcc_v1=Scripts/Helper/LTCC_Validation_1.csv
ltcc_v2=Scripts/Helper/LTCC_Validation_2.csv
ltcc_v3=Scripts/Helper/LTCC_Validation_3.csv

celeb=/home/c3-0/datasets/ID-Dataset/Celeb-reID/
celeb=/data/priyank/synthetic/Celeb-reID/
Celeb_Wt=celeb/scratch_image/checkpoint_ep200.pth.tar

cd ~/CCReID/
conda activate bert2
ROOT=/data/priyank/synthetic
DATASET_PATH=$ROOT/orig_RGB_vids/
TIGHT_PATH=$ROOT/Original_Tight/
STATIC_PATH=$ROOT/Original_Static/
CROP_SIL=$ROOT/Casia_Silhouettes_Crops/
CORRUPT=Scripts/corrupt.txt
Youtube_Corrupt="Scripts/Youtube_corrupt.txt"
Root="/squash/casia-youtube-dump-lzo"
BKD=/data/priyank/synthetic/dataset-backgrounds/

ltcc=/data/priyank/synthetic/LTCC/
ltcc_sil="/data/priyank/synthetic/LTCC/masks/ltcc/" 
ltcc_bkd=/data/priyank/synthetic/LTCC/masks/ltcc/background/
ltcc_enchanced=/data/priyank/synthetic/LTCC/Enhanced
ltcc_pose=/data/priyank/synthetic/LTCC/AlphaPose/2DPose/
ltcc_gender=Scripts/Helper/LTCC_Gender.csv

ltcc_v1=Helper/LTCC_Validation_1.csv
ltcc_v2=Helper/LTCC_Validation_2.csv
ltcc_v3=Helper/LTCC_Validation_3.csv

Celeb_Wt=logs/celeb/scratch_image/checkpoint_ep200.pth.tar
Celeb_Wt_KL=logs/celeb/B=40_KL_4/checkpoint_ep200.pth.tar
Celeb_Wt_NO_KL=logs/celeb/B=40_NO_KL_4/checkpoint_ep200.pth.tar
Valid=logs/ltcc_colors/color_aug/checkpoint_ep200.pth.tar
R_LA_15_2_ABS_GID=logs/celeb_cc_colors/R_LA_15_2_ABS_GID/best_model.pth.tar
R_LAC_25_1_ABS_GID=logs/celeb_cc_colors/R_LAC_25_1_ABS_GID/best_model.pth.tar

R152_R_LAC_25_1_ABS_GID=logs/celeb_cc_colors/R152_R_LAC_25_1_ABS_GID/best_model.pth.tar
R152_R_LA_15_2_ABS_GID=logs/celeb_cc_colors/R152_R_LA_15_2_ABS_GID/best_model.pth.tar

C_R152_32=logs/celeb_colors/R152_KL_2_32/best_model.pth.tar
Celeb_Wt_Only_Cal=logs/celeb/CAL_40/best_model.pth.tar

celeb=/home/c3-0/datasets/ID-Dataset/Celeb-reID/
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
ccvid=/data/priyank/synthetic/CCVID/


ROOT=$ltcc
DATASET=ltcc_cc_gender
SIL=$ltcc_sil
POSE=$ltcc_pose
GENDER=$ltcc_gender
BATCH_SIZE=28
DATASET_COLORS=ltcc_colors
DATASET_NAME=LTCC
PORT=12361
RUN_NO=1

PORT=12365
NUM_GPU=2
BATCH_SIZE=64

# Wt=logs/ltcc_colors/TS_7_2/checkpoint_ep40.pth.tar
# CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors \
#     --gpu $GPUS --output ./ --tag analysis --root $ltcc --image --max_epochs 200 --silhouettes=$ltcc_sil --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
#     --resume=$Wt --eval --no-classifier --pair_loss "cloth_triplet"

# python Scripts/analysis.py  ~/CCReID/analysis /data/priyank/synthetic/LTCC/         
 
############ Validation ############
    Valid=logs/ltcc_colors/color_aug/checkpoint_ep150.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors \
        --gpu $GPUS --output ./ --tag validation2 --root $ltcc --image --max_epochs 200 --silhouettes=$ltcc_sil --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
        --resume=$Valid --use-validation $ltcc_v1 --eval --no-classifier
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors \
        --gpu $GPUS --output ./ --tag validation2_colors --root $ltcc --image --max_epochs 200 --silhouettes=$ltcc_sil --sil_mode "foreround_overlap_test" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
        --resume=$Valid --use-validation $ltcc_v1 --eval --no-classifier
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors \
        --gpu $GPUS --output ./ --tag validation2_black_bkd --root $ltcc --image --max_epochs 200 --silhouettes=$ltcc_sil --sil_mode "foreround_bkd_test" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
        --resume=$Valid --use-validation $ltcc_v1 --eval --no-classifier
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors \
        --gpu $GPUS --output ./ --tag validation2_colors_black_bkd --root $ltcc --image --max_epochs 200 --silhouettes=$ltcc_sil --sil_mode "foreround_overlap_bkd_test" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
        --resume=$Valid --use-validation $ltcc_v1 --eval --no-classifier

    Valid=logs/ltcc_colors/color_aug_w_bkd/checkpoint_ep130.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors \
        --gpu $GPUS --output ./ --tag validation3 --root $ltcc --image --max_epochs 200 --silhouettes=$ltcc_sil --sil_mode "foreround_overlap" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
        --resume=$Valid --use-validation $ltcc_v1 --eval --no-classifier
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors \
        --gpu $GPUS --output ./ --tag validation3_colors --root $ltcc --image --max_epochs 200 --silhouettes=$ltcc_sil --sil_mode "foreround_overlap_test" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
        --resume=$Valid --use-validation $ltcc_v1 --eval --no-classifier
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors \
        --gpu $GPUS --output ./ --tag validation3_black_bkd --root $ltcc --image --max_epochs 200 --silhouettes=$ltcc_sil --sil_mode "foreround_bkd_test" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
        --resume=$Valid --use-validation $ltcc_v1 --eval --no-classifier
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors \
        --gpu $GPUS --output ./ --tag validation3_colors_black_bkd --root $ltcc --image --max_epochs 200 --silhouettes=$ltcc_sil --sil_mode "foreround_overlap_bkd_test" --backbone="resnet50_joint2" --batch_size $BATCH_SIZE --train_fn="2feats_pair3" \
        --resume=$Valid --use-validation $ltcc_v1 --eval --no-classifier


#
############ Celeb HD only CAL ############
    checkpoint=logs/ltcc/LT_CAL/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb --gpu $GPUS --output ./ --root $celeb --image \
        --tag CAL_32_1_Celeb_HD --resume $checkpoint --eval --no-classifier
    # top1:37.3% top5:44.2% top10:47.6% top20:53.8% mAP:3.9%    
    ############# Celeb LR only CAL ############
    checkpoint=logs/ltcc/LT_CAL/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb --gpu $GPUS --output ./ --root $celeb --image \
        --tag CAL_32_1_Celeb_LR --resume $checkpoint --eval --LR-MODE --no-classifier --dataset-specific
    # top1:22.9% top5:31.5% top10:35.7% top20:41.2% mAP:2.4%
    ############# Celeb HD only CAL + UBD  ############
    checkpoint=logs/ltcc/CAL_UBD_32_2/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb --gpu $GPUS --output ./ --root $celeb --image \
        --tag CAL_UBD_32_2_Celeb_HD --resume $checkpoint --eval --no-classifier
    # top1:43.0% top5:51.6% top10:56.4% top20:63.4% mAP:5.2%
    ############# Celeb LR only CAL + UBD  ############
    checkpoint=logs/ltcc/CAL_UBD_32_2/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb --gpu $GPUS --output ./ --root $celeb --image \
        --tag CAL_UBD_32_2_Celeb_LR --resume $checkpoint --eval --LR-MODE --no-classifier --dataset-specific
    # top1:31.8% top5:40.9% top10:47.0% top20:53.7% mAP:3.4%
    ############# Celeb HR BASE + UBD  ############
    checkpoint=logs/ltcc_colors/BM_TS_28_2/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --silhouettes=$celeb_sil --root $celeb --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_TS_Celeb_HD --resume $checkpoint --eval --no-classifier
    # top1:45.3% top5:57.0% top10:64.1% top20:71.2% mAP:7.0%   
    ############# Celeb LR BASE + UBD  ############
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --silhouettes=$celeb_sil --root $celeb --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_TS_Celeb_LR --resume $checkpoint --eval --LR-MODE --no-classifier --dataset-specific
    # top1:31.0% top5:44.0% top10:50.7% top20:59.1% mAP:4.1%
    
    python Scripts/analysis/low_res_analysis_Celeb.py  BM_28_1_TS_Celeb_HD BM_28_1_TS_Celeb_LR
    python Scripts/analysis/low_res_analysis_Celeb.py  CAL_32_1_Celeb_HD CAL_32_1_Celeb_LR
    python Scripts/analysis/low_res_analysis_Celeb.py  CAL_UBD_32_2_Celeb_HD CAL_UBD_32_2_Celeb_LR
    python Scripts/analysis/low_res_analysis_Celeb.py  CAL_UBD_32_2_Celeb_LR CAL_32_1_Celeb_LR ubd_lr
#
############ EVAL  ############
    checkpoint=logs/ltcc_colors/BM_28_1/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_LTCC --resume $checkpoint --eval 
    # top1:73.8% top5:83.6% top10:86.4% top20:89.2% mAP:39.3%
    # top1:41.6% top5:55.4% top10:60.2% top20:68.4% mAP:19.0%
    
    
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ \
        --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_CAL --resume $checkpoint --eval --eval-cal
    # top1:72.6% top5:81.5% top10:85.2% top20:88.2% mAP:37.9%
    # top1:36.0% top5:51.0% top10:56.6% top20:64.8% mAP:17.2%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ \
        --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_BOT --resume $checkpoint --eval --eval-sim 
    # top1:68.2% top5:80.5% top10:83.6% top20:88.4% mAP:33.2%
    # top1:33.9% top5:54.6% top10:61.7% top20:70.4% mAP:16.9%
    checkpoint=logs/ltcc_colors/BM_TS_28_1/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ \
        --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag ONLY_DIST_1_28 --resume $checkpoint --eval 
    # top1:73.4% top5:84.6% top10:87.4% top20:90.7% mAP:39.8%    
    # top1:42.6% top5:57.7% top10:62.2% top20:68.9% mAP:20.9%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ \
        --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag ONLY_DIST_1_28 --resume $checkpoint --eval --eval-cal
    # top1:72.4% top5:82.4% top10:85.6% top20:89.7% mAP:38.2%    
    # top1:39.0% top5:53.1% top10:60.2% top20:66.6% mAP:18.5%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ \
        --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag ONLY_DIST_1_28 --resume $checkpoint --eval --eval-sim 
    # top1:70.2% top5:82.2% top10:85.6% top20:89.7% mAP:33.6%
    # top1:41.1% top5:56.9% top10:63.3% top20:71.4% mAP:18.8%

    # BM_28_1
    # LTCC 
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_LTCC --resume $checkpoint --eval 
    # Celeb HD 
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --silhouettes=$celeb_sil --root $celeb --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_Celeb_HD --resume $checkpoint --eval --no-classifier
    # top1:32.3% top5:41.6% top10:47.0% top20:53.3% mAP:3.6%
    # Celeb LR  
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --silhouettes=$celeb_sil --root $celeb --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_Celeb_LR --resume $checkpoint --eval --LR-MODE --no-classifier --dataset-specific
    # top1:16.5% top5:24.6% top10:29.6% top20:37.2% mAP:1.8%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_TS_LTCC --resume $checkpoint --eval 
    # top1:41.6% top5:55.4% top10:60.2% top20:68.4% mAP:19.0%    
    checkpoint=logs/ltcc_colors/BM_TS_28_2/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_2_TS_LTCC --resume $checkpoint --eval 
    # top1:42.6% top5:57.7% top10:62.2% top20:68.9% mAP:20.9%    
    
    
    checkpoint=logs/ltcc/LT_CAL/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal.yaml --dataset ltcc \
        --gpu $GPUS --output ./ --tag LT_CAL --root $ltcc --image --backbone="resnet50" --batch_size 40 --only-CAL --eval --resume $checkpoint 
    # top1:41.3% top5:54.3% top10:59.9% top20:64.3% mAP:18.4%

    DATASET=ltcc_cc_gender
    POSE=$ltcc_pose
    GENDER=$ltcc_gender
    ROOT=$ltcc
    # checkpoint=logs/ltcc_colors/R_LA_15_DS_NC_B=40_2_2/best_model.pth.tar
    checkpoint=logs/ltcc_colors/R_LA_15_DS_NC_B=40_2_2/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LA_15_2_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G --seed=$RUN_NO \
        --eval --resume $checkpoint --tag LT_R_LA_15_DS_NC_B=40_2_2
    # top1:46.7% top5:61.0% top10:65.8% top20:70.7% mAP:21.3%
    # top1:74.2% top5:83.8% top10:87.2% top20:89.5% mAP:40.4%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LA_15_2_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G --seed=$RUN_NO \
        --eval --resume $checkpoint --tag LT_R_LA_15_DS_NC_B=40_2_2_ONLY_CAL --eval-cal
    # top1:39.8% top5:54.3% top10:62.0% top20:69.4% mAP:18.4%
    # top1:72.6% top5:82.6% top10:85.6% top20:89.0% mAP:37.6%  
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LA_15_2_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G --seed=$RUN_NO \
        --eval --resume $checkpoint --tag LT_R_LA_15_DS_NC_B=40_2_2_ONLY_SIM --eval-sim 
    # top1:43.6% top5:58.7% top10:65.6% top20:71.7% mAP:19.7%
    # top1:71.2% top5:81.1% top10:85.6% top20:88.4% mAP:36.3%

    # python Scripts/analysis/analysis.py LT_CAL LTCC /data/priyank/synthetic/LTCC/
    # python Scripts/analysis/analysis.py LT_R_LA_15_DS_NC_B=40_2_2 LTCC /data/priyank/synthetic/LTCC/
    # python Scripts/analysis/compare_mistakes.py LT_CAL LT_R_LA_15_DS_NC_B=40_2_2 LTCC /data/priyank/synthetic/LTCC/
    
    checkpoint=logs/ltcc_cc_gender/R_LA_15_B=32_1_ucf4/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LA_15_2_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G \
        --eval --resume $checkpoint --tag R_LA_15_B=32_1_ucf4
    # top1:46.4% top5:58.4% top10:64.5% top20:70.9% mAP:21.2%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LA_15_2_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G \
        --eval --resume $checkpoint --tag R_LA_15_B=32_1_ucf4_ONLY_CAL --eval-cal
    # top1:37.5% top5:53.8% top10:60.5% top20:67.1% mAP:18.0%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LA_15_2_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G \
        --eval --resume $checkpoint --tag R_LA_15_B=32_1_ucf4_ONLY_SIM --eval-sim 
    # top1:43.1% top5:61.0% top10:66.3% top20:73.7% mAP:19.6%

    checkpoint=logs/ltcc_cc_gender/R_LA_15_B=32_1/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LA_15_2_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G \
        --eval --resume $checkpoint --tag R_LA_15_B=32_1
    # top1:46.4% top5:58.4% top10:64.5% top20:70.9% mAP:21.2%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LA_15_2_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G \
        --eval --resume $checkpoint --tag R_LA_15_B=32_1_ONLY_CAL --eval-cal
    # top1:37.5% top5:53.8% top10:60.5% top20:67.1% mAP:18.0%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LA_15_2_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G \
        --eval --resume $checkpoint --tag R_LA_15_B=32_1_ONLY_SIM --eval-sim 
    # top1:43.1% top5:61.0% top10:66.3% top20:73.7% mAP:19.6%

    DATASET=prcc_cc_gender
    SIL=$prcc_sil
    POSE=$prcc_pose
    GENDER=$prcc_gender
    ROOT=$prcc
    checkpoint=logs/prcc_cc_gender/R_LAC_25_DS_NC_B=40_4/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LAC_25_1_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=26 --Pose=$POSE --pose-mode="R_LAC_25" --overlap_2=-2 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G --seed=$RUN_NO \
        --eval --resume $checkpoint --tag R_LAC_25_DS_NC_B=40_2_2
    # top1:63.3% top5:69.0% top10:71.0% top20:73.5% mAP:59.4%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LAC_25_1_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=26 --Pose=$POSE --pose-mode="R_LAC_25" --overlap_2=-2 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G --seed=$RUN_NO \
        --eval --resume $checkpoint --tag R_LAC_25_DS_NC_B=40_2_2_ONLY_CAL --eval-cal
    # top1:55.0% top5:60.1% top10:62.9% top20:66.3% mAP:50.7%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LAC_25_1_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=26 --Pose=$POSE --pose-mode="R_LAC_25" --overlap_2=-2 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G --seed=$RUN_NO \
        --eval --resume $checkpoint --tag R_LAC_25_DS_NC_B=40_2_2_ONLY_SIM --eval-sim 
    # top1:64.2% top5:69.6% top10:72.3% top20:75.4% mAP:60.4%
    
    checkpoint='logs/prcc_cc_gender/R_LA_15_B=40_1/best_model.pth.tar'
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset prcc_cc_gender --gpu $GPUS --output ./ --root $prcc --image \
        --class_2=16 --Pose=$prcc_pose --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $prcc_gender --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Final_PR_R_LA_15_B=32_1 --resume $checkpoint --eval --no-classifier
    # top1:65.0% top5:70.1% top10:72.1% top20:74.3% mAP:61.0%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset prcc_cc_gender --gpu $GPUS --output ./ --root $prcc --image \
        --class_2=16 --Pose=$prcc_pose --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $prcc_gender --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Final_PR_R_LA_15_B=32_1 --resume $checkpoint --eval --no-classifier --eval-cal
    # top1:59.9% top5:65.3% top10:67.8% top20:69.7% mAP:54.9%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset prcc_cc_gender --gpu $GPUS --output ./ --root $prcc --image \
        --class_2=16 --Pose=$prcc_pose --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $prcc_gender --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Final_PR_R_LA_15_B=32_1 --resume $checkpoint --eval --no-classifier --eval-sim 
    # top1:65.1% top5:71.7% top10:74.8% top20:78.1% mAP:61.5%

    # python Scripts/analysis/compare_mistakes.py R_LAC_25_DS_NC_B=40_2_2 PRCC /data/priyank/synthetic/PRCC/
    
    # no normalization (Concat to get back the features of base model)
    checkpoint=logs/ltcc_colors/BM_28_1/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_LTCC --resume $checkpoint --eval 
    # top1:41.6% top5:55.4% top10:60.2% top20:68.4% mAP:19.0%
    checkpoint=logs/ltcc_colors/BM_28_2/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_2_LTCC --resume $checkpoint --eval 
    # top1:42.1% top5:58.4% top10:63.0% top20:69.9% mAP:20.7%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ \
        --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_CAL --resume $checkpoint --eval --eval-cal --no-normalization
    # top1:34.2% top5:50.3% top10:57.4% top20:64.0% mAP:16.6%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ \
        --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_BOT --resume $checkpoint --eval --eval-sim --no-normalization
    # top1:29.8% top5:49.5% top10:58.2% top20:69.4% mAP:15.7%

#
############ Gender analysis ############
    checkpoint=logs/ltcc/LT_CAL/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ --root $ltcc --image \
        --tag LT_CAL_32_1 --resume $checkpoint --eval --no-classifier
    # top1:41.3% top5:54.3% top10:59.9% top20:64.3% mAP:18.4%
    checkpoint=logs/ltcc_colors/BM_28_2/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ --root $ltcc --image \
        --backbone="resnet50_joint2" --tag LT_BM_28_2 --resume $checkpoint --eval --no-classifier
    # top1:42.1% top5:58.4% top10:63.0% top20:69.9% mAP:20.7%
    checkpoint='logs/ltcc_cc_gender/R_LA_15_B=32_1/best_model.pth.tar'
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_cc_gender --gpu $GPUS --output ./ --root $ltcc --image \
        --class_2=16 --Pose=$ltcc_pose --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $ltcc_gender --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Final_R_LA_15_B=32_1 --resume $checkpoint --eval --no-classifier
    # top1:46.4% top5:58.4% top10:64.5% top20:70.9% mAP:21.3%

    checkpoint=logs/prcc/CAL/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset prcc_colors --gpu $GPUS --output ./ --root $prcc --image \
        --tag PR_CAL_32_1 --resume $checkpoint --eval --no-classifier
    # top1:55.2% top5:59.0% top10:60.9% top20:63.4% mAP:56.1%
    checkpoint=logs/prcc_colors/BM_32_1/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset prcc_colors --gpu $GPUS --output ./ --root $prcc --image \
        --backbone="resnet50_joint2" --tag PR_BM_32_1 --resume $checkpoint --eval --no-classifier
    # top1:59.1% top5:63.2% top10:65.0% top20:67.6% mAP:59.2%
    checkpoint='logs/prcc_cc_gender/R_LA_15_B=40_1/best_model.pth.tar'
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset prcc_cc_gender --gpu $GPUS --output ./ --root $prcc --image \
        --class_2=16 --Pose=$prcc_pose --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $prcc_gender --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Final_PR_R_LA_15_B=32_1 --resume $checkpoint --eval --no-classifier
    # top1:65.0% top5:70.1% top10:72.1% top20:74.3% mAP:61.0%

    python Scripts/analysis/gender_res_analysis.py LT_CAL_32_1 LTCC /data/priyank/synthetic/LTCC/
    python Scripts/analysis/gender_res_analysis.py PR_CAL_32_1 PRCC /data/priyank/synthetic/PRCC/
    

#
############ LowRes analysis ############
    ############# CAL - LTCC 
    checkpoint=logs/ltcc/LT_CAL/best_model.pth.tar
    checkpoint=logs/prcc/CAL/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --tag Celeb_CAL_32_1_HD --resume $checkpoint --eval --no-classifier  >>CAL.txt
    # top1:37.3% top5:44.2% top10:47.6% top20:53.8% mAP:3.9%
    # top1:22.0% top5:27.0% top10:31.2% top20:36.9% mAP:2.0%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --tag Celeb_CAL_32_1_LR --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="LR"  >>CAL.txt
    # top1:25.3% top5:33.0% top10:37.9% top20:43.8% mAP:2.5%
    # top1:17.1% top5:23.3% top10:27.1% top20:32.6% mAP:1.6%
    # -20
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --tag Celeb_CAL_32_1_MB --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="MB"  >>CAL.txt
    # top1:21.9% top5:29.8% top10:34.9% top20:41.3% mAP:2.2%        
    # top1:14.0% top5:20.0% top10:24.0% top20:30.1% mAP:1.5%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --tag Celeb_CAL_32_1_OOF --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="OOF" >>CAL.txt
    # top1:30.3% top5:37.8% top10:43.0% top20:49.1% mAP:3.1%        
    # top1:18.8% top5:26.0% top10:30.0% top20:34.8% mAP:1.8%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --tag Celeb_CAL_32_1_LR --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific
    # top1:22.8% top5:30.4% top10:35.6% top20:41.8% mAP:2.3%

    ############# BM - LTCC 
    checkpoint=logs/ltcc_colors/BM_28_2/best_model.pth.tar
    checkpoint=logs/prcc_colors/BM_32_1/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --backbone="resnet50_joint2" --tag Celeb_BM_28_2_HD --resume $checkpoint --eval --no-classifier >>BM.txt
    # top1:35.7% top5:44.1% top10:48.9% top20:55.5% mAP:3.9%
    # top1:24.7% top5:31.4% top10:35.1% top20:41.4% mAP:2.3%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --backbone="resnet50_joint2" --tag Celeb_BM_28_2_LR --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="LR" >>BM.txt
    # top1:23.6% top5:32.5% top10:37.7% top20:43.8% mAP:2.4%
    # top1:16.9% top5:23.1% top10:27.2% top20:33.0% mAP:1.6%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --backbone="resnet50_joint2" --tag Celeb_BM_28_2_MB --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="MB" >>BM.txt
    # top1:18.8% top5:27.5% top10:31.8% top20:39.1% mAP:1.9%
    # top1:13.8% top5:20.3% top10:25.1% top20:32.3% mAP:1.4%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --backbone="resnet50_joint2" --tag Celeb_BM_28_2_OOF --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="OOF" >>BM.txt
    # top1:28.9% top5:37.9% top10:42.9% top20:49.5% mAP:3.0%
    # top1:19.7% top5:25.8% top10:31.4% top20:36.9% mAP:1.9%
    ############# RLQ - LTCC 
    checkpoint='logs/ltcc_cc_gender/R_LA_15_B=32_1/best_model.pth.tar'
    checkpoint='logs/prcc_cc_gender/R_LA_15_B=40_1/best_model.pth.tar'
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --class_2=16 --overlap_2=-3 --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Celeb_Final_R_LA_15_B=32_1_HD --resume $checkpoint --eval --no-classifier  >> FINAL.txt
    # top1:45.1% top5:57.5% top10:63.9% top20:70.6% mAP:7.1%
    # top1:32.1% top5:40.5% top10:45.5% top20:52.6% mAP:3.3%
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --class_2=16 --overlap_2=-3 --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Celeb_Final_R_LA_15_B=32_1_LR --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="LR" >> FINAL.txt
    # top1:31.1% top5:42.7% top10:49.8% top20:57.5% mAP:3.9%
    # top1:23.1% top5:30.7% top10:35.3% top20:42.2% mAP:2.2%
    # -9
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --class_2=16 --overlap_2=-3 --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Celeb_Final_R_LA_15_B=32_1_MB --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="MB" >> FINAL.txt
    # top1:31.0% top5:41.9% top10:48.1% top20:56.4% mAP:3.7%
    # top1:14.5% top5:21.4% top10:25.6% top20:32.2% mAP:1.4%
    # -16
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --class_2=16 --overlap_2=-3 --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Celeb_Final_R_LA_15_B=32_1_OOF --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="OOF" >> FINAL.txt
    # top1:39.9% top5:52.4% top10:59.0% top20:65.7% mAP:5.6%
    # top1:28.5% top5:36.5% top10:41.7% top20:48.1% mAP:2.8%
    # -4
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --class_2=16 --overlap_2=-3 --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Celeb_Final_R_LA_15_B=32_1_LQ --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific
    # top1:33.0% top5:44.3% top10:50.8% top20:58.7% mAP:4.1%

    
    
    
    
# mv ~/CCReID/*.pkl /data/priyank/synthetic/LTCC/masks/ltcc/
# cd /data/priyank/synthetic/LTCC/masks/ltcc/
# ls
# rsync -a *.pkl ucf:/home/c3-0/datasets/ID-Dataset/ltcc/    

# rsync -a ucf:~/CCReID/ltcc_colors/color_aug/ ~/CCReID/logs/ltcc_colors/color_aug/     
# rsync -a ucf:~/CCReID/ltcc_colors/color_aug_w_bkd/ ~/CCReID/logs/ltcc_colors/color_aug_w_bkd/     

# bash Scripts/analysis.sh