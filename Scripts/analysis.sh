cd ~/RLQ-CGAL-UBD/
conda activate bert2


Celeb_Wt_KL=logs/celeb/B=40_KL_4/checkpoint_ep200.pth.tar
R_LA_15_2_ABS_GID=logs/celeb_cc_colors/R_LA_15_2_ABS_GID/best_model.pth.tar

ltcc=/data/priyank/synthetic/LTCC/
ltcc_sil="/data/priyank/synthetic/LTCC/masks/ltcc/" 
ltcc_enchanced=/data/priyank/synthetic/LTCC/Enhanced
ltcc_pose=/data/priyank/synthetic/LTCC/AlphaPose/2DPose/
ltcc_gender=Scripts/Helper/LTCC_Gender.csv

ltcc_v1=Helper/LTCC_Validation_1.csv
ltcc_v2=Helper/LTCC_Validation_2.csv
ltcc_v3=Helper/LTCC_Validation_3.csv

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



############ Evaluation on Celeb ReID (Not trained on Celeb)
    ############ only CAL  ############
    # Celeb HQ
    checkpoint=logs/ltcc/LT_CAL/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb --gpu $GPUS --output ./ --root $celeb --image \
        --tag CAL_32_1_Celeb_HD --resume $checkpoint --eval --no-classifier
    # Celeb LQ
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb --gpu $GPUS --output ./ --root $celeb --image \
        --tag CAL_32_1_Celeb_LR --resume $checkpoint --eval --LR-MODE --no-classifier --dataset-specific
    ############ CAL + UBD  ############
    # Celeb HQ
    checkpoint=logs/ltcc/CAL_UBD_32_2/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb --gpu $GPUS --output ./ --root $celeb --image \
        --tag CAL_UBD_32_2_Celeb_HD --resume $checkpoint --eval --no-classifier
    # Celeb LQ
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb --gpu $GPUS --output ./ --root $celeb --image \
        --tag CAL_UBD_32_2_Celeb_LR --resume $checkpoint --eval --LR-MODE --no-classifier --dataset-specific    
    ############ BASE Model + UBD ############
    # Celeb HQ
    checkpoint=logs/ltcc_colors/BM_TS_28_2/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --silhouettes=$celeb_sil --root $celeb --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_TS_Celeb_HD --resume $checkpoint --eval --no-classifier
    # Celeb LQ
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --silhouettes=$celeb_sil --root $celeb --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_TS_Celeb_LR --resume $checkpoint --eval --LR-MODE --no-classifier --dataset-specific
    ############ BASE Model ############
    # Celeb HQ 
    checkpoint=logs/ltcc_colors/BM_28_1/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --silhouettes=$celeb_sil --root $celeb --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_Celeb_HD --resume $checkpoint --eval --no-classifier
    # Celeb LQ  
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --silhouettes=$celeb_sil --root $celeb --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_Celeb_LR --resume $checkpoint --eval --LR-MODE --no-classifier --dataset-specific
    
############ Evaluation on LTCC
    # Base Model 
    checkpoint=logs/ltcc_colors/BM_28_1/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag BM_28_1_LTCC --resume $checkpoint --eval 
    
    # Base Model + UBD 
    checkpoint=logs/ltcc_colors/BM_TS_28_1/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset ltcc_colors --gpu $GPUS --output ./ --silhouettes=$ltcc_sil --root $ltcc --image --sil_mode "foreround_overlap" --backbone="resnet50_joint2" \
        --tag ONLY_DIST_1_28 --resume $checkpoint --eval 
    
    # Only CAL
    checkpoint=logs/ltcc/LT_CAL/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal.yaml --dataset ltcc \
        --gpu $GPUS --output ./ --tag LT_CAL --root $ltcc --image --backbone="resnet50" --batch_size 40 --only-CAL --eval --resume $checkpoint 
    
    # RQL-1 
    checkpoint=logs/ltcc_colors/R_LA_15_DS_NC_B=40_2_2/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LA_15_2_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G --seed=$RUN_NO \
        --eval --resume $checkpoint --tag LT_R_LA_15_DS_NC_B=40_2_2
    # RQL-2
    checkpoint=logs/ltcc_cc_gender/R_LA_15_B=32_1_ucf4/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT teacher_student.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset $DATASET \
        --gpu $GPUS --output ./ --root $ROOT --image --teacher-diff "resnet50_joint3_8" --backbone="resnet50_joint3_3" --batch_size 40 --train_fn="2feats_pair23" --teacher_wt $R_LA_15_2_ABS_GID --teacher_dataset celeb --teacher_dir $celeb --class_2=16 --Pose=$POSE --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $GENDER --extra_class_embed 4096 --extra_class_no 2 --gender_id --T-P-G \
        --eval --resume $checkpoint --tag R_LA_15_B=32_1_ucf4
    
############ Evaluation on PRCC
    DATASET=prcc_cc_gender
    SIL=$prcc_sil
    POSE=$prcc_pose
    GENDER=$prcc_gender
    ROOT=$prcc
    # RQL-1 
    checkpoint='logs/prcc_cc_gender/R_LA_15_B=40_1/best_model.pth.tar'
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset prcc_cc_gender --gpu $GPUS --output ./ --root $prcc --image \
        --class_2=16 --Pose=$prcc_pose --pose-mode="R_LA_15" --overlap_2=-3 --use_gender $prcc_gender --extra_class_embed 4096 --extra_class_no 2 --gender_id --backbone="resnet50_joint3_3" --tag Final_PR_R_LA_15_B=32_1 --resume $checkpoint --eval --no-classifier
    # top1:65.0% top5:70.1% top10:72.1% top20:74.3% mAP:61.0%
    
    # CAL 
    checkpoint=logs/prcc/CAL/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset prcc_colors --gpu $GPUS --output ./ --root $prcc --image \
        --tag PR_CAL_32_1 --resume $checkpoint --eval --no-classifier
    
    # Base Model  
    checkpoint=logs/prcc_colors/BM_32_1/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset prcc_colors --gpu $GPUS --output ./ --root $prcc --image \
        --backbone="resnet50_joint2" --tag PR_BM_32_1 --resume $checkpoint --eval --no-classifier
    

############ Low Quality analysis
    ############ CAL (LTCC)
    # HQ
    checkpoint=logs/ltcc/LT_CAL/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --tag Celeb_CAL_32_1_HD --resume $checkpoint --eval --no-classifier 

    # LQ : Low resolution 
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --tag Celeb_CAL_32_1_LR --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="LR"  
    
    # LQ : Motion Blur  
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --tag Celeb_CAL_32_1_MB --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="MB"  
    
    # LQ : OOF 
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --tag Celeb_CAL_32_1_OOF --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="OOF" >>CAL.txt
    
    # LQ randomly selected 
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --tag Celeb_CAL_32_1_LR --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific
    
    ############# BM (LTCC)
    # HQ
    checkpoint=logs/ltcc_colors/BM_28_2/best_model.pth.tar
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --backbone="resnet50_joint2" --tag Celeb_BM_28_2_HD --resume $checkpoint --eval --no-classifier
    
    # LQ : Low resolution 
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --backbone="resnet50_joint2" --tag Celeb_BM_28_2_LR --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="LR" 
    
    # LQ : Motion Blur  
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --backbone="resnet50_joint2" --tag Celeb_BM_28_2_MB --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="MB" 
    
    # LQ : OOF 
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --backbone="resnet50_joint2" --tag Celeb_BM_28_2_OOF --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="OOF" 
    
    ############# RLQ (LTCC)
    checkpoint='logs/ltcc_cc_gender/R_LA_15_B=32_1/best_model.pth.tar'
    # HQ
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --class_2=16 --overlap_2=-3 --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Celeb_Final_R_LA_15_B=32_1_HD --resume $checkpoint --eval --no-classifier 
    
    # LQ : Low resolution 
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --class_2=16 --overlap_2=-3 --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Celeb_Final_R_LA_15_B=32_1_LR --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="LR" 
    
    # LQ : Motion Blur  
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --class_2=16 --overlap_2=-3 --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Celeb_Final_R_LA_15_B=32_1_MB --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="MB" 
    
    # LQ : OOF 
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --class_2=16 --overlap_2=-3 --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Celeb_Final_R_LA_15_B=32_1_OOF --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific --LR-TYPE="OOF" 
    
    # LQ randomly selected 
    CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT main.py --cfg configs/res50_cels_cal_tri_16x4.yaml --dataset celeb_colors --gpu $GPUS --output ./ --root $celeb --image \
        --class_2=16 --overlap_2=-3 --extra_class_embed 4096 --extra_class_no 2 --gender_id \
        --backbone="resnet50_joint3_3" --tag Celeb_Final_R_LA_15_B=32_1_LQ --resume $checkpoint --eval --no-classifier --LR-MODE --dataset-specific
    
    
# bash Scripts/analysis.sh