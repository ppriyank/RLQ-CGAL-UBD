import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO" 
import sys
import time
import datetime
import argparse
import logging
import os.path as osp
import numpy as np
import json 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import distributed as dist

from configs.default_img import get_img_config
from data import build_dataloader
from models import *
from losses import *
from tools.utils import save_checkpoint, set_seed, get_logger, save_pickle
from train import *
from test import *

from main import parse_option, modify_config
import json

from main import parse_option, modify_config, save_model, \
    create_optimizers, resume_model, distributed_model, eval_model


def additional_argument(parser):

    parser.add_argument('--teacher_wt', type=str, default=None)
    parser.add_argument('--teacher_dataset', type=str, default=None)
    parser.add_argument('--teacher_dir', type=str, default=None)
    parser.add_argument('--Debug', action='store_true')
    
    parser.add_argument('--teacher-diff', type=str, default=None)
    parser.add_argument('--unused_param', action='store_true')
    parser.add_argument('--T-P-G', action='store_true')

    parser.add_argument('--sampling', type=int, default=None)

    
    return parser

def main(config, args):
    student_model = None  
    config.defrost()
    config.DATA.DATASET = args.teacher_dataset
    config.DATA.ROOT = args.teacher_dir
    student_model = config.MODEL.NAME
    if args.teacher_diff:
        config.MODEL.NAME = args.teacher_diff
    config.freeze()
    

    if args.sampling:
        print(" ... USING SAMPLED TEACHER SET .... ")
        trainloader_teacher, _, _, dataset_teacher, _ = build_dataloader(config, args.sampling)
    else:
        trainloader_teacher, _, _, dataset_teacher, _ = build_dataloader(config, )
    modify_config(config, dataset_teacher)
    
    teacher_model, teacher_classifier, teacher_clothes_classifier = build_model(config, dataset_teacher.num_train_pids, dataset_teacher.num_train_clothes)
    teacher_id_classifier = None
    if not config.TRAIN.ONLY_CAL:
        teacher_id_classifier = build_extra_id_classifier(config, dataset_teacher.num_train_pids)
        teacher_id_classifier.eval()
    teacher_model.eval()
    teacher_classifier.eval()
    teacher_clothes_classifier.eval()

    assert config.MODEL.TEACHER_WT, f"Teacher can't be loaded {config.MODEL.TEACHER_WT}"
    logger.info("Loading checkpoint from '{}'".format(config.MODEL.TEACHER_WT))
    checkpoint = torch.load(config.MODEL.TEACHER_WT)
    new_checkpoint = {}
    classificaiton_checkpoint = {}
    if args.T_P_G:
        for key in checkpoint['model_state_dict']:
            if "classifier" in key and "identity_classifier2" not in key:
                new_key = key.replace("identity_classifier.", "")
                classificaiton_checkpoint[new_key] = checkpoint['model_state_dict'][key]
            else:
                new_checkpoint[key] = checkpoint['model_state_dict'][key]
    else:    
        for key in checkpoint['model_state_dict']:
            if "classifier" in key:
                new_key = key.replace("identity_classifier.", "")
                classificaiton_checkpoint[new_key] = checkpoint['model_state_dict'][key]
            else:
                new_checkpoint[key] = checkpoint['model_state_dict'][key]
    teacher_model.load_state_dict(new_checkpoint, strict=True)
    if not config.TRAIN.ONLY_CAL:
        teacher_id_classifier.load_state_dict(classificaiton_checkpoint, strict=True)
    teacher_classifier.load_state_dict(checkpoint['classifier_state_dict'])
    if not args.T_P_G:
        teacher_clothes_classifier.load_state_dict(checkpoint['clothes_classifier_state_dict'])
    else:    
        del teacher_model.branch3
        del teacher_model.identity_classifier2
        

    config.defrost()
    config.DATA.DATASET = args.dataset
    config.DATA.ROOT = args.root
    config.MODEL.NAME = student_model
    config.freeze()

    config.defrost()
    config.MODEL.TEACHER_MODE = False
    config.freeze()

    trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config)
    modify_config(config, dataset)
    pid2clothes = torch.from_numpy(dataset.pid2clothes)
    
    model, classifier, clothes_classifier = build_model(config, dataset.num_train_pids, dataset.num_train_clothes)
    if args.extra_class_embed:
        extra_classifier = build_extra_id_classifier(config, args.extra_class_no, input_dim=args.extra_class_embed)

    # Build identity classification loss, pairwise loss, clothes classificaiton loss, and adversarial loss.
    criterion_cla, criterion_pair, criterion_clothes, criterion_adv, criteria_feat_mse, criteria_logit_KL = build_losses(config, dataset.num_train_clothes)
    criteria_DL_mse, criteria_DL_KL = build_pair_loss(config, args)
    extra_loss = build_additional_losses(config, dataset.num_train_clothes)

    # Build optimizer
    if args.extra_class_embed:
        parameters = list(model.parameters()) + list(classifier.parameters()) + list(extra_classifier.parameters())
    else:
        parameters = list(model.parameters()) + list(classifier.parameters())
    optimizer, optimizer_cc, scheduler = create_optimizers(config, parameters, clothes_classifier)
                                     
    start_epoch = config.TRAIN.START_EPOCH
    model, classifier, criterion_adv, clothes_classifier = resume_model(config, logger, model, classifier, criterion_adv, clothes_classifier)
    
    local_rank = dist.get_rank()
    model, classifier, criterion_adv, clothes_classifier = distributed_model(config, local_rank, model, classifier, criterion_adv, clothes_classifier, find_unused_parameters=args.unused_param)
    teacher_model, teacher_id_classifier, _, teacher_clothes_classifier = distributed_model(config, local_rank, teacher_model, teacher_id_classifier, None, teacher_clothes_classifier, find_unused_parameters=args.unused_param)
    
    teacher_classifier = teacher_classifier.cuda(local_rank)
    torch.cuda.set_device(local_rank)
    teacher_classifier = nn.parallel.DistributedDataParallel(teacher_classifier, device_ids=[local_rank], output_device=local_rank)
    
    
    if args.extra_class_embed:
        extra_classifier = extra_classifier.cuda(local_rank)
        extra_classifier = nn.parallel.DistributedDataParallel(extra_classifier, device_ids=[local_rank], output_device=local_rank)

    additional_args = {}
    if config.DATA.SIL_MODE is None:
        additional_args["single_data"] = True 

    if config.EVAL_MODE:
        eval_model(config, logger, model, queryloader, galleryloader, dataset, queryloader_same=None, queryloader_diff=None, message="Evaluate only")
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_map = -np.inf
    best_epoch = 0
    logger.info("==> Start training")
    
    params = dict(config=config, model=model, classifier=classifier, clothes_classifier=clothes_classifier, criterion_cla=criterion_cla,
        criterion_pair=criterion_pair, criterion_clothes=criterion_clothes, criterion_adv=criterion_adv, optimizer=optimizer, optimizer_cc=optimizer_cc, 
        trainloader=trainloader, pid2clothes=pid2clothes, criteria_feat_mse=criteria_feat_mse, criteria_logit_KL=criteria_logit_KL, extra_loss=extra_loss)
    
    Teacher_params = dict(trainloader_teacher=trainloader_teacher, teacher_model=teacher_model, teacher_classifier=teacher_classifier, teacher_id_classifier=teacher_id_classifier)
    DL_loss =  dict(criteria_DL_mse=criteria_DL_mse, criteria_DL_KL=criteria_DL_KL)
    
    if args.extra_class_embed:
        params["extra_classifier"] = extra_classifier
    
    if args.gender_id or args.gender_clothes or args.gender_overall:
        if args.gender_id:
            params["gender_id"] = True 
        elif args.gender_clothes:
            params["gender_clothes"] = True 
        elif args.gender_overall:
            params["gender_overall"] = True 
        params["use_gender"] = True 

    if config.TRAIN.ONLY_CAL:
        params["teacher_clothes_classifier"] = teacher_clothes_classifier
        
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        train_sampler.set_epoch(epoch)
        start_train_time = time.time()
        
        if not args.Debug:
            if config.TRAIN.ONLY_CAL:
                train_cal_pair26_ind_2feat(epoch=epoch,  **params, **Teacher_params, **DL_loss, **additional_args)    
            
            elif "T_P_G" in args and args.T_P_G and config.MODEL.NAME == "resnet50_joint3_5":
                train_cal_pair30_ind_2feat(epoch=epoch,  **params, **Teacher_params, **DL_loss, **additional_args)    
            
            else:    
                train_cal_pair9_ind_2feat(epoch=epoch,  **params, **Teacher_params, **DL_loss, **additional_args)    
                # train_cal_pair9_ind_2feat2(epoch=epoch,  **params, **Teacher_params, **DL_loss, **additional_args)    
                
        if config.TRAIN.FN == "2feats_pair3": # Baseline
            train_cal_pair3_ind_2feat(epoch=epoch, **params, **additional_args)
        elif config.TRAIN.FN == "2feats_pair14": # Baseline + Gender
            train_cal_pair14_ind_2feat(epoch=epoch, **params, **additional_args)
        elif config.TRAIN.FN == "2feats_pair16": # Baseline + Pose
            train_cal_pair16_ind_2feat(epoch=epoch, **params, **additional_args)
        elif config.TRAIN.FN == "2feats_pair23": # Gender + Pose 
            train_cal_pair23_ind_2feat(epoch=epoch, **params, **additional_args)
        elif config.TRAIN.FN == "2feats_pair27": # Baseline (Only CAL)
            assert config.TRAIN.ONLY_CAL 
            train_cal_pair27_ind_2feat()
        else:
            train_cal(epoch=epoch, **params, **additional_args)
        
            
        
        train_time += round(time.time() - start_train_time)        
        
        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
            (epoch+1) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:

            rank1, map_acc = eval_model(config, logger, model, queryloader, galleryloader, dataset, None, None, message="==> Test")
            torch.cuda.empty_cache()

            torch.cuda.empty_cache()
            is_best = rank1 > best_rank1
            if not is_best and rank1 == best_rank1:
                if map_acc > best_map:
                    is_best = True 
            best_map = max(best_map, map_acc)
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1
            
            save_model(config, model, classifier, criterion_adv, clothes_classifier, local_rank, rank1, epoch, is_best, )

        scheduler.step()

    logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}. Best MaP {:.1%}".format(best_rank1, best_epoch, best_map))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    logger.info("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    

if __name__ == '__main__':
    
    parser = parse_option()
    parser = additional_argument(parser)

    args, unparsed = parser.parse_known_args()
    print(args)
    config = get_img_config(args)

    # Set GPU
    torch.backends.cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    # Init dist
    dist.init_process_group(backend="nccl", init_method='env://')
    local_rank = dist.get_rank()
    # Set random seed
    set_seed(config.SEED + local_rank)
    print(f"Changing the seed..... {config.SEED}")
    # get logger

    if not config.EVAL_MODE:
        output_file = osp.join(config.OUTPUT, 'log_train_.log')
    else:
        output_file = osp.join(config.OUTPUT, 'log_test.log')
    logger = get_logger(output_file, local_rank, 'reid')
    logger.info("Config:\n-----------------------------------------")
    logger.info(config)
    logger.info("-----------------------------------------")
    
    main(config, args)