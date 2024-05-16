import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO" 
try:
    from mmcv.runner.utils import set_random_seed
except:
    from mmengine.runner import set_random_seed
set_random_seed(1)

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
from models import build_model, build_extra_id_classifier
from losses import build_losses, build_additional_losses
from tools.utils import save_checkpoint, set_seed, get_logger, save_pickle
from train import *
from test import *

def parse_option():
    parser = argparse.ArgumentParser(description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='ltcc', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    
    # Miscs
    parser.add_argument('--output', type=str, default='logs', help="your output path to save model and logs")    
    parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
    
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--size', type=lambda a: json.loads('['+a.replace(" ",",")+']'), default=None, help="Size")                                      
    parser.add_argument('--image', action='store_true', help="train video as image")
    parser.add_argument('--max_epochs', type=int, help='')
    parser.add_argument('--batch_size', type=int, default=None, help='')
    
    parser.add_argument('--subset-dataset', type=str, default=None)
    parser.add_argument('--silhouettes', type=str, default=None, help="your root path to data silhouettes directory")
    parser.add_argument('--Pose', type=str, default=None)
    parser.add_argument('--pose-mode', type=str, default=None)
    
    # Loading Weights & Eval  
    parser.add_argument('--strict', action='store_true', help="evaluation only")
    parser.add_argument('--no-classifier', action='store_true', help="evaluation only")
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--eval_step', type=int, default=None)
    parser.add_argument('--LR-MODE', action='store_true', help="evaluation only")
    parser.add_argument('--LR-TYPE',  type=str, default=None)
    
    # Train fns 
    parser.add_argument('--pair', action='store_true')
    parser.add_argument('--pair2', action='store_true')
    parser.add_argument('--train_fn', type=str, )
    
    # Train Misc
    parser.add_argument('--sil_mode', type=str, default=None, help="your root path to data silhouettes directory")
    parser.add_argument('--use-validation', type=str, default=None,)
    parser.add_argument('--use_gender', type=str, default=None,)
    parser.add_argument('--cal_on_orig', action='store_true', help="evaluation only")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--only-CAL', action='store_true')

    parser.add_argument('--gender_id', action='store_true')
    parser.add_argument('--gender_clothes', action='store_true')
    parser.add_argument('--gender_overall', action='store_true')

    parser.add_argument('--pose_id', action='store_true')
    parser.add_argument('--pose_cl', action='store_true')
    parser.add_argument('--pose_both', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    
    # losses
    parser.add_argument('--pair_loss', type=str, default=None)
    parser.add_argument('--loss_mode', type=str, default=None)
    parser.add_argument('--additional_loss', type=str, default=None)

    # Models 
    parser.add_argument('--backbone', type=str, default=None)
    parser.add_argument('--sampler', type=str, default=None)
    parser.add_argument('--splits', type=int, default=None)
    
    parser.add_argument('--class_2', type=int, default=None)
    parser.add_argument('--class_3', type=int, default=None)

    parser.add_argument('--overlap_1', type=int, default=None)
    parser.add_argument('--overlap_2', type=int, default=None)
    parser.add_argument('--overlap_3', type=int, default=None)
    
    parser.add_argument('--extra_class_embed', type=int, default=None)
    parser.add_argument('--extra_class_no', type=int, default=None)
    parser.add_argument('--dataset-specific', action='store_true')
    
    return parser
    
def modify_config(config, dataset):
    config.defrost()
    config.DATA.NUM_CLASS = dataset.num_train_pids
    if config.MODEL.Class_2 == -1:
        config.MODEL.Class_2 = dataset.num_train_pids
    config.freeze()

def create_optimizers(config, parameters, clothes_classifier):
    optimizer_cc = None
    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        if clothes_classifier:
            optimizer_cc = optim.Adam(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, 
                                    weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        if clothes_classifier:
            optimizer_cc = optim.AdamW(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, 
                                    weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
        if clothes_classifier:
            optimizer_cc = optim.SGD(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))
    
    # Build lr_scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                         gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    return optimizer, optimizer_cc, scheduler

def resume_model(config, logger, model, classifier, criterion_adv, clothes_classifier):
    if config.MODEL.RESUME:
        revise_keys= [(r'^module.', '')]
        from mmengine.runner.checkpoint import _load_checkpoint_to_model
        logger.info("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        if config.MODEL.RESUME_NO_CLASSIFIER:
            checkpoint['model_state_dict'] = {k:v for k,v in checkpoint['model_state_dict'].items() if (("classifer" not in k) and ("classifier" not in k))}
            # _ = _load_checkpoint_to_model(model, checkpoint['model_state_dict'], config.MODEL.STRICT, revise_keys=revise_keys)
            model.load_state_dict(checkpoint['model_state_dict'], strict=config.MODEL.STRICT)
        else:
            model.load_state_dict(checkpoint['model_state_dict'], strict=config.MODEL.STRICT)
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
            if config.LOSS.CAL == 'calwithmemory':
                criterion_adv.load_state_dict(checkpoint['clothes_classifier_state_dict'])
            else:
                clothes_classifier.load_state_dict(checkpoint['clothes_classifier_state_dict'])
        # start_epoch = checkpoint['epoch']
    return model, classifier, criterion_adv, clothes_classifier

def distributed_model(config, local_rank, model, classifier, criterion_adv, clothes_classifier, find_unused_parameters=False):
    model = model.cuda(local_rank)
    if clothes_classifier is not None:
        if config.LOSS.CAL == 'calwithmemory':
            criterion_adv = criterion_adv.cuda(local_rank)
        else:
            clothes_classifier = clothes_classifier.cuda(local_rank)
    if classifier is not None:
        classifier = classifier.cuda(local_rank)
    torch.cuda.set_device(local_rank)
    # from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
    # model = MMDistributedDataParallel(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False, find_unused_parameters=False)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters)
    if classifier is not None:
        classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank], output_device=local_rank)
    if clothes_classifier is not None:
        if config.LOSS.CAL != 'calwithmemory':
            clothes_classifier = nn.parallel.DistributedDataParallel(clothes_classifier, device_ids=[local_rank], output_device=local_rank)
    return model, classifier, criterion_adv, clothes_classifier

def eval_model(config, logger, model, queryloader, galleryloader, dataset, queryloader_same, queryloader_diff, message=None):
    if message:
        logger.info(message)
    torch.cuda.empty_cache()
    with torch.no_grad():
        if "prcc" in config.DATA.DATASET:
            queryloader_same, queryloader_diff = queryloader
            if config.EVAL_MODE:
                rank1, map_acc, feature_dump = test_prcc(config, model, queryloader_same, queryloader_diff, galleryloader, dataset, dump=True)
                save_pickle(feature_dump, name=config.TAG)
            else:
                rank1, map_acc = test_prcc(config, model, queryloader_same, queryloader_diff, galleryloader, dataset)
        elif config.EVAL_MODE:
            rank1, map_acc, feature_dump = test(config, model, queryloader, galleryloader, dataset, dump=True)
            save_pickle(feature_dump, name=config.TAG)
        elif config.TEST.VALIDATION:
            rank1, map_acc, feature_dump = test(config, model, queryloader, galleryloader, dataset)
        else:
            rank1, map_acc = test(config, model, queryloader, galleryloader, dataset)
        return rank1, map_acc    
        
def save_model(config, model, classifier, criterion_adv, clothes_classifier, 
    local_rank, rank1, epoch, is_best, ):
    model_state_dict = model.module.state_dict()
    if classifier:
        classifier_state_dict = classifier.module.state_dict()
    else:
        classifier_state_dict = {}
    
    clothes_classifier_state_dict = clothes_classifier.module.state_dict()
    if local_rank == 0:
        save_checkpoint({
            'model_state_dict': model_state_dict,
            'classifier_state_dict': classifier_state_dict,
            'clothes_classifier_state_dict': clothes_classifier_state_dict,
            'rank1': rank1,
            'epoch': epoch,
        }, is_best, osp.join(config.OUTPUT, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'),
        normal_save=config.TEST.SAVE_WT, )

        
            
        
            
def main(config):
    # Build dataloader
    trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config)
    modify_config(config, dataset)
    # Define a matrix pid2clothes with shape (num_pids, num_clothes). 
    # pid2clothes[i, j] = 1 when j-th clothes belongs to i-th identity. Otherwise, pid2clothes[i, j] = 0.
    pid2clothes = torch.from_numpy(dataset.pid2clothes)
    # Build model
    model, classifier, clothes_classifier = build_model(config, dataset.num_train_pids, dataset.num_train_clothes)
    if args.extra_class_embed:
        extra_classifier = build_extra_id_classifier(config, args.extra_class_no, input_dim=args.extra_class_embed)
    if config.TRAIN.FN == "2feats_pair4":
        del classifier, clothes_classifier
        classifier, clothes_classifier = None, None
    # Build identity classification loss, pairwise loss, clothes classificaiton loss, and adversarial loss.
    criterion_cla, criterion_pair, criterion_clothes, criterion_adv, criteria_feat_mse, criteria_logit_KL = build_losses(config, dataset.num_train_clothes)
    extra_loss = build_additional_losses(config, dataset.num_train_clothes)

    if config.TRAIN.FN == "2feats_pair31": 
        classifier = None 
    if config.TRAIN.FN in ["2feats_pair31", "2feats_pair32", "2feats_pair33"]:
        criterion_pair = None

    # Build optimizer
    if config.TRAIN.FN == "2feats_pair4":
        parameters = list(model.parameters())
    else:
        if args.extra_class_embed:
            parameters = list(model.parameters()) + list(classifier.parameters()) + list(extra_classifier.parameters())
        else:
            parameters = list(model.parameters()) + list(classifier.parameters())
    optimizer, optimizer_cc, scheduler = create_optimizers(config, parameters, clothes_classifier)

    start_epoch = config.TRAIN.START_EPOCH
    model, classifier, criterion_adv, clothes_classifier = resume_model(config, logger, model, classifier, criterion_adv, clothes_classifier)
    
    local_rank = dist.get_rank()
    
    model, classifier, criterion_adv, clothes_classifier = distributed_model(config, local_rank, model, classifier, criterion_adv, clothes_classifier)
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

    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        train_sampler.set_epoch(epoch)
        trainloader.dataset.set_epoch(epoch)

        start_train_time = time.time()

        if config.TRAIN.FN == "2feats_pair27": # Baseline (Only CAL) [Foreground Aug]
            train_cal_pair27_ind_2feat(epoch=epoch, **params, **additional_args)
        elif config.TRAIN.FN == "2feats_pair3":# Baseline
            train_cal_pair3_ind_2feat(epoch=epoch, **params, **additional_args)
        elif config.TRAIN.FN == "2feats_pair14": # Baseline + Gender
            train_cal_pair14_ind_2feat(epoch=epoch, **params, **additional_args)
        elif config.TRAIN.FN == "2feats_pair16": # Baseline + Pose
            train_cal_pair16_ind_2feat(epoch=epoch, **params, **additional_args)
        elif config.TRAIN.FN == "2feats_pair23": # Gender + Pose 
            train_cal_pair23_ind_2feat(epoch=epoch, **params, **additional_args)
        elif config.TRAIN.FN == "2feats_pair4":
            train_cal_pair4_ind_2feat(epoch=epoch, **params, **additional_args)
        else: # vanilla CAL 
            train_cal(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
                criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes)
        

        train_time += round(time.time() - start_train_time)        
        scheduler.step()
    
        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
            (epoch+1) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:

            rank1, map_acc = eval_model(config, logger, model, queryloader, galleryloader, dataset, None, None, message="==> Test")
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
    
    logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}. Best MaP {:.1%}".format(best_rank1, best_epoch, best_map))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    logger.info("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    

if __name__ == '__main__':
    
    parser = parse_option()
    args, unparsed = parser.parse_known_args()
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
    
    main(config)