from torch import nn
from losses.cross_entropy_loss_with_label_smooth import CrossEntropyWithLabelSmooth
from losses.triplet_loss import TripletLoss
from losses.contrastive_loss import ContrastiveLoss
from losses.arcface_loss import ArcFaceLoss
from losses.cosface_loss import CosFaceLoss, PairwiseCosFaceLoss
from losses.circle_loss import CircleLoss, PairwiseCircleLoss
from losses.clothes_based_adversarial_loss import ClothesBasedAdversarialLoss, ClothesBasedAdversarialLossWithMemoryBank
from losses.custom import *

from torch import distributed as dist

def build_losses(config, num_train_clothes):
    # Build identity classification loss
    if config.LOSS.CLA_LOSS == 'crossentropy':
        criterion_cla = nn.CrossEntropyLoss()
    elif config.LOSS.CLA_LOSS == 'crossentropylabelsmooth':
        criterion_cla = CrossEntropyWithLabelSmooth()
    elif config.LOSS.CLA_LOSS == 'arcface':
        criterion_cla = ArcFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'cosface':
        criterion_cla = CosFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'circle':
        criterion_cla = CircleLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    else:
        raise KeyError("Invalid classification loss: '{}'".format(config.LOSS.CLA_LOSS))

    # Build pairwise loss
    if config.LOSS.PAIR_LOSS == 'triplet':
        criterion_pair = TripletLoss(margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'contrastive':
        criterion_pair = ContrastiveLoss(scale=config.LOSS.PAIR_S)
    elif config.LOSS.PAIR_LOSS == 'cosface':
        criterion_pair = PairwiseCosFaceLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'circle':
        criterion_pair = PairwiseCircleLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'center':
        criterion_pair = Center_Loss(num_classes=config.DATA.NUM_CLASS, feat=config.MODEL.FEATURE_DIM, metric=config.LOSS.LOSS_MODE)
        local_rank = dist.get_rank()
        criterion_pair = criterion_pair.cuda(local_rank)
        criterion_pair = nn.parallel.DistributedDataParallel(criterion_pair, device_ids=[local_rank], output_device=local_rank)
    elif config.LOSS.PAIR_LOSS == 'none':
        criterion_pair = None 
    else:
        raise KeyError("Invalid pairwise loss: '{}'".format(config.LOSS.PAIR_LOSS))

    # Build clothes classification loss
    if config.LOSS.CLOTHES_CLA_LOSS == 'crossentropy':
        criterion_clothes = nn.CrossEntropyLoss()
    elif config.LOSS.CLOTHES_CLA_LOSS == 'cosface':
        criterion_clothes = CosFaceLoss(scale=config.LOSS.CLA_S, margin=0)
    else:
        raise KeyError("Invalid clothes classification loss: '{}'".format(config.LOSS.CLOTHES_CLA_LOSS))

    # Build clothes-based adversarial loss
    if config.LOSS.CAL == 'cal':
        criterion_cal = ClothesBasedAdversarialLoss(scale=config.LOSS.CLA_S, epsilon=config.LOSS.EPSILON)
    elif config.LOSS.CAL == 'calwithmemory':
        criterion_cal = ClothesBasedAdversarialLossWithMemoryBank(num_clothes=num_train_clothes, feat_dim=config.MODEL.FEATURE_DIM,
                             momentum=config.LOSS.MOMENTUM, scale=config.LOSS.CLA_S, epsilon=config.LOSS.EPSILON)
    else:
        raise KeyError("Invalid clothing classification loss: '{}'".format(config.LOSS.CAL))

    criteria_feat_mse = None 
    criteria_logit_KL = KL_Loss()
    
    return criterion_cla, criterion_pair, criterion_clothes, criterion_cal, criteria_feat_mse, criteria_logit_KL


def build_classification_loss(config):
    if config.LOSS.CLOTHES_CLA_LOSS == 'crossentropy':
        criterion_clothes = nn.CrossEntropyLoss()
    elif config.LOSS.CLOTHES_CLA_LOSS == 'cosface':
        criterion_clothes = CosFaceLoss(scale=config.LOSS.CLA_S, margin=0)
    else:
        raise KeyError("Invalid clothes classification loss: '{}'".format(config.LOSS.CLOTHES_CLA_LOSS))
    return criterion_clothes

def build_pair_loss(config, args):
    criteria_feat_mse = MSE(mode=config.LOSS.LOSS_MODE)
    criteria_logit_KL = KL_Loss()        
    return criteria_feat_mse, criteria_logit_KL

def null_losses(**kwargs):
    return 0
    

def build_additional_losses(config, num_train_clothes):
    local_rank = dist.get_rank()
    if config.LOSS.ADDITIONAL_LOSS == None:
        return null_losses
    elif config.LOSS.ADDITIONAL_LOSS == "Maximize_pose_dist":
        return Maximize_pose_dist()
    elif config.LOSS.ADDITIONAL_LOSS == "Maximize_pose_dist2":
        return Maximize_pose_dist2()
    elif config.LOSS.ADDITIONAL_LOSS == "Maximize_pose_dist3":
        return Maximize_pose_dist3()
    elif config.LOSS.ADDITIONAL_LOSS == "Maximize_pose_dist4":
        return Maximize_pose_dist4(dim = config.MODEL.FEATURE_DIM)
    elif config.LOSS.ADDITIONAL_LOSS == "Pose_TripletLoss":
        return Pose_TripletLoss(margin=config.LOSS.PAIR_M)
    elif config.LOSS.ADDITIONAL_LOSS == 'triplet':
        return TripletLoss(margin=config.LOSS.PAIR_M)
    elif config.LOSS.ADDITIONAL_LOSS == 'kl_o_oid':
        return KL_Loss_o_oid()    
    elif config.LOSS.ADDITIONAL_LOSS == 'center_id':
        criterion = Center_ID(num_classes=config.DATA.NUM_CLASS, feat=config.MODEL.FEATURE_DIM, metric=config.LOSS.LOSS_MODE)
        return nn.parallel.DistributedDataParallel(criterion.cuda(local_rank), device_ids=[local_rank], output_device=local_rank)
    elif config.LOSS.ADDITIONAL_LOSS == 'center_cl':
        criterion = Center_CL(num_classes=config.DATA.NUM_CLASS, feat=config.MODEL.FEATURE_DIM, metric=config.LOSS.LOSS_MODE)
        return nn.parallel.DistributedDataParallel(criterion.cuda(local_rank), device_ids=[local_rank], output_device=local_rank)
    elif config.LOSS.ADDITIONAL_LOSS == 'center_conc':
        criterion = Center_Conc(num_classes=config.DATA.NUM_CLASS, feat=config.MODEL.FEATURE_DIM * 2, metric=config.LOSS.LOSS_MODE)
        return nn.parallel.DistributedDataParallel(criterion.cuda(local_rank), device_ids=[local_rank], output_device=local_rank)
    elif config.LOSS.ADDITIONAL_LOSS == 'center_sep':
        criterion = Center_Sep(num_classes=config.DATA.NUM_CLASS, feat=config.MODEL.FEATURE_DIM, metric=config.LOSS.LOSS_MODE)
        return nn.parallel.DistributedDataParallel(criterion.cuda(local_rank), device_ids=[local_rank], output_device=local_rank)
    elif config.LOSS.ADDITIONAL_LOSS == 'relation':
        return Relation_loss(num_classes=config.DATA.NUM_CLASS)
    elif config.LOSS.ADDITIONAL_LOSS == 'center_cl_kl_o_oid':
        criterion = Center_CL_KL_OOID(num_classes=config.DATA.NUM_CLASS, feat=config.MODEL.FEATURE_DIM, metric=config.LOSS.LOSS_MODE)
        return nn.parallel.DistributedDataParallel(criterion.cuda(local_rank), device_ids=[local_rank], output_device=local_rank)
    elif config.LOSS.ADDITIONAL_LOSS == 'center_sep_kl_o_oid':
        criterion = Center_Sep_KL_OOID(num_classes=config.DATA.NUM_CLASS, feat=config.MODEL.FEATURE_DIM, metric=config.LOSS.LOSS_MODE)
        return nn.parallel.DistributedDataParallel(criterion.cuda(local_rank), device_ids=[local_rank], output_device=local_rank)
    elif config.LOSS.ADDITIONAL_LOSS == 'center_id_kl_o_oid':
        criterion = Center_ID_KL_OOID(num_classes=config.DATA.NUM_CLASS, feat=config.MODEL.FEATURE_DIM, metric=config.LOSS.LOSS_MODE)
        return nn.parallel.DistributedDataParallel(criterion.cuda(local_rank), device_ids=[local_rank], output_device=local_rank)
    
    elif config.LOSS.ADDITIONAL_LOSS == 'Pose_kl_o_oid': # "Maximize_pose_dist"
        return KL_Pose(pose_id=config.TRAIN.POSE_ID, pose_cl=config.TRAIN.POSE_CL, pose_both=config.TRAIN.POSE_BOTH)    
    elif config.LOSS.ADDITIONAL_LOSS == 'Pose3_kl_o_oid': # "Maximize_pose_dist3"
        return KL_Pose3(pose_id=config.TRAIN.POSE_ID, pose_cl=config.TRAIN.POSE_CL, pose_both=config.TRAIN.POSE_BOTH)      
        
        
    
    
