import time
import datetime
import logging
import torch
from tools.utils import AverageMeter, rearrange, repeat, normalize, rearrange_mlr, expand_input, save_image, reverse_arrange
from torch.nn import functional as F
from mmagic.models.utils import set_requires_grad
from data.img_transforms import RandomErasing_colors
from losses.custom import cosine_dist, features_avg_pid, pairwise_l2_dist
import torch.nn as nn 


def print_logs(logger, epoch, batch_time, data_time, batch_cla_loss, batch_pair_loss,
    batch_clo_loss, batch_adv_loss, corrects, clothes_corrects):
    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.4f} '
                  'PairLoss:{pair_loss.avg:.4f} '
                  'CloLoss:{clo_loss.avg:.4f} '
                  'AdvLoss:{adv_loss.avg:.4f} '
                  'Acc:{acc.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, 
                   cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                   clo_loss=batch_clo_loss, adv_loss=batch_adv_loss, 
                   acc=corrects, clo_acc=clothes_corrects))

class Stats:
    def __init__(self, ):
        self.corrects = AverageMeter()
        self.clothes_corrects = AverageMeter()
        self.batch_cla_loss = AverageMeter()
        self.batch_pair_loss = AverageMeter()
        self.batch_clo_loss = AverageMeter()
        self.batch_adv_loss = AverageMeter()
        
    def update(self, preds, pids, clothes_preds, clothes_ids, cla_loss, pair_loss, clothes_loss, adv_loss):
        # statistics
        self.corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        self.clothes_corrects.update(torch.sum(clothes_preds == clothes_ids.data).float()/clothes_ids.size(0), clothes_ids.size(0))
        self.batch_cla_loss.update(cla_loss.item(), pids.size(0))
        self.batch_pair_loss.update(pair_loss.item(), pids.size(0))
        self.batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        self.batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))

class Stats2:
    def __init__(self, keys={}):
        self.metrics = {}
        for key in keys:
            self.metrics[key] = AverageMeter()
        
    def update(self, batch_size, loss_dict):
        # statistics
        for key in loss_dict:
            self.metrics[key].update(loss_dict[key].item(), batch_size)

def clothes_discriminator_update(criterion_clothes, pred_clothes, clothes_ids, epoch, START_EPOCH_CC, optimizer_cc,):
    # Update the clothes discriminator
    clothes_loss = criterion_clothes(pred_clothes, clothes_ids)
    if epoch >= START_EPOCH_CC:
        optimizer_cc.zero_grad()
        clothes_loss.backward()
        optimizer_cc.step()
    return clothes_loss

def handle_replica_data(imgs, pids, clothes_ids, pid2clothes, ):
    B = imgs.shape[0]
    N_replicas = imgs.shape[1] -1
    pids = expand_input(pids, N_replicas + 1)

    imgs = rearrange_mlr(imgs)
    clothes_ids = rearrange_mlr(clothes_ids)
    # save_image(normalize(imgs), "t1.png")
    
    pos_mask = pid2clothes[pids.cpu()]
    imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()

    return imgs, pids, clothes_ids, pos_mask, B, N_replicas 

def compute_loss(criterion_pair, criterion_adv, features, pids, new_pred_clothes, clothes_ids, pos_mask,
    epoch, START_EPOCH_ADV, PAIR_LOSS_WEIGHT, additional_loss, optimizer, **kwargs):
    if criterion_pair and PAIR_LOSS_WEIGHT:
        pair_loss = criterion_pair(features, pids, **kwargs)
    else:
        pair_loss = torch.tensor(0).to(pids.device)
    if criterion_adv:
        adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)
    else:
        adv_loss = 0 
    if epoch >= START_EPOCH_ADV:
        loss = additional_loss + adv_loss + PAIR_LOSS_WEIGHT * pair_loss   
    else:
        loss = additional_loss + PAIR_LOSS_WEIGHT * pair_loss   
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return pair_loss, adv_loss, loss
    
def handle_single_data(imgs, pids, clothes_ids, pid2clothes, ):
    B = imgs.shape[0]
    N_replicas = 0
    
    pos_mask = pid2clothes[pids.cpu()]
    imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()

    return imgs, pids, clothes_ids, pos_mask, B, N_replicas 




########### Baselines ###########
def train_cal(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes, **kwargs):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids.cpu()]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # save_image(rearrange(imgs, "B C N H W -> (B N) C H W" ), "t1.png")
        # save_image(normalize(imgs), "t1.png"); quit()

        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        pred_clothes = clothes_classifier(features.detach())
        _, preds = torch.max(outputs.data, 1)

        # Update the clothes discriminator
        clothes_loss = criterion_clothes(pred_clothes, clothes_ids)
        if epoch >= config.TRAIN.START_EPOCH_CC:
            optimizer_cc.zero_grad()
            if config.TRAIN.AMP:
                with amp.scale_loss(clothes_loss, optimizer_cc) as scaled_loss:
                    scaled_loss.backward()
            else:
                clothes_loss.backward()
            optimizer_cc.step()

        # Update the backbone
        new_pred_clothes = clothes_classifier(features)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)
        adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)
        if epoch >= config.TRAIN.START_EPOCH_ADV:
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(clothes_preds == clothes_ids.data).float()/clothes_ids.size(0), clothes_ids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.4f} '
                  'PairLoss:{pair_loss.avg:.4f} '
                  'CloLoss:{clo_loss.avg:.4f} '
                  'AdvLoss:{adv_loss.avg:.4f} '
                  'Acc:{acc.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, 
                   cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                   clo_loss=batch_clo_loss, adv_loss=batch_adv_loss, 
                   acc=corrects, clo_acc=clothes_corrects))

# soft triplet (baseline) on ONLY CAL 
def train_cal_pair27_ind_2feat(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes, criteria_feat_mse, criteria_logit_KL, 
    extra_loss=None, single_data=False, extra_classifier=None, **kwargs):
    logger = logging.getLogger('reid.train')
    
    metric_class = Stats()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()

    end = time.time()

    HANDLE_DATA = handle_replica_data
    if single_data:
        HANDLE_DATA = handle_single_data
    features_id = None
    outputs_id = None
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample

        imgs, pids, clothes_ids, pos_mask, B, N_replicas  = HANDLE_DATA(imgs, pids, clothes_ids, pid2clothes, )
        # print(imgs.shape, pids, clothes_ids.shape, pos_mask.shape, B, N_replicas)
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features_clothes = model(imgs)
        outputs = classifier(features_clothes)
        pred_clothes = clothes_classifier(features_clothes.detach())
        _, preds = torch.max(outputs.data, 1)

        # Update the clothes discriminator
        clothes_loss = clothes_discriminator_update(criterion_clothes, pred_clothes, clothes_ids, epoch, config.TRAIN.START_EPOCH_CC, optimizer_cc,)

        # Update the backbone
        new_pred_clothes = clothes_classifier(features_clothes)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        additional_loss = extra_loss(features_clothes=features_clothes, outputs=outputs, features_id=features_id, outputs_id=outputs_id, pids=pids, extra_classifier=extra_classifier)
        additional_loss += cla_loss
        
        kwargs = {}
        
        pair_loss, adv_loss, loss = compute_loss(criterion_pair, criterion_adv, features_clothes, pids, new_pred_clothes, clothes_ids, pos_mask,
            epoch, config.TRAIN.START_EPOCH_ADV, config.LOSS.PAIR_LOSS_WEIGHT, additional_loss, optimizer, **kwargs)

        if config.LOSS.ADDITIONAL_LOSS and "center" in config.LOSS.ADDITIONAL_LOSS:
            extra_loss.module.update_params()


        metric_class.update(preds, pids, clothes_preds, clothes_ids, cla_loss, pair_loss, clothes_loss, adv_loss)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    print_logs(logger, epoch, batch_time, data_time, metric_class.batch_cla_loss, metric_class.batch_pair_loss,
        metric_class.batch_clo_loss, metric_class.batch_adv_loss, metric_class.corrects, metric_class.clothes_corrects)

# soft triplet (baseline)
def train_cal_pair3_ind_2feat(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes, criteria_feat_mse, criteria_logit_KL, 
    extra_loss=None, single_data=False, extra_classifier=None):
    logger = logging.getLogger('reid.train')
    
    metric_class = Stats()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()

    end = time.time()

    HANDLE_DATA = handle_replica_data
    if single_data:
        HANDLE_DATA = handle_single_data

    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample

        imgs, pids, clothes_ids, pos_mask, B, N_replicas  = HANDLE_DATA(imgs, pids, clothes_ids, pid2clothes, )
        # print(imgs.shape, pids, clothes_ids.shape, pos_mask.shape, B, N_replicas)
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features_clothes, features_id, outputs_id = model(imgs)
        outputs = classifier(features_clothes)
        pred_clothes = clothes_classifier(features_clothes.detach())
        _, preds = torch.max(outputs.data, 1)

        # Update the clothes discriminator
        clothes_loss = clothes_discriminator_update(criterion_clothes, pred_clothes, clothes_ids, epoch, config.TRAIN.START_EPOCH_CC, optimizer_cc,)

        # Update the backbone
        new_pred_clothes = clothes_classifier(features_clothes)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        cla_loss += criterion_cla(outputs_id, pids)
        
        additional_loss = extra_loss(features_clothes=features_clothes, outputs=outputs, features_id=features_id, outputs_id=outputs_id, pids=pids, extra_classifier=extra_classifier)
        additional_loss += cla_loss
        
        kwargs = {}
        pair_loss, adv_loss, loss = compute_loss(criterion_pair, criterion_adv, features_id, pids, new_pred_clothes, clothes_ids, pos_mask,
            epoch, config.TRAIN.START_EPOCH_ADV, config.LOSS.PAIR_LOSS_WEIGHT, additional_loss, optimizer, **kwargs)

        if config.LOSS.ADDITIONAL_LOSS and "center" in config.LOSS.ADDITIONAL_LOSS:
            extra_loss.module.update_params()


        metric_class.update(preds, pids, clothes_preds, clothes_ids, cla_loss, pair_loss, clothes_loss, adv_loss)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    print_logs(logger, epoch, batch_time, data_time, metric_class.batch_cla_loss, metric_class.batch_pair_loss,
        metric_class.batch_clo_loss, metric_class.batch_adv_loss, metric_class.corrects, metric_class.clothes_corrects)

# train_cal_pair3_ind_2feat 
def train_cal_pair4_ind_2feat(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes, criteria_feat_mse, criteria_logit_KL, 
    extra_loss=None, single_data=False, extra_classifier=None):
    logger = logging.getLogger('reid.train')
    
    metric_class = Stats2(keys=["id", "tri"])
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    
    end = time.time()

    HANDLE_DATA = handle_replica_data
    if single_data:
        HANDLE_DATA = handle_single_data

    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample

        imgs, pids, clothes_ids, pos_mask, B, N_replicas  = HANDLE_DATA(imgs, pids, clothes_ids, pid2clothes, )
        # print(imgs.shape, pids, clothes_ids.shape, pos_mask.shape, B, N_replicas)
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features_id, outputs = model(imgs)
        _, preds = torch.max(outputs.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        
        kwargs = {}
        pair_loss, adv_loss, loss = compute_loss(criterion_pair, None, features_id, pids, None, clothes_ids, pos_mask,
            epoch, config.TRAIN.START_EPOCH_ADV, config.LOSS.PAIR_LOSS_WEIGHT, cla_loss, optimizer, **kwargs)

        metric_class.update(batch_size=pids.size(0), loss_dict=dict(id = cla_loss, tri=pair_loss))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    logger.info('Epoch{0} \t'  'Time:{batch_time.sum:.1f}s \t' 'Data:{data_time.sum:.1f}s \t' 'Tri:{TRI.avg:.4f} \t' 'ID Loss:{id_loss.avg:.4f}'.format(  epoch+1, batch_time=batch_time, data_time=data_time,  TRI=metric_class.metrics["tri"],  id_loss=metric_class.metrics["id"]))
    model.module.student_mode = False
    

########### Gender ###########
# train_cal_pair3_ind_2feat (overall gender)
def train_cal_pair14_ind_2feat(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes, criteria_feat_mse, criteria_logit_KL, extra_classifier=None, single_data=False, 
    gender_id=None, gender_clothes=None, gender_overall=None, extra_loss=None, use_gender=None):
    logger = logging.getLogger('reid.train')
    
    metric_class = Stats()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()

    end = time.time()

    HANDLE_DATA = handle_replica_data
    if single_data:
        HANDLE_DATA = handle_single_data

    for batch_idx, (imgs, pids, camids, clothes_ids, gender, _) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample

        imgs, pids, clothes_ids, pos_mask, B, N_replicas  = HANDLE_DATA(imgs, pids, clothes_ids, pid2clothes, )
        gender = expand_input(gender, N_replicas + 1)
        # print(imgs.shape, pids, clothes_ids.shape, pos_mask.shape, B, N_replicas)
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Forward
        features_clothes, features_id, outputs_id = model(imgs)
        outputs = classifier(features_clothes)
        pred_clothes = clothes_classifier(features_clothes.detach())
        _, preds = torch.max(outputs.data, 1)
        if gender_id:
            gender_pred = features_id
        elif gender_clothes:
            gender_pred = features_clothes
        else:
            gender_pred = torch.cat([features_clothes, features_id],-1)

        gender_pred = extra_classifier(gender_pred)
        
        # Update the clothes discriminator
        clothes_loss = clothes_discriminator_update(criterion_clothes, pred_clothes, clothes_ids, epoch, config.TRAIN.START_EPOCH_CC, optimizer_cc,)

        # Update the backbone
        new_pred_clothes = clothes_classifier(features_clothes)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        cla_loss += criterion_cla(outputs_id, pids)
        cla_loss += criterion_cla(gender_pred, gender)

        additional_loss = extra_loss(features_clothes=features_clothes, outputs=outputs, features_id=features_id, outputs_id=outputs_id, pids=pids, extra_classifier=extra_classifier)
        additional_loss += cla_loss
        kwargs = {}
        pair_loss, adv_loss, loss = compute_loss(criterion_pair, criterion_adv, features_id, pids, new_pred_clothes, clothes_ids, pos_mask,
            epoch, config.TRAIN.START_EPOCH_ADV, config.LOSS.PAIR_LOSS_WEIGHT, additional_loss, optimizer, **kwargs)

        if config.LOSS.ADDITIONAL_LOSS and "center" in config.LOSS.ADDITIONAL_LOSS:
            extra_loss.module.update_params()

        metric_class.update(preds, pids, clothes_preds, clothes_ids, cla_loss, pair_loss, clothes_loss, adv_loss)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    print_logs(logger, epoch, batch_time, data_time, metric_class.batch_cla_loss, metric_class.batch_pair_loss,
        metric_class.batch_clo_loss, metric_class.batch_adv_loss, metric_class.corrects, metric_class.clothes_corrects)


########### UBD ###########
# train_cal_pair9_ind_2feat (distillation) ONLY CAL 
def train_cal_pair26_ind_2feat(config, epoch, model, classifier, clothes_classifier, criterion_cla, optimizer,
    trainloader_teacher=None, teacher_model=None, teacher_classifier=None, teacher_id_classifier=None, criteria_DL_mse=None, criteria_DL_KL=None, 
    teacher_clothes_classifier=None, **kwargs):
    logger = logging.getLogger('reid.train')
    
    metric_class = Stats2(keys=["mse", "KL"])
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()
    end = time.time()

    model.module.student_mode = True
    
    for batch_idx, (img_HR, pids, camids, _, img_LR) in enumerate(trainloader_teacher):
        B = img_HR.shape[0]
        img_HR, pids, img_LR = img_HR.cuda(), pids.cuda(), img_LR.cuda()
        HR_LR = torch.cat([img_HR, img_LR])
        # save_image(HR_LR, "hr_lr.png")
        # Get all positive clothes classes (belonging to the same identity) for each sample
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        with torch.autograd.set_detect_anomaly(True):
            features_clothes_Teacher_HR = teacher_model(img_HR)
            outputs_id_Teacher_HR = teacher_classifier(features_clothes_Teacher_HR)
            # pred_clothes_Teacher_HR = teacher_clothes_classifier(features_clothes_Teacher_HR)
            features_clothes_Student = model(HR_LR)
            outputs_student = classifier(features_clothes_Student)
            pred_clothes = clothes_classifier(features_clothes_Student.detach())

        features_clothes_Student_HR, features_clothes_Student_LR  = features_clothes_Student[:B], features_clothes_Student[B:]
        outputs_student_HR, outputs_student_LR = outputs_student[:B] , outputs_student[B:]
        pred_clothes_HR, pred_clothes_LR = pred_clothes[:B] , pred_clothes[B:]

        # Compute MSE Loss
        mse1 = criteria_DL_mse(features_clothes_Teacher_HR, features_clothes_Student_HR)
        mse2 = criteria_DL_mse(features_clothes_Teacher_HR, features_clothes_Student_LR)
        MSE = mse1 + mse2
        
        # Compute KL Loss
        kl4 = criteria_DL_KL(outputs_student_HR, outputs_student_LR)
        kl5 = criteria_DL_KL(pred_clothes_HR, pred_clothes_LR)
        KL = kl4 + kl5

        # Compute Total loss
        loss = MSE + KL
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_class.update(batch_size=pids.size(0), loss_dict=dict(mse = MSE, KL=KL))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    logger.info('Epoch{0} \t'  'Time:{batch_time.sum:.1f}s \t' 'Data:{data_time.sum:.1f}s \t'
                  'MSE:{MSE.avg:.4f} \t' 'KL-Loss:{kl_loss.avg:.4f}'.format( 
                epoch+1, batch_time=batch_time, data_time=data_time,  MSE=metric_class.metrics["mse"],  kl_loss=metric_class.metrics["KL"]))
    model.module.student_mode = False

# Hr - LR (Teacher - Student)
def train_cal_pair9_ind_2feat(config, epoch, model, classifier, clothes_classifier, criterion_cla, optimizer,
    trainloader_teacher=None, teacher_model=None, teacher_classifier=None, teacher_id_classifier=None, criteria_DL_mse=None, criteria_DL_KL=None, **kwargs):
    logger = logging.getLogger('reid.train')
    
    metric_class = Stats2(keys=["mse", "KL"])
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()
    end = time.time()

    model.module.student_mode = True
    
    for batch_idx, (img_HR, pids, camids, _, img_LR) in enumerate(trainloader_teacher):
        B = img_HR.shape[0]
        img_HR, pids, img_LR = img_HR.cuda(), pids.cuda(), img_LR.cuda()
        HR_LR = torch.cat([img_HR, img_LR])
        # save_image(HR_LR, "hr_lr.png")
        # Get all positive clothes classes (belonging to the same identity) for each sample
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        
        with torch.autograd.set_detect_anomaly(True):
            features_clothes_Teacher_HR, features_id_Teacher_HR = teacher_model(img_HR)
            outputs_id_Teacher_HR = teacher_id_classifier(features_id_Teacher_HR)
            features_clothes_Student, features_id_Student, outputs_id_Student = model(HR_LR)
            outputs_student = classifier(features_clothes_Student)
            pred_clothes = clothes_classifier(features_clothes_Student.detach())

        features_clothes_Student_HR, features_clothes_Student_LR  = features_clothes_Student[:B], features_clothes_Student[B:]
        features_id_Student_HR, features_id_Student_LR = features_id_Student[:B], features_id_Student[B:]
        outputs_id_Student_HR, outputs_id_Student_LR = outputs_id_Student[:B], outputs_id_Student[B:]
        outputs_student_HR, outputs_student_LR = outputs_student[:B] , outputs_student[B:]
        pred_clothes_HR, pred_clothes_LR = pred_clothes[:B] , pred_clothes[B:]

        outputs_id_Teacher_Student_HR_LR = teacher_id_classifier(features_id_Student)
        outputs_id_Teacher_Student_HR, outputs_id_Teacher_Student_LR = outputs_id_Teacher_Student_HR_LR[:B], outputs_id_Teacher_Student_HR_LR[B:]

        # Compute MSE Loss
        mse1 = criteria_DL_mse(features_clothes_Teacher_HR, features_clothes_Student_HR)
        mse2 = criteria_DL_mse(features_clothes_Teacher_HR, features_clothes_Student_LR)
        mse3 = criteria_DL_mse(features_id_Teacher_HR, features_id_Student_HR)
        mse4 = criteria_DL_mse(features_id_Teacher_HR, features_id_Student_LR)
        MSE = mse1 + mse2 + mse3 + mse4
        
        # Compute KL Loss
        kl3 = criteria_DL_KL(outputs_id_Student_HR, outputs_id_Student_LR)
        kl4 = criteria_DL_KL(outputs_student_HR, outputs_student_LR)
        kl5 = criteria_DL_KL(pred_clothes_HR, pred_clothes_LR)
        KL = kl3 + kl4 + kl5

        # Compute Total loss
        loss = MSE + KL
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_class.update(batch_size=pids.size(0), loss_dict=dict(mse = MSE, KL=KL))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    logger.info('Epoch{0} \t'  'Time:{batch_time.sum:.1f}s \t' 'Data:{data_time.sum:.1f}s \t'
                  'MSE:{MSE.avg:.4f} \t' 'KLLoss:{kl_loss.avg:.4f}'.format( 
                epoch+1, batch_time=batch_time, data_time=data_time,  MSE=metric_class.metrics["mse"],  kl_loss=metric_class.metrics["KL"]))
    model.module.student_mode = False

########### POSE  ###########
# soft triplet (baseline) + Pose Branch
def train_cal_pair16_ind_2feat(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes, criteria_feat_mse, criteria_logit_KL, extra_classifier=None, single_data=False, 
    extra_loss=None, use_gender=None):
    logger = logging.getLogger('reid.train')
    
    metric_class = Stats()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()

    end = time.time()

    HANDLE_DATA = handle_replica_data
    if single_data:
        HANDLE_DATA = handle_single_data

    for batch_idx, (imgs, pids, camids, clothes_ids, gender, pose) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample

        imgs, pids, clothes_ids, pos_mask, B, N_replicas  = HANDLE_DATA(imgs, pids, clothes_ids, pid2clothes, )
        pose = expand_input(pose, N_replicas + 1)
        # print(imgs.shape, pids, clothes_ids.shape, pos_mask.shape, B, N_replicas)
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Forward        
        features_clothes, features_id, features_pose, outputs_id, outputs_pose = model(imgs)
        outputs = classifier(features_clothes)
        pred_clothes = clothes_classifier(features_clothes.detach())
        _, preds = torch.max(outputs.data, 1)
        
        # Update the clothes discriminator
        clothes_loss = clothes_discriminator_update(criterion_clothes, pred_clothes, clothes_ids, epoch, config.TRAIN.START_EPOCH_CC, optimizer_cc,)

        # Update the backbone
        new_pred_clothes = clothes_classifier(features_clothes)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        cla_loss += criterion_cla(outputs_id, pids)
        cla_loss += criterion_cla(outputs_pose, pose)

        additional_loss = extra_loss(features_clothes=features_clothes, outputs=outputs, features_id=features_id, outputs_id=outputs_id, pids=pids, extra_classifier=extra_classifier, features_pose=features_pose)
        additional_loss += cla_loss
        
        kwargs = {}
        pair_loss, adv_loss, loss = compute_loss(criterion_pair, criterion_adv, features_id, pids, new_pred_clothes, clothes_ids, pos_mask,
            epoch, config.TRAIN.START_EPOCH_ADV, config.LOSS.PAIR_LOSS_WEIGHT, additional_loss, optimizer, **kwargs)

        if config.LOSS.ADDITIONAL_LOSS and "center" in config.LOSS.ADDITIONAL_LOSS:
            extra_loss.module.update_params()

        metric_class.update(preds, pids, clothes_preds, clothes_ids, cla_loss, pair_loss, clothes_loss, adv_loss)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    print_logs(logger, epoch, batch_time, data_time, metric_class.batch_cla_loss, metric_class.batch_pair_loss,
        metric_class.batch_clo_loss, metric_class.batch_adv_loss, metric_class.corrects, metric_class.clothes_corrects)

########### Gender + POSE ###########
# soft triplet (baseline) + Pose Branch + Gender
def train_cal_pair23_ind_2feat(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes, criteria_feat_mse, criteria_logit_KL, extra_classifier=None, single_data=False,
    gender_id=None, gender_clothes=None, gender_overall=None, extra_loss=None, use_gender=None):
    
    logger = logging.getLogger('reid.train')
    
    metric_class = Stats()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()

    end = time.time()

    HANDLE_DATA = handle_replica_data
    if single_data:
        HANDLE_DATA = handle_single_data

    for batch_idx, (imgs, pids, camids, clothes_ids, gender, pose) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample

        imgs, pids, clothes_ids, pos_mask, B, N_replicas  = HANDLE_DATA(imgs, pids, clothes_ids, pid2clothes, )
        pose = expand_input(pose, N_replicas + 1)
        gender = expand_input(gender, N_replicas + 1) 
        # print(imgs.shape, pids, clothes_ids.shape, pos_mask.shape, B, N_replicas)
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Forward        
        features_clothes, features_id, features_pose, outputs_id, outputs_pose = model(imgs)
        if gender_id:
            gender_pred = features_id
        elif gender_clothes:
            gender_pred = features_clothes
        else:
            gender_pred = torch.cat([features_clothes, features_id],-1)
        gender_pred = extra_classifier(gender_pred)

        outputs = classifier(features_clothes)
        pred_clothes = clothes_classifier(features_clothes.detach())
        _, preds = torch.max(outputs.data, 1)
        
        # Update the clothes discriminator
        clothes_loss = clothes_discriminator_update(criterion_clothes, pred_clothes, clothes_ids, epoch, config.TRAIN.START_EPOCH_CC, optimizer_cc,)

        # Update the backbone
        new_pred_clothes = clothes_classifier(features_clothes)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        cla_loss += criterion_cla(outputs_id, pids)
        cla_loss += criterion_cla(outputs_pose, pose)
        cla_loss += criterion_cla(gender_pred, gender)

        additional_loss = extra_loss(features_clothes=features_clothes, outputs=outputs, features_id=features_id, outputs_id=outputs_id, pids=pids, extra_classifier=extra_classifier, features_pose=features_pose)
        additional_loss += cla_loss
        
        kwargs = {}
        pair_loss, adv_loss, loss = compute_loss(criterion_pair, criterion_adv, features_id, pids, new_pred_clothes, clothes_ids, pos_mask,
            epoch, config.TRAIN.START_EPOCH_ADV, config.LOSS.PAIR_LOSS_WEIGHT, additional_loss, optimizer, **kwargs)

        metric_class.update(preds, pids, clothes_preds, clothes_ids, cla_loss, pair_loss, clothes_loss, adv_loss)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    print_logs(logger, epoch, batch_time, data_time, metric_class.batch_cla_loss, metric_class.batch_pair_loss,
        metric_class.batch_clo_loss, metric_class.batch_adv_loss, metric_class.corrects, metric_class.clothes_corrects)

