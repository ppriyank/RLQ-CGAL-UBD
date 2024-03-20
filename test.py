import time
import datetime
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from tools.eval_metrics import evaluate, evaluate_with_clothes
from tools.utils import rearrange, repeat, normalize, rearrange_mlr, expand_input, reverse_arrange
from torchvision.utils import save_image
import pickle

def save_pickel(obj, name):
    with open(f'{name}.pkl', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickel(name):
    with open(f'{name}.pkl', 'rb') as handle:
        return pickle.load(handle)

def concat_all_gather(tensors, num_total_examples):
    '''
    Performs all_gather operation on the provided tensor list.
    '''
    outputs = []
    for tensor in tensors:
        tensor = tensor.cuda()
        tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0).cpu()
        # truncate the dummy elements added by DistributedInferenceSampler
        outputs.append(output[:num_total_examples])
    return outputs

@torch.no_grad()
def extract_img_feature(model, dataloader):
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids) in enumerate(dataloader):
        flip_imgs = torch.flip(imgs, [3])
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs)
        batch_features_flip = model(flip_imgs)
        batch_features += batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)

        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = torch.cat(features, 0)

    return features, pids, camids, clothes_ids

@torch.no_grad()
def extract_img_feature_path(model, dataloader):
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    image_names = []
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids, img_path) in enumerate(dataloader):
        image_names += img_path
        flip_imgs = torch.flip(imgs, [3])
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs)
        batch_features_flip = model(flip_imgs)
        batch_features += batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)

        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = torch.cat(features, 0)

    return features, pids, camids, clothes_ids, image_names

def compute_distance(qf, gf, output):
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
    distmat = distmat.numpy()
    
    time_elapsed = time.time() - since
    output('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return distmat, qf, gf

def compute_scores(qf, gf, time_elapsed, 
        q_pids, q_camids, q_clothes_ids,  g_pids, g_camids, g_clothes_ids, 
        output=None, dataset_name=None):
    
    output("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    output("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    output('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    distmat, qf, gf = compute_distance(qf, gf, output)

    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    since = time.time()
    output("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    output("Results ---------------------------------------------------")
    output('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    output("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    output('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if ("last" in dataset_name) or ("deepchange" in dataset_name) or ("vcclothes" in dataset_name): return cmc, mAP
    # if dataset_name in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']: 

    output("Computing CMC and mAP only for the same clothes setting")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
    output("Results ---------------------------------------------------")
    output('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    output("-----------------------------------------------------------")

    output("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
    output("Results ---------------------------------------------------")
    output('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    output("-----------------------------------------------------------")
    return cmc, mAP


def test(config, model, queryloader, galleryloader, dataset, dump=False):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features 
    if config.EVAL_MODE:
            qf, q_pids, q_camids, q_clothes_ids, q_image_paths = extract_img_feature_path(model, queryloader)
            gf, g_pids, g_camids, g_clothes_ids, g_image_paths = extract_img_feature_path(model, galleryloader)
    else:
        qf, q_pids, q_camids, q_clothes_ids = extract_img_feature(model, queryloader)
        gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(model, galleryloader)
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query))
    gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since

    cmc, mAP = compute_scores(qf, gf, time_elapsed, q_pids, q_camids, q_clothes_ids,  g_pids, g_camids, g_clothes_ids, 
        output=logger.info, dataset_name=config.DATA.DATASET )
    
    if dump:
        feature_dump = dict(
            qf=qf.cpu().numpy(), q_pids=q_pids, q_camids=q_camids, q_clothes_ids=q_clothes_ids, q_image_paths=q_image_paths,
            gf=gf.cpu().numpy(), g_pids=g_pids, g_camids=g_camids, g_clothes_ids=g_clothes_ids, g_image_paths=g_image_paths,
        )
        return cmc[0], mAP, feature_dump

    return cmc[0], mAP

def test_prcc(config, model, queryloader_same, queryloader_diff, galleryloader, dataset, dump=None):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    if config.EVAL_MODE:
        # Extract features for query set
        qsf, qs_pids, qs_camids, qs_clothes_ids, qs_image_paths = extract_img_feature_path(model, queryloader_same)
        qdf, qd_pids, qd_camids, qd_clothes_ids, qd_image_paths = extract_img_feature_path(model, queryloader_diff)        
        gf, g_pids, g_camids, g_clothes_ids, g_image_paths = extract_img_feature_path(model, galleryloader)
    else:
        # Extract features for query set
        qsf, qs_pids, qs_camids, qs_clothes_ids = extract_img_feature(model, queryloader_same)
        qdf, qd_pids, qd_camids, qd_clothes_ids = extract_img_feature(model, queryloader_diff)
        # Extract features for gallery set
        gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(model, galleryloader)
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    qsf, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather([qsf, qs_pids, qs_camids, qs_clothes_ids], len(dataset.query_same))
    qdf, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather([qdf, qd_pids, qd_camids, qd_clothes_ids], len(dataset.query_diff))
    # dump
    gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set (with same clothes), obtained {} matrix".format(qsf.shape))
    logger.info("Extracted features for query set (with different clothes), obtained {} matrix".format(qdf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = torch.zeros((m, k))
    distmat_diff = torch.zeros((n, k))
    qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat_same[i] = (- torch.mm(qsf[i:i+1], gf.t())).cpu()
    for i in range(n):
        distmat_diff[i] = (- torch.mm(qdf[i:i+1], gf.t())).cpu()
    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()
    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    logger.info("Computing CMC and mAP for the same clothes setting")
    cmc, mAP = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes changing")
    cmc, mAP = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    if dump:
        feature_dump = dict(
            qsf=qsf.cpu().numpy(), qs_pids=qs_pids, qs_camids=qs_camids, qs_clothes_ids=qs_clothes_ids, qs_image_paths=qs_image_paths,
            qdf=qdf.cpu().numpy(), qd_pids=qd_pids, qd_camids=qd_camids, qd_clothes_ids=qd_clothes_ids, qd_image_paths=qd_image_paths,
            gf=gf.cpu().numpy(), g_pids=g_pids, g_camids=g_camids, g_clothes_ids=g_clothes_ids, g_image_paths=g_image_paths,
        )
        return cmc[0], mAP, feature_dump
        
    return cmc[0], mAP


