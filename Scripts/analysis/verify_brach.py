import sys
import os 

currentdir = os.getcwd() 
parentdir = os.path.dirname(currentdir)
grandfatherdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
sys.path.append(grandfatherdir)


from tools.utils import load_pickle
import torch
import time 
import sys 
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import torch.nn.functional as F

def compute_acc(index, good_index, junk_index):
    cmc = np.zeros(len(index)) 

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    if rows_good[0] !=  0:
        return False, index[0]
    return True, None  
    
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

def make_folder(name):
    try: 
        os.mkdir(name) 
    except OSError as error: 
        _ = 0 
    

make_folder("Samples/")
os.system("rm -rf Samples/*")
    
dataset_name=  sys.argv[1]
pickle_path1 = sys.argv[2]
pickle_path2 = sys.argv[3]
feature_dump1 = load_pickle(pickle_path1)
feature_dump2 = load_pickle(pickle_path2)



def load_q_g(feature_dump):
    qf = torch.tensor(feature_dump["qf"])
    gf = torch.tensor(feature_dump["gf"])

    q_pids=feature_dump["q_pids"].numpy()
    q_camids=feature_dump["q_camids"].numpy()
    q_clothes_ids=feature_dump["q_clothes_ids"].numpy()

    g_pids=feature_dump["g_pids"].numpy()
    g_camids=feature_dump["g_camids"].numpy()
    g_clothes_ids=feature_dump["g_clothes_ids"].numpy()
    
    return qf, gf, q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids

def load_qs_qd_g(feature_dump):
    qsf=feature_dump["qsf"]
    qs_pids=feature_dump["qs_pids"]
    qs_camids=feature_dump["qs_camids"]
    qs_clothes_ids=feature_dump["qs_clothes_ids"]

    qdf=feature_dump["qdf"]
    qd_pids=feature_dump["qd_pids"]
    qd_camids=feature_dump["qd_camids"]
    qd_clothes_ids=feature_dump["qd_clothes_ids"]

    # qs_image_paths=qs_image_paths,
    # qd_image_paths = qd_image_paths

    gf = torch.tensor(feature_dump["gf"])
    g_pids=feature_dump["g_pids"]
    g_camids=feature_dump["g_camids"]
    g_clothes_ids=feature_dump["g_clothes_ids"]
    
    return qsf, qs_pids, qs_camids, qs_clothes_ids, \
        qdf, qd_pids, qd_camids, qd_clothes_ids, \
        gf, g_pids, g_camids, g_clothes_ids

def misfit_ltcc(dismat, g_pids, q_pids, g_camids, q_camids, g_clothes_ids, q_clothes_ids):
    num_q, num_g = dismat.shape
    index = np.argsort(dismat, axis=1) # from small to large
    CMC = np.zeros(len(g_pids))
    mode = "CC"
    count = 0 
    r1 = 0
    misfits = []
    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        cloth_index = np.argwhere(g_clothes_ids==q_clothes_ids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if mode == 'CC':
            good_index = np.setdiff1d(good_index, cloth_index, assume_unique=True)
            # remove gallery samples that have the same (pid, camid) or (pid, clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.intersect1d(query_index, cloth_index)
            junk_index = np.union1d(junk_index1, junk_index2)
        else:
            good_index = np.intersect1d(good_index, cloth_index)
            # remove gallery samples that have the same (pid, camid) or 
            # (the same pid and different clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.setdiff1d(query_index, cloth_index)
            junk_index = np.union1d(junk_index1, junk_index2)

        if good_index.size == 0:
            continue
        count += 1
        correct , mis_fit = compute_acc(index[i], good_index, junk_index)
        if correct:
            r1 += 1
        else:
            misfits.append([i, mis_fit])
    return r1, misfits, count

def simple_indentifier(x):
    return x.split("/")[-1][:-4]

def prcc_indentifier(x):
    id, image_name = x.split("/")[-2:]
    identifier = id + "_" + image_name[:-4]
    return identifier 


def ltcc_dist(q, g ):
    q = F.normalize(q, p=2, dim=1)
    g = F.normalize(g, p=2, dim=1)
    dismat, _, _ = compute_distance(q, g, print)
    return dismat

# LTCC
if dataset_name == "LTCC":
    indentifier_fn = simple_indentifier
    
    qf1, gf1, q_pids1, q_camids1, q_clothes_ids1, g_pids1, g_camids1, g_clothes_ids1 = load_q_g(feature_dump1)
    qf2, gf2, q_pids2, q_camids2, q_clothes_ids2, g_pids2, g_camids2, g_clothes_ids2 = load_q_g(feature_dump2)

    dismat1 = ltcc_dist(qf1, gf1)
    dismat2 = ltcc_dist(qf2, gf2)

    r1, _, count = misfit_ltcc(dismat1, g_pids1, q_pids1, g_camids1, q_camids1, g_clothes_ids1, q_clothes_ids1)
    print(r1, r1 / count)
    r2, _, count = misfit_ltcc(dismat2, g_pids2, q_pids2, g_camids2, q_camids2, g_clothes_ids2, q_clothes_ids2)
    print(r2, r2 / count)


    qf = torch.cat([qf1, qf2],-1)
    gf = torch.cat([gf1, gf2],-1)
    dismat3 = ltcc_dist(qf, gf)
    R, _, count = misfit_ltcc(dismat3, g_pids1, q_pids1, g_camids1, q_camids1, g_clothes_ids1, q_clothes_ids1)
    print(R, R / count)
    

    
# # PRCC
# elif dataset_name == "PRCC":
#     dataset = PRCC(root=sys.argv[3])
#     query_set = dataset.query_diff
#     indentifier_fn = prcc_indentifier
    
#     qsf, qs_pids, qs_camids, qs_clothes_ids, \
#         qdf, qd_pids, qd_camids, qd_clothes_ids, \
#         gf, g_pids, g_camids, g_clothes_ids = load_qs_qd_g(feature_dump)

#     m, n, k = qsf.shape[0], qdf.shape[0], gf.shape[0]
#     distmat_same = torch.zeros((m, k))
#     distmat_diff = torch.zeros((n, k))
    
#     qsf, qdf, gf = torch.tensor(qsf).cuda(), torch.tensor(qdf).cuda(), torch.tensor(gf).cuda()
#     # Cosine similarity
#     for i in range(m):
#         distmat_same[i] = (- torch.mm(qsf[i:i+1], gf.t())).cpu()
#     for i in range(n):
#         distmat_diff[i] = (- torch.mm(qdf[i:i+1], gf.t())).cpu()
#     distmat_same = distmat_same.numpy()
#     distmat_diff = distmat_diff.numpy()
    
#     r1, misfits, count = misfit_ltcc(distmat_diff, g_pids, qd_pids, g_camids, qd_camids, g_clothes_ids, qd_clothes_ids)




# python Scripts/analysis/verify_brach.py LTCC BM_28_1_BOT BM_28_1_CAL
# python Scripts/analysis.py Sim_Debug LTCC /data/priyank/synthetic/LTCC/    
# python Scripts/analysis.py prcc_debug PRCC /data/priyank/synthetic/PRCC/
# python Scripts/analysis.py prcc_debug PRCC /data/priyank/synthetic/PRCC/

# prcc/best_model.pth.tar