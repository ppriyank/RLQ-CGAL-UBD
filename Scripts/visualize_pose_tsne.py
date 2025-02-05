import sys
import os 

currentdir = os.getcwd() 
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)

from tools.utils import load_pickle
import torch
import time 
import sys 
import numpy as np

from data import LTCC, PRCC

import pandas as pd 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def make_folder(name):
    try: 
        os.mkdir(name) 
    except OSError as error: 
        _ = 0 
    
def load_q_g(feature_dump):
    qf = torch.tensor(feature_dump["qf"])
    gf = torch.tensor(feature_dump["gf"])

    q_pids=feature_dump["q_pids"]
    q_camids=feature_dump["q_camids"]
    q_clothes_ids=feature_dump["q_clothes_ids"]

    g_pids=feature_dump["g_pids"]
    g_camids=feature_dump["g_camids"]
    g_clothes_ids=feature_dump["g_clothes_ids"]
    
    q_path = feature_dump["q_image_paths"]
    g_path = feature_dump["g_image_paths"]

    return qf, gf, q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, q_path, g_path

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

def compute_distance_l2(qf, gf, output):
    diff = qf.unsqueeze(1) - gf.unsqueeze(0)
    diff = (diff ** 2).sum(-1)
    return diff, None, None


if __name__ == "__main__":
    make_folder("Samples/")
    os.system("rm -rf Samples/*")

    # ['Scripts/visualize_pose_tsne.py', 'LTCC', '/data/priyank/synthetic/LTCC/', 'BM_28_4_LTCC', 'RLQ_25_B=32_1']

    dataset_name=  sys.argv[1]
    dataset_root = sys.argv[2]
    pickle_path1 = sys.argv[3]
    pickle_path2 = sys.argv[4]
    feature_dump1 = load_pickle(pickle_path1)
    feature_dump2 = load_pickle(pickle_path2)

    
    # LTCC
    if dataset_name == "LTCC":
        pose_df = pd.read_csv("Scripts/Helper/LTCC_TEST_Pose_Cluster.csv")
        POSE = 'R_LA_25'
        pose_df = pose_df[['Image', POSE]]
        pose_df = pose_df[pose_df[POSE] !=0 ]
        

        dataset = LTCC(root=dataset_root)
        query_set = dataset.query
        gallery_set = dataset.gallery
        
        qf1, gf1, q_pids1, q_camids1, q_clothes_ids1, g_pids1, g_camids1, g_clothes_ids1, q_paths1, g_paths1 = load_q_g(feature_dump1)
        qf2, gf2, q_pids2, q_camids2, q_clothes_ids2, g_pids2, g_camids2, g_clothes_ids2, q_paths2, g_paths2 = load_q_g(feature_dump2)

        feats = []
        Y_g, Y_q = [], []
        X1_g, X2_g= [], []
        X1_q, X2_q= [], []
        Z_g, Z_q = [], []

        for i,g_path in enumerate(g_paths1):
            feat1 = gf1[i]
            feat2 = gf2[i]
            
            assert g_path == g_paths2[i]
            pose = pose_df[pose_df.Image == g_path[:-4]]
            if len(pose) == 0:continue 

            Z_g.append( g_pids1[i] )
            Y_g.append(pose[POSE].item())
            X1_g.append(feat1)
            X2_g.append(feat2)
        
        for i,q_path in enumerate(q_paths1):
            feat1 = qf1[i]
            feat2 = qf2[i]
            
            assert q_path == q_paths2[i]
            pose = pose_df[pose_df.Image == q_path[:-4]]
            if len(pose) == 0:continue 

            Z_q.append( q_pids1[i] )
            Y_q.append(pose[POSE].item())
            X1_q.append(feat1)
            X2_q.append(feat2)
        
        X1_g = torch.stack(X1_g)
        X1_q = torch.stack(X1_q)

        X2_g = torch.stack(X2_g)
        X2_q = torch.stack(X2_q)
        Z_q = torch.stack(Z_q)
        Z_g = torch.stack(Z_g)
        
        Y_g = torch.tensor(Y_g)
        Y_q = torch.tensor(Y_q)

        pose_sim_map = Y_q.unsqueeze(1) == Y_g.unsqueeze(0)
        label_sim = Z_q.unsqueeze(1) == Z_g.unsqueeze(0)
        # X1_tsne = TSNE(n_components=2, random_state=0).fit_transform(X1)
        # X2_tsne = TSNE(n_components=2, random_state=0).fit_transform(X2)

        
        # # Plot the t-SNE result
        # plt.figure(figsize=(8, 6))
        # plt.scatter(X1_tsne[:, 0], X1_tsne[:, 1], c=Y, cmap='viridis')
        # plt.title('t-SNE plot')
        # plt.xlabel('t-SNE feature 1')
        # plt.ylabel('t-SNE feature 2')
        # plt.savefig('temp1.png')
        # plt.clf()

        # plt.figure(figsize=(8, 6))
        # plt.scatter(X2_tsne[:, 0], X2_tsne[:, 1], c=Y, cmap='viridis')
        # plt.title('t-SNE plot')
        # plt.xlabel('t-SNE feature 1')
        # plt.ylabel('t-SNE feature 2')
        # plt.savefig('temp2.png')

        dismat_1, _, _ = compute_distance_l2(X1_q, X1_g, print)
        dismat_2, _, _ = compute_distance_l2(X2_q, X2_g, print)

        # dismat_1.shape, dismat_2.shape, pose_sim_map.shape, label_sim.shape
        
        similar_pose_similar_id_1 = dismat_1[(pose_sim_map & label_sim).numpy()]
        similar_pose_differ_id_1 = dismat_1[(pose_sim_map & ~label_sim).numpy()]
        differ_pose_similar_id_1 = dismat_1[(~pose_sim_map & label_sim).numpy()]
        differ_pose_differ_id_1 = dismat_1[(~pose_sim_map & ~label_sim).numpy()]

        similar_pose_similar_id_2 = dismat_2[(pose_sim_map & label_sim).numpy()]
        similar_pose_differ_id_2 = dismat_2[(pose_sim_map & ~label_sim).numpy()]
        differ_pose_similar_id_2 = dismat_2[(~pose_sim_map & label_sim).numpy()]
        differ_pose_differ_id_2 = dismat_2[(~pose_sim_map & ~label_sim).numpy()]
        

        print(similar_pose_similar_id_1.mean(), similar_pose_differ_id_1.mean(), differ_pose_similar_id_1.mean(), differ_pose_differ_id_1.mean())
        print(similar_pose_similar_id_2.mean(), similar_pose_differ_id_2.mean(), differ_pose_similar_id_2.mean(), differ_pose_differ_id_2.mean())

        
        
        

# python Scripts/visualize_pose_tsne.py LTCC /data/priyank/synthetic/LTCC/ "BM_28_4_LTCC" 'RLQ_25_B=32_1'    