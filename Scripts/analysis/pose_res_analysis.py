import torch 
import os
import sys
import pandas as pd 
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
grand_parentdir = os.path.dirname(currentdir)
sys.path.append(grand_parentdir)

from analysis import load_pickle, make_folder, LTCC, simple_indentifier, load_q_g, \
    compute_distance, misfit_ltcc, prcc_indentifier, PRCC, load_qs_qd_g, ToTensor, Image, save_image


import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset_name=  sys.argv[2]
    pickle_path1 = sys.argv[1]
    feature_dump1 = load_pickle(pickle_path1)

    # LTCC
    if dataset_name == "LTCC":
        dataset = LTCC(root=sys.argv[3])
        query_set = dataset.query
        gallery_set = dataset.gallery
        indentifier_fn = simple_indentifier
        qf1, gf1, q_pids1, q_camids1, q_clothes_ids1, g_pids1, g_camids1, g_clothes_ids1 = load_q_g(feature_dump1)
        dismat1, _, _ = compute_distance(qf1, gf1, print)
        q_pids1, q_camids1, q_clothes_ids1 = q_pids1.numpy(), q_camids1.numpy(), q_clothes_ids1.numpy()
        g_pids1, g_camids1, g_clothes_ids1 = g_pids1.numpy(), g_camids1.numpy(), g_clothes_ids1.numpy()
        r1, misfits1, corrects1, count = misfit_ltcc(dismat1, g_pids1, q_pids1, g_camids1, q_camids1, g_clothes_ids1, q_clothes_ids1)

        test_pose = pd.read_csv("Scripts/Helper/LTCC_TEST_Pose_Cluster.csv")
        query_pose = test_pose[test_pose.Category == "query"]
        gallery_pose = test_pose[test_pose.Category == "test"]
        
        
    # PRCC
    elif dataset_name == "PRCC":
        dataset = PRCC(root=sys.argv[3])
        query_set = dataset.query_diff
        indentifier_fn = prcc_indentifier
        
        qsf, qs_pids, qs_camids, qs_clothes_ids, \
            qdf, qd_pids, qd_camids, qd_clothes_ids, \
            gf, g_pids, g_camids, g_clothes_ids = load_qs_qd_g(feature_dump1)

        m, n, k = qsf.shape[0], qdf.shape[0], gf.shape[0]
        distmat_same = torch.zeros((m, k))
        distmat_diff = torch.zeros((n, k))
        
        qsf, qdf, gf = torch.tensor(qsf).cuda(), torch.tensor(qdf).cuda(), torch.tensor(gf).cuda()
        # Cosine similarity
        for i in range(m):
            distmat_same[i] = (- torch.mm(qsf[i:i+1], gf.t())).cpu()
        for i in range(n):
            distmat_diff[i] = (- torch.mm(qdf[i:i+1], gf.t())).cpu()
        distmat_same = distmat_same.numpy()
        distmat_diff = distmat_diff.numpy()
        
        r1, misfits1, corrects1, count = misfit_ltcc(distmat_diff, g_pids, qd_pids, g_camids, qd_camids, g_clothes_ids, qd_clothes_ids)
        test_gender = pd.read_csv("Scripts/Helper/PRCC_test_Gender.csv")
        query_gender = test_gender
        gallery_gender = test_gender

    print(r1, r1 / count)
    
    def process_dis(x,y, output_commnd=print):
        x = torch.tensor(np.array(x)).squeeze(1)
        y = torch.tensor(np.array(y)).squeeze(1)
        dismat, _, _ = compute_distance(x, y, output_commnd)
        return dismat
    
    Vec_list = load_pickle("Vec_list")
    T_Resized_Vec_list = Vec_list["T_Resized_Vec_list"]
    T_Resized_Vec_Cooridnates_list = Vec_list["T_Resized_Vec_Cooridnates_list"]
    T_Resized_Only_Angle_list = Vec_list["T_Resized_Only_Angle_list"]
    
    T_Norm_Vec_list = Vec_list["T_Norm_Vec_list"]
    T_Norm_Vec_Cooridnates_list = Vec_list["T_Norm_Vec_Cooridnates_list"]
    T_Norm_Only_Angle_list = Vec_list["T_Norm_Only_Angle_list"]
    image_names = Vec_list["image_names"]


    pose_query = []
    q_Resized_Vec_list = []
    q_Resized_Vec_Cooridnates_list = [] 
    q_Resized_Only_Angle_list = []
    
    q_Norm_Vec_list = []
    q_Norm_Vec_Cooridnates_list = [] 
    q_Norm_Only_Angle_list = []
    q_index = []

    q_Resized_Vec_dist, q_Resized_Vec_Cooridnates_dist, q_Resized_Only_Angle_dist = [], [], []
    q_Norm_Vec_dist, q_Norm_Vec_Cooridnates_dist, q_Norm_Only_Angle_dist = [], [], []

    correctness = []
    correct_dict = { x[0]:x[1] for x in corrects1}
    mistake_dict = { x[0]:x[1] for x in misfits1}
    
    pose_labels_dict = {}
    pose_labels = [x for x in gallery_pose.columns if x != "Category" and x!= "Image"]
    for key in pose_labels :
        pose_labels_dict[key] = [] 

    for i,query in enumerate(query_set):
        query_path = query[0]
        query_identifier = indentifier_fn(query_path)
        q_id = "query/" + query_identifier
        index_ = image_names == q_id
        
        if index_.any() and (i in correct_dict or i in mistake_dict):   
            if i in correct_dict:
                correctness.append(1)
                matched_g =  correct_dict[i]
            else:
                correctness.append(0)
                matched_g =  mistake_dict[i]
            q_id = "query/" + query_identifier
            q_pose = query_pose[query_pose.Image == q_id]        
            
            gallery_path = dataset.gallery[matched_g][0]
            gallery_identifier1 = indentifier_fn(gallery_path)
            g_id = "test/" + gallery_identifier1
            g_pose = gallery_pose[gallery_pose.Image == g_id]

            index_g = image_names == g_id
            g_Resized_Vec = T_Resized_Vec_list[index_g]
            g_Resized_Vec_Cooridnates = T_Resized_Vec_Cooridnates_list[index_g]
            g_Resized_Only_Angle = T_Resized_Only_Angle_list[index_g]
            
            g_Norm_Vec =  T_Norm_Vec_list[index_g]
            g_Norm_Vec_Cooridnates = T_Norm_Vec_Cooridnates_list[index_g]
            g_Norm_Only_Angle = T_Norm_Only_Angle_list[index_g]
            if len(g_Resized_Vec) == 0:
                q_index.append(False)
                correctness.pop(-1)
                continue 

            for key in pose_labels:
                if q_pose[key].item() == g_pose[key].item():
                    pose_labels_dict[key].append(1)
                else:
                    pose_labels_dict[key].append(0)
            q_index.append(True)

            q_Resized_Vec = T_Resized_Vec_list[index_]
            q_Resized_Vec_Cooridnates = T_Resized_Vec_Cooridnates_list[index_]
            q_Resized_Only_Angle = T_Resized_Only_Angle_list[index_]
            
            q_Norm_Vec = T_Norm_Vec_list[index_]
            q_Norm_Vec_Cooridnates = T_Norm_Vec_Cooridnates_list[index_]
            q_Norm_Only_Angle = T_Norm_Only_Angle_list[index_]
            
            q_Resized_Vec_list.append(q_Resized_Vec)
            q_Resized_Vec_Cooridnates_list.append(q_Resized_Vec_Cooridnates)
            q_Resized_Only_Angle_list.append(q_Resized_Only_Angle)
            
            q_Norm_Vec_list.append(q_Norm_Vec)
            q_Norm_Vec_Cooridnates_list.append(q_Norm_Vec_Cooridnates)
            q_Norm_Only_Angle_list.append(q_Norm_Only_Angle)
            
        
            q_Resized_Vec_dist.append(process_dis(q_Resized_Vec, g_Resized_Vec, output_commnd=lambda x:None)[0][0])
            q_Resized_Vec_Cooridnates_dist.append(process_dis(q_Resized_Vec_Cooridnates, g_Resized_Vec_Cooridnates, output_commnd=lambda x:None)[0][0])
            q_Resized_Only_Angle_dist.append(process_dis(q_Resized_Only_Angle, g_Resized_Only_Angle, output_commnd=lambda x:None)[0][0])
            
            q_Norm_Vec_dist.append(process_dis(g_Norm_Vec, q_Norm_Vec, output_commnd=lambda x:None)[0][0])
            q_Norm_Vec_Cooridnates_dist.append(process_dis(g_Norm_Vec_Cooridnates, q_Norm_Vec_Cooridnates, output_commnd=lambda x:None)[0][0])
            q_Norm_Only_Angle_dist.append(process_dis(g_Norm_Only_Angle, q_Norm_Only_Angle, output_commnd=lambda x:None)[0][0])
        else:
            q_index.append(False)
            
    pose_query = []
    g_Resized_Vec_list = []
    g_Resized_Vec_Cooridnates_list = [] 
    g_Resized_Only_Angle_list = []
    
    g_Norm_Vec_list = []
    g_Norm_Vec_Cooridnates_list = [] 
    g_Norm_Only_Angle_list = []
    g_index = []
    for i,query in enumerate(gallery_set):
        query_path = query[0]
        query_identifier = indentifier_fn(query_path)
        q_id = "test/" + query_identifier
        index_ = image_names == q_id
        if index_.any():            
            g_Resized_Vec_list.append(T_Resized_Vec_list[index_])
            g_Resized_Vec_Cooridnates_list.append(T_Resized_Vec_Cooridnates_list[index_])
            g_Resized_Only_Angle_list.append(T_Resized_Only_Angle_list[index_])
            
            g_Norm_Vec_list.append(T_Norm_Vec_list[index_])
            g_Norm_Vec_Cooridnates_list.append(T_Norm_Vec_Cooridnates_list[index_])
            g_Norm_Only_Angle_list.append(T_Norm_Only_Angle_list[index_])
            g_index.append(True)
        else:
            g_index.append(False)

    q_index = np.array(q_index)
    g_index = np.array(g_index)
    correctness = np.array(correctness)
    

    qf1 = qf1[q_index]
    print(qf1.shape )
    gf1 = gf1[g_index]
    print(gf1.shape )
    
    dismat_feats, _, _ = compute_distance(qf1, gf1, print)
    
    dismat_Resized_Vec_listq_Resized_Vec_list = process_dis(q_Resized_Vec_list, g_Resized_Vec_list)
    dismat_Resized_Vec_Cooridnates_list = process_dis(q_Resized_Vec_Cooridnates_list, g_Resized_Vec_Cooridnates_list)
    dismat_Resized_Only_Angle_list = process_dis(q_Resized_Only_Angle_list, g_Resized_Only_Angle_list)

    dismat_Norm_Vec_list = process_dis(q_Norm_Vec_list, g_Norm_Vec_list)
    dismat_Norm_Vec_Cooridnates_list = process_dis(q_Norm_Vec_Cooridnates_list, g_Norm_Vec_Cooridnates_list)
    dismat_Norm_Only_Angle_list = process_dis(q_Norm_Only_Angle_list, g_Norm_Only_Angle_list)

    df = pd.DataFrame({'Correct': correctness})
    # for key in pose_labels:
    #     df[key] = np.array(pose_labels_dict[key])
    
    df["Resized_Vec_dist"] = np.array(q_Resized_Vec_dist)
    df["Resized_Vec_Cooridnates_dist"] = np.array(q_Resized_Vec_Cooridnates_dist)
    df["Resized_Only_Angle_dist"] = np.array(q_Resized_Only_Angle_dist)
    df["Norm_Vec_dist"] = np.array(q_Norm_Vec_dist)
    df["Norm_Vec_Cooridnates_dist"] = np.array(q_Norm_Vec_Cooridnates_dist)
    df["Norm_Only_Angle_dist"] = np.array(q_Norm_Only_Angle_dist)

    df.corr(method='spearman')
    import seaborn as sns

    sns.heatmap(df.corr(method='pearson'))
    plt.savefig(f"{pickle_path1}_pearson.png")
    plt.clf()

    sns.heatmap(df.corr(method='kendall'))
    plt.savefig(f"{pickle_path1}_kendall.png")
    plt.clf()
    
    sns.heatmap(df.corr(method='spearman'))
    plt.savefig(f"{pickle_path1}_spearman.png")
    plt.clf()
    

    # import pickle
    # with open(f'GOOD.pkl', 'wb') as handle:
    #     pickle.dump(dict(
    #         dismat_Resized_Vec_listq_Resized_Vec_list=dismat_Resized_Vec_listq_Resized_Vec_list, 
    #         dismat_Resized_Vec_Cooridnates_list=dismat_Resized_Vec_Cooridnates_list,
    #         dismat_Resized_Only_Angle_list = dismat_Resized_Only_Angle_list,
    #         dismat_Norm_Vec_list = dismat_Norm_Vec_list, 
    #         dismat_Norm_Vec_Cooridnates_list = dismat_Norm_Vec_Cooridnates_list, 
    #         dismat_Norm_Only_Angle_list = dismat_Norm_Only_Angle_list, 
    #     ), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # with open(f'GT.pkl', 'wb') as handle:
    #     pickle.dump(dict(
    #         dismat_feats=dismat_feats, 
    #     ), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def plot_corelation(x,y, suffix=".png"):
        C = np.corrcoef(x.flatten(), y.flatten())
        # C = np.corrcoef(x, y)
        print(C.shape, C)
        fig, ax = plt.subplots()
        # Create a heatmap
        cax = ax.matshow(C, cmap='coolwarm')
        # Create a colorbar
        fig.colorbar(cax)
        # Show the plot
        plt.savefig(pickle_path1 + suffix)
        plt.clf()

    
    # plot_corelation(dismat_Resized_Vec_listq_Resized_Vec_list , dismat_feats, suffix = "_c1.png")
    # plot_corelation(dismat_Resized_Vec_Cooridnates_list , dismat_feats, suffix = "_c2.png")
    # plot_corelation(dismat_Resized_Only_Angle_list , dismat_feats, suffix = "_c3.png")
    # plot_corelation(dismat_Norm_Vec_list , dismat_feats, suffix = "_c4.png")
    # plot_corelation(dismat_Norm_Vec_Cooridnates_list , dismat_feats, suffix = "_c5.png")
    # plot_corelation(dismat_Norm_Only_Angle_list , dismat_feats, suffix = "_c6.png")



# python Scripts/analysis/pose_res_analysis.py LT_CAL_32_1 LTCC /data/priyank/synthetic/LTCC/
# 162 0.413265306122449
# (2, 2) [[1.         0.03281749]
#         [0.03281749 1.        ]]
# (2, 2) [[ 1.         -0.01605964]
#         [-0.01605964  1.        ]]
# (2, 2) [[1.         0.03363197]
#         [0.03363197 1.        ]]
# (2, 2) [[1.         0.02570529]
#         [0.02570529 1.        ]]
# (2, 2) [[1.         0.02408821]
#  [0.02408821 1.        ]]
# (2, 2) [[1.         0.02575568]
#         [0.02575568 1.        ]]


# python Scripts/analysis/pose_res_analysis.py LT_BM_28_2 LTCC /data/priyank/synthetic/LTCC/
# 165 0.42091836734693877
# (2, 2) [[1.         0.03581237]
#         [0.03581237 1.        ]]
# (2, 2) [[1.         0.00693332]
#         [0.00693332 1.        ]]
# (2, 2) [[1.         0.03515836]
#         [0.03515836 1.        ]]
# (2, 2) [[1.         0.02173604]
#         [0.02173604 1.        ]]
# (2, 2) [[1.         0.02124703]
#  [0.02124703 1.        ]]
# (2, 2) [[1.         0.02165559]
#         [0.02165559 1.        ]]

# python Scripts/analysis/pose_res_analysis.py Final_R_LA_15_B=32_1 LTCC /data/priyank/synthetic/LTCC/
# 182 0.4642857142857143
# (2, 2) [[1.         0.03227294]
#         [0.03227294 1.        ]]
# (2, 2) [[1.       0.004827]
#         [0.004827 1.      ]]
# (2, 2) [[1.         0.03144286]
#         [0.03144286 1.        ]]
# (2, 2) [[1.         0.01355787]
#         [0.01355787 1.        ]]
# (2, 2) [[1.         0.01386721]
#         [0.01386721 1.        ]]
# (2, 2) [[1.         0.01352155]
#         [0.01352155 1.        ]]

# python Scripts/analysis/gender_res_analysis.py PR_CAL_32_1 PRCC /data/priyank/synthetic/PRCC/
# 1954 0.5515100197572679

# python Scripts/analysis/gender_res_analysis.py PR_BM_32_1 PRCC /data/priyank/synthetic/PRCC/
# 2095 0.5913068021450748

# python Scripts/analysis/gender_res_analysis.py Final_PR_R_LA_15_B=32_1 PRCC /data/priyank/synthetic/PRCC/
# 2303 0.65001411233418