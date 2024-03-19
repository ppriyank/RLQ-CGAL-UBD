import torch
import sys 
import numpy as np 
import time
import pickle 
import logging 
import random 
import pandas as pd 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
import numpy as np


def load_pickle(name):
    # Load data (deserialize)
    with open(f'{name}.pkl', 'rb') as handle:
        data = pickle.load(handle)
    return data


def load_features(pickle_path):
    feature_dump = load_pickle(pickle_path)
    qf = torch.tensor(feature_dump["qf"])
    gf=torch.tensor(feature_dump["gf"])

    q_pids=feature_dump["q_pids"]
    q_camids=feature_dump["q_camids"]
    q_clothes_ids=feature_dump["q_clothes_ids"]

    g_pids=feature_dump["g_pids"]
    g_camids=feature_dump["g_camids"]
    g_clothes_ids=feature_dump["g_clothes_ids"]
    g_image_paths = feature_dump["g_image_paths"]

    f = torch.cat([qf, gf])
    pid = torch.cat([q_pids, g_pids])
    return f, pid

if len(sys.argv) == 3:
    name = sys.argv[1]
else:
    name = sys.argv[3]
HR_F, HR_pid = load_features(sys.argv[1])
LR_F, LR_pid = load_features(sys.argv[2])

def pid_separator(row):
    return int(row.Image.split("_")[0])
    

randomly_selected=  HR_pid.unique()
randomly_selected = [int(e.item()) for e in random.choices(randomly_selected, k = 100)]
print(randomly_selected)
colors = [] 

NUM_COLORS = len(randomly_selected)
# cm = plt.get_cmap('Dark2')
cm = plt.get_cmap('brg')
COLORS = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
area_HR = []
area_LR = []
R_HR_list = [] 
R_LR_list = []
X_HR = []
Y_HR = []
X_LR = []
Y_LR = []

R_HD = 5
R_LR = 2

A_HD = 40
A_LR = 2
            
for pid in randomly_selected:
    shortlisted = HR_pid == pid
    X_HR.append(HR_F[shortlisted])
    Y_HR.append(HR_pid[shortlisted])
    
    X_LR.append(LR_F[shortlisted])
    Y_LR.append(LR_pid[shortlisted])

    c = COLORS.pop(0)
    colors += [c for i in range(shortlisted.sum())]
    
    area_HR += [A_HD for i in range(shortlisted.sum())]
    area_LR += [A_LR for i in range(shortlisted.sum())] 

    R_HR_list += [R_HD for i in range(shortlisted.sum())]
    R_LR_list += [R_LR for i in range(shortlisted.sum())]




X = torch.cat(X_HR + X_LR).numpy()
Y = torch.cat(Y_HR + Y_LR).numpy()
R = np.array(R_HR_list + R_LR_list)
area = np.array(area_HR + area_LR)
colors = np.array(colors + colors)

X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(X)
N_HR = len(X_HR)
N = X_embedded.shape[0]
N_HR = N // 2
    
name_id = name.split("/")[-1]    
with open(f'{name}_tsne.pkl', 'wb') as handle:
    pickle.dump(dict(X=X_embedded, N_HR=N_HR, area=area, 
    ), handle, protocol=pickle.HIGHEST_PROTOCOL)
    


def plot_scatter_by_res(X_embedded, Y, area, colors, subset_HR=True):
    LR_X_x = X_embedded[:,0][R <= 2]
    LR_X_y = X_embedded[:,1][R <= 2]
    LR_Y = Y[R <= 2]
    LR_area = area[R <= 2]
    LR_colors = colors[R <= 2]
    plt.scatter(LR_X_x, LR_X_y, s=5, c="blue", label=LR_Y, alpha=0.9, marker="o")
    N= len(LR_X_x)
    print(len(LR_X_x))

    LR_X_x = X_embedded[:,0][R > 2]
    LR_X_y = X_embedded[:,1][R > 2]
    LR_Y = Y[R > 2]
    LR_area = area[R > 2]
    LR_colors = colors[R > 2]
    if subset_HR:
        L = R[R > 2]
        index = random.choices(range(len(L)), k = N)
        LR_X_x = LR_X_x[index]
        LR_X_y = LR_X_y[index]
        LR_Y = LR_Y[index]
        LR_area = LR_area[index]
        LR_colors = LR_colors[index]
    
    print(len(LR_X_x))
    plt.scatter(LR_X_x, LR_X_y, s=5, c="red", label=LR_Y, alpha=0.9, marker="x")


plt.scatter(X_embedded[:N_HR,0], X_embedded[:N_HR,1], s=area[:N_HR], c=colors[:N_HR], label=Y[:N_HR], alpha=0.2, marker=".")
plt.scatter(X_embedded[N_HR:,0], X_embedded[N_HR:,1], s=area[N_HR:], c=colors[N_HR:], label=Y[N_HR:], alpha=0.2, marker="x")
plt.savefig(f'{name.split("/")[-1]}_pid_res1.png')
plt.clf()

plt.scatter(X_embedded[:N_HR,0], X_embedded[:N_HR,1], s=2, c=colors[:N_HR], label=Y[:N_HR], alpha=0.3, marker=".")
plt.scatter(X_embedded[N_HR:,0], X_embedded[N_HR:,1], s=2, c=colors[N_HR:], label=Y[N_HR:], alpha=0.3, marker="x")
plt.savefig(f'{name.split("/")[-1]}_pid_res2.png')
plt.clf()

plt.scatter(X_embedded[N_HR:,0], X_embedded[N_HR:,1], s=2, c="red", label="LR", alpha=0.6, marker="o")
plt.scatter(X_embedded[:N_HR,0], X_embedded[:N_HR,1], s=2, c="blue", label="HR", alpha=0.3, marker="o")
plt.legend()
plt.savefig(f'{name.split("/")[-1]}_res.png')
plt.clf()



# python Scripts/low_res_analysis_Celeb.py  BM_28_1_TS_Celeb_HD BM_28_1_TS_Celeb_LR
# python Scripts/low_res_analysis_Celeb.py  BM_28_1_Celeb_HD BM_28_1_Celeb_LR


# python Scripts/analysis/low_res_analysis_Celeb.py Celeb_CAL_32_1_HD Celeb_CAL_32_1_LR
# python Scripts/analysis/low_res_analysis_Celeb.py CAL_UBD_32_2_Celeb_HD CAL_UBD_32_2_Celeb_LR
# python Scripts/analysis/low_res_analysis_Celeb.py BM_28_1_TS_Celeb_HD BM_28_1_TS_Celeb_LR
# python Scripts/analysis/low_res_analysis_Celeb.py Celeb_Final_R_LA_15_B=32_1_HD Celeb_Final_R_LA_15_B=32_1_LQ 

# Celeb_CAL_32_1_HD Celeb_CAL_32_1_LR
# CAL_UBD_32_2_Celeb_HD CAL_UBD_32_2_Celeb_LR
# BM_28_1_TS_Celeb_HD BM_28_1_TS_Celeb_LR
# Celeb_Final_R_LA_15_B=32_1_HD Celeb_Final_R_LA_15_B=32_1_LQ