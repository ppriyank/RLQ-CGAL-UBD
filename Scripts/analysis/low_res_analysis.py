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
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from kneed import KneeLocator


def load_pickle(name):
    # Load data (deserialize)
    with open(f'{name}.pkl', 'rb') as handle:
        data = pickle.load(handle)
    return data

root = "/data/priyank/synthetic/LTCC/LTCC_ReID/"
size_table = pd.read_csv("/home/priyank/CCReID/Scripts/Helper/LTCC_Size.csv")
pickle_path = sys.argv[1]
feature_dump = load_pickle(pickle_path)

qf = torch.tensor(feature_dump["qf"])
gf=torch.tensor(feature_dump["gf"])

q_pids=feature_dump["q_pids"]
q_camids=feature_dump["q_camids"]
q_clothes_ids=feature_dump["q_clothes_ids"]
q_image_paths=feature_dump['q_image_paths']


g_pids=feature_dump["g_pids"]
g_camids=feature_dump["g_camids"]
g_clothes_ids=feature_dump["g_clothes_ids"]
g_image_paths = feature_dump["g_image_paths"]


def pid_separator(row):
    return int(row.Image.split("_")[0])
    
def sample_datapoints():
    lr_samples = size_table[( (size_table["<32"] > 0) | (size_table["32-64"] > 0)  ) & (size_table["Category"] == "test")]
    lr_samples["pid"] = lr_samples.apply(pid_separator, 1)
    lr_samples = lr_samples.groupby("pid").sum()
    lr_samples["LR"] = lr_samples["<32"] + lr_samples["32-64"]

    pids = (lr_samples).sort_values(by=['LR'], ascending=False).LR.reset_index()
    print(pids)
    return pids

def generate_tsne_embeddings():
    gf_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(gf)
    return gf_embedded

def group_as_size(gf_embedded, randomly_selected):
    X = []
    Y = []
    # <32  32-64  64-128  128-256  >256
    # 1     2      3       4        5
    R = []
    colors = [] 
    area = []
    for pid in randomly_selected:
        shortlisted = g_pids == pid
        X.append(gf_embedded[shortlisted])
        Y.append(g_pids[shortlisted])
        Z = np.array(g_image_paths)[shortlisted]
        c = COLORS.pop(0)
        for img_path in Z:
            category, name = img_path.split("/")
            resolution = size_table[(size_table.Category == category) & (size_table.Image == name)]
            x = 0
            if resolution["<32"].item():
                x = 1 
                r = 2
            elif resolution["32-64"].item():
                x = 2
                r = 2
            elif resolution["64-128"].item():
                x = 3
                r = 40
            elif resolution["128-256"].item():
                x = 4
                r = 40
            elif resolution[">256"].item() :
                x = 5
                r = 40
            R.append(x)
            assert x != 0 
            colors.append(c)  
            area.append( r )

    X_embedded = np.concatenate(X)
    Y = torch.cat(Y).numpy()
    R = np.array(R)
    area = np.array(area)
    colors = np.array(colors)
    N = X_embedded.shape[0]
    return X_embedded, Y, R, area, colors, N




pids = sample_datapoints()

gf_embedded = gf
# gf_embedded = generate_tsne_embeddings()
pids = pids[pids.LR > 40]
randomly_selected=  g_pids.unique()
randomly_selected = set(pids.pid)
print(randomly_selected)
NUM_COLORS = len(randomly_selected)
# cm = plt.get_cmap('Dark2')
cm = plt.get_cmap('brg')
COLORS = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
X_embedded, Y, R, area, colors, N = group_as_size(gf_embedded, randomly_selected)
subset_HR = False


def k_means_find_k(X):
    inertias = []
    for i in range(1, 6):
        kmeans = KMeans(n_clusters=K, random_state=0, max_iter=1000).fit(X)
        intertia = kmeans.inertia_
        inertias.append(intertia)
    K = KneeLocator(x=range(1, 6),  y=inertias,  curve='convex',  direction='decreasing', S=3).knee
    if K is None:
        K = 2
    kmeans = KMeans(n_clusters=K, random_state=0, max_iter=1000).fit(X)
    c_lr = kmeans.cluster_centers_
    return c_lr, K 
        
def k_means_only(X):
    kmeans = KMeans(n_clusters=K, random_state=0, max_iter=1000).fit(X)
    c_hr = kmeans.cluster_centers_        
    return c_hr
        
    


def plot_scatter_by_pid(X_embedded, Y, area, colors, subset_HR=True):
    LR_X_x = X_embedded[:,0][R <= 2]
    LR_X_y = X_embedded[:,1][R <= 2]
    LR_Y = Y[R <= 2]
    LR_area = area[R <= 2]
    LR_colors = colors[R <= 2]
    # plt.scatter(LR_X_x, LR_X_y, s=LR_area, c=LR_colors, label=LR_Y, alpha=0.9, marker=".")
    plt.scatter(LR_X_x, LR_X_y, s=10, c=LR_colors, label=LR_Y, alpha=0.9, marker="o")
    N= len(LR_X_x)
    print(len(LR_X_x))

    LR_X_x = X_embedded[:,0][R > 2]
    LR_X_y = X_embedded[:,1][R > 2]
    LR_Y = Y[R > 2]
    LR_area = area[R > 2]
    LR_colors = colors[R > 2]

    L = R[R > 2]
    index = random.choices(range(len(L)), k = N)
    if subset_HR:
        LR_X_x = LR_X_x[index]
        LR_X_y = LR_X_y[index]
        LR_Y = LR_Y[index]
        LR_area = LR_area[index]
        LR_colors = LR_colors[index]

    print(len(LR_X_x))
    plt.scatter(LR_X_x, LR_X_y, s=10, c=LR_colors, label=LR_Y, alpha=0.3, marker="o")

def plot_scatter_by_res(X_embedded, Y, area, colors, subset_HR=True):
    LR_X_x = X_embedded[:,0][R <= 2]
    LR_X_y = X_embedded[:,1][R <= 2]
    LR_Y = Y[R <= 2]
    LR_area = area[R <= 2]
    LR_colors = colors[R <= 2]
    plt.scatter(LR_X_x, LR_X_y, s=10, c="red", label=LR_Y, alpha=0.9, marker=".")
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
    plt.scatter(LR_X_x, LR_X_y, s=10, c="blue", label=LR_Y, alpha=0.3, marker="o")

def plot_scatter_by_Center(X_embedded, Y, area, colors, subset_HR=True, all_v_all=True, K = 3, only_lr=False, no_line=None ):

    fig, ax = plt.subplots(figsize=(8, 6))

    LR_X_x = X_embedded[:,0][R <= 2]
    LR_X_y = X_embedded[:,1][R <= 2]
    LR_Y = Y[R <= 2]
    LR_area = area[R <= 2]
    LR_colors = colors[R <= 2]

    HR_X_x = X_embedded[:,0][R > 2]
    HR_X_y = X_embedded[:,1][R > 2]
    HR_Y = Y[R > 2]
    HR_area = area[R > 2]
    HR_colors = colors[R > 2]
    H = R[R > 2]
    
    y_max = max(HR_X_y.max(), LR_X_y.max(), )
    y_min = min(LR_X_y.min(), HR_X_y.min()) 
    x_min = min(LR_X_x.min(), HR_X_x.min()) 
    x_max = max(LR_X_x.max(), HR_X_x.max(), ) 
    
    for pid in np.unique(LR_Y):
        index = LR_Y == pid
        cx_lr = LR_X_x[index].mean()
        cy_lr = LR_X_y[index].mean()
        color = LR_colors[index][0]
        X = np.concatenate([[LR_X_x[index]], [LR_X_y[index]]]).T
        # c_lr, K = k_means_find_k(X)

        clustering = DBSCAN(eps=10, min_samples=5).fit(X)
        centroids = []
        for label in np.unique(clustering.labels_):
            cluster_values = X[clustering.labels_ == label]
            # Could also use median here
            centroid = np.mean(cluster_values, axis=0)
            print(centroid)
            centroids.append(centroid)
        c_lr = np.array(centroids) 
        ax.scatter(LR_X_x[index], LR_X_y[index], fc=color, s=50, lw=1, ec="white", zorder=12, alpha=0.5, marker=".", label=pid)
        ax.scatter(c_lr[:,0], c_lr[:,1], fc=color, s=100, lw=1.5, label=pid, zorder=12, alpha=0.9, marker="x")

        index = HR_Y == pid
        cx_hr = HR_X_x[index].mean()
        cy_hr = HR_X_y[index].mean()
        # plt.scatter(HR_X_x[index], HR_X_y[index], s=10, c=color, label=pid, alpha=0.2, marker=".")
        ax.scatter(HR_X_x[index], HR_X_y[index], fc=color, s=20, lw=1, ec="white", zorder=12, alpha=0.1, marker="o", label=pid)        
        X = np.concatenate([[HR_X_x[index]], [HR_X_y[index]]]).T
        clustering = DBSCAN(eps=10, min_samples=5).fit(X)
        centroids = []
        for label in np.unique(clustering.labels_):
            cluster_values = X[clustering.labels_ == label]
            # Could also use median here
            centroid = np.mean(cluster_values, axis=0)
            print(centroid)
            centroids.append(centroid)
        c_hr = np.array(centroids) 
        # c_hr = k_means_only(X)
        ax.scatter(c_hr[:,0], c_hr[:,1], fc=color, s=100, lw=1.5,  label=pid, zorder=12, alpha=0.3, marker="x")

        c_lr_x = c_lr[:,0].reshape(-1)
        c_lr_y = c_lr[:,1].reshape(-1)

        c_hr_x = c_hr[:,0].reshape(-1)
        c_hr_y = c_hr[:,1].reshape(-1)

        if all_v_all:
            
            x_values = []
            y_values = []
            for i in range(K):
                for j in range(K):
                    x_values.append(c_lr_x[i])
                    x_values.append(c_hr_x[j])
                    y_values.append(c_lr_y[i])
                    y_values.append(c_hr_y[j])

            # x_values = [cx_lr, cx_hr]
            # y_values = [cy_lr, cy_hr]
            # plt.plot(x_values, y_values, c=color, alpha=0.5)
            ax.plot(x_values, y_values, color=color, lw=5, alpha=0.2)
            # plt.savefig(f'temp.png')
        elif only_lr:
            ax.plot(np.concatenate((c_lr_x , [c_lr_x[0]])) , np.concatenate((c_lr_y , [c_lr_y[0]])), color=color, lw=1, alpha=0.5)
        elif no_line:
            '''
            do nothing
            '''
        else:
            index = pairwise_distances(c_lr, c_hr).argmin(-1)            

            x_values = np.concatenate([[c_lr_x], [c_hr_x[index]]]).T
            x_values = x_values.reshape(-1)

            y_values = np.concatenate([[c_lr_y], [c_hr_y[index]]]).T
            y_values = y_values.reshape(-1)
            for i in range(K):
                # x_values[2*i:2*i+2],y_values[2*i:2*i+2]
                # plt.plot(x_values[2*i:2*i+2], y_values[2*i:2*i+2], c=color, alpha=0.5)
                ax.plot(x_values[2*i:2*i+2], y_values[2*i:2*i+2], color=color, lw=1, alpha=0.9)
            
        
        ax.yaxis.set_ticks((y_min, y_max))
        # ax.yaxis.set_ticklabels((y_min, y_max))
        ax.yaxis.set_tick_params(length=6, width=1.2)
        # plt.savefig(f'temp.png')                
        # Customize y-axis ticks
        ax.xaxis.set_ticks((x_min, x_max))
        # ax.xaxis.set_ticklabels((x_min, x_max), fontsize=20)
        ax.xaxis.set_tick_params(length=6, width=1.2)

        # Make gridlines be below most artists.
        ax.set_axisbelow(True)

        # Add grid lines
        # ax.grid(axis = "both", color="#A8BAC4", lw=1.2)
        
        # Remove all spines but the one in the bottom
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # ax.spines["left"].set_visible(False)

        # Customize bottom spine
        ax.spines["bottom"].set_lw(1.2)
        ax.spines["bottom"].set_capstyle("butt")

        # # Set custom limits
        ax.set_ylim((y_min, y_max))
        ax.set_xlim((x_min, x_max))
        # plt.savefig(f'temp.png')

            
            
    # plt.scatter(LR_X_x, LR_X_y, s=10, c=LR_colors, label=LR_Y, alpha=0.1, marker="o")

def normalize(v):
    norm = np.linalg.norm(v, axis=-1)
    norm += 1e-6 
    return v / np.expand_dims(norm,-1)

def plot_histogram_intra_class(X_embedded, Y, area, colors):

    LR_X = X_embedded[R <= 2]
    LR_Y = Y[R <= 2]
    LR_area = area[R <= 2]
    LR_colors = colors[R <= 2]

    HR_X = X_embedded[R > 2]
    HR_Y = Y[R > 2]
    HR_area = area[R > 2]
    HR_colors = colors[R > 2]
    H = R[R > 2]
    K1 = 1000
    K2 = 50
    N = 1000
    metrics = 'manhattan'
    # metrics = 'l1'
    metrics = 'l2'
    
    X_NEG_lr = []
    X_POS_lr = []
    X_NEG_lr_hr = []
    X_POS_lr_hr = []
    X_POS_hr = []
    X_NEG_hr = []

    Y = [] 
    pids = sorted(np.unique(LR_Y))
    # LR_X = normalize(LR_X)
    # HR_X = normalize(HR_X)

    done_lr = np.full((len(LR_Y)), True)
    done_hr = np.full((len(HR_Y)), True)

    for pid in pids:
        pos_index = LR_Y == pid
        neg_index = ~pos_index & done_lr
        done_lr[pos_index] = False 

        X_LR = LR_X[pos_index]
        X_LR_NPID = LR_X[neg_index]

        POS_lr = pairwise_distances(X_LR, X_LR, metric=metrics)
        POS_lr = np.sort(POS_lr, axis=-1)[:, 1:]
        # POS_lr = POS_lr[:K2+1].mean(-1)
        POS_lr = POS_lr.reshape(-1)
        # X_POS_lr_hr = random.choices(X_POS_lr_hr, k = N_LR_LR)
        X_POS_lr.append(POS_lr)

        if neg_index.sum() != 0 :
            NEG_lr = pairwise_distances(X_LR, X_LR_NPID, metric=metrics)
            # NEG_lr = np.sort(NEG_lr, axis=-1)[:, :K1].mean(-1)
            NEG_lr = NEG_lr.reshape(-1)
            X_NEG_lr.append( NEG_lr )

        pos_index = HR_Y == pid
        neg_index = ~pos_index & done_hr
        done_hr[pos_index] = False 
        
        X_HR = HR_X[pos_index]
        X_HR_NPID = HR_X[neg_index]

        POS_lr_hr = pairwise_distances(X_LR, X_HR, metric=metrics)
        # POS_lr_hr = np.sort(POS_lr_hr, axis=-1)[:, :K2].mean(-1)
        POS_lr_hr = POS_lr_hr.reshape(-1)
        X_POS_lr_hr.append(POS_lr_hr )

        POS_hr = pairwise_distances(X_HR, X_HR, metric=metrics)
        POS_hr = np.sort(POS_hr, axis=-1)[:, 1:]
        # POS_hr = POS_hr[:K1].mean(-1)
        POS_hr = POS_hr.reshape(-1)
        X_POS_hr.append(POS_hr)

        if neg_index.sum() != 0 :
            NEG_hr = pairwise_distances(X_HR, X_HR_NPID, metric=metrics)
            # NEG_hr = np.sort(NEG_hr, axis=-1)[:, :K1].mean(-1)
            NEG_hr = NEG_hr.reshape(-1)
            X_NEG_hr.append( NEG_hr )
            NEG_lr_hr = pairwise_distances(X_LR, X_HR_NPID, metric=metrics)
            # NEG_lr_hr = np.sort(NEG_lr_hr, axis=-1)[:, :K1].mean(-1)
            NEG_lr_hr = NEG_lr_hr.reshape(-1)
            X_NEG_lr_hr.append( NEG_lr_hr )

        
        
        # X.append(np.mean(X_LR_list))
        # Y.append(np.mean(X_HR_list))

    X_NEG_lr = np.concatenate(X_NEG_lr)
    X_POS_lr = np.concatenate(X_POS_lr)
    
    X_NEG_lr_hr = np.concatenate(X_NEG_lr_hr)
    X_POS_lr_hr = np.concatenate(X_POS_lr_hr)

    X_POS_hr = np.concatenate(X_POS_hr)
    X_NEG_hr = np.concatenate(X_NEG_hr)

    print(X_NEG_lr.shape, X_NEG_lr_hr.shape, X_NEG_hr.shape)
    print(X_POS_lr.shape, X_POS_lr_hr.shape, X_POS_hr.shape)

    N_LR_LR = len(X_NEG_lr)
    X_NEG_lr_hr = random.choices(X_NEG_lr_hr, k = N_LR_LR)
    X_NEG_hr = random.choices(X_NEG_hr, k = N_LR_LR)
    
    n1, bins1, _ = plt.hist(X_NEG_lr, bins=N, alpha=0.5, label="LR-LR", color="red")
    n2, bins2, _ = plt.hist(X_NEG_hr, bins=N, alpha=0.5, label="HR-HR", color="blue")
    
    intersection = np.minimum(n1, n2)
    excluded = n1 - intersection
    print(f"NEG LR - LR  <----> NEG HR - HR", excluded.sum())
    # plt.hist(X_NEG_lr_hr, bins=N, alpha=0.5, label="LR-HR", color="purple")
    plt.legend()
    plt.savefig(f'{pickle_path.split("/")[-1]}_NEG_Hist.png')
    plt.clf()

    N_LR_LR = len(X_POS_lr)
    X_POS_lr_hr = random.choices(X_POS_lr_hr, k = N_LR_LR)
    X_POS_hr = random.choices(X_POS_hr, k = N_LR_LR)

    plt.hist(X_POS_hr, bins=N, alpha=0.5, label="HR-HR", color="blue")
    plt.hist(X_POS_lr, bins=N, alpha=0.5, label="LR-LR", color="red")
    # plt.hist(X_POS_lr_hr, bins=N, alpha=0.35, label="LR-HR", color="purple")
    # plt.vlines(x = np.mean(X), ymin = 0, ymax = 150, colors = 'red', label = 'Mean intra LR  Distance',  linestyle='dashed')
    # plt.vlines(x = np.mean(Y), ymin = 0, ymax = 150, colors = 'blue', label = 'Mean intra HR  Distance',  linestyle='dashed')
    # plt.vlines(x = np.mean(X_POS_hr), ymin = 0, ymax = 2200, colors = 'blue', linestyle='dashed')
    # plt.vlines(x = np.mean(X_POS_lr), ymin = 0, ymax = 2200, colors = 'red', linestyle='dashed')
    # plt.vlines(x = np.mean(X_POS_lr_hr), ymin = 0, ymax = 2200, colors = 'purple', linestyle='dashed')
    plt.legend()
    plt.savefig(f'{pickle_path.split("/")[-1]}_POS_Hist.png')
    
    with open(f'{pickle_path.split("/")[-1]}_Dist.pkl', 'wb') as handle:
        pickle.dump(dict(
            X_NEG_lr=X_NEG_lr,  X_NEG_hr=X_NEG_hr, X_NEG_lr_hr=X_NEG_lr_hr, X_POS_hr=X_POS_hr, X_POS_lr=X_POS_lr, X_POS_lr_hr=X_POS_lr_hr,
        ), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # plt.hist(X_NEG_hr, bins=N, alpha=0.5, label="HR-HR", color="blue")
    # plt.hist(X_NEG_lr_hr, bins=N, alpha=0.5, label="LR-HR", color="violet")
    # plt.hist(X_NEG_lr, bins=N, alpha=0.5, label="LR-LR", color="red")
    # plt.savefig(f'{pickle_path.split("/")[-1]}_NEG_Hist.png')
    
    # plt.hist(Y, bins=N, alpha=0.5, label="HR", color="blue")
    # plt.savefig(f'{pickle_path.split("/")[-1]}_Hist.png')
    # plt.clf()
    
    # plt.xlim(-1.5, 0.5)
    # plt.savefig(f'temp.png')                
    
    



# plot_scatter_by_pid(X_embedded, Y, area, colors, subset_HR=subset_HR)
# plt.savefig(f'{pickle_path.split("/")[-1]}_pid_res.png')
# plt.clf()

# plot_scatter_by_Center(X_embedded, Y, area, colors, subset_HR=subset_HR, all_v_all=True, K=3)
# plt.savefig(f'{pickle_path.split("/")[-1]}_pid_Centers_ALL.png')
# plt.clf()

# plot_scatter_by_Center(X_embedded, Y, area, colors, subset_HR=subset_HR, all_v_all=False, only_lr=True, K=3, )
# plt.savefig(f'{pickle_path.split("/")[-1]}_pid_Centers_lr.png')
# plt.clf()

# plot_scatter_by_Center(X_embedded, Y, area, colors, subset_HR=subset_HR, all_v_all=False, only_lr=False, K=3)
# plt.savefig(f'{pickle_path.split("/")[-1]}_pid_Centers_Nearest.png')
# plt.clf()

# plot_scatter_by_Center(X_embedded, Y, area, colors, subset_HR=subset_HR, all_v_all=False, only_lr=False, K=3, no_line=True)
# plt.savefig(f'{pickle_path.split("/")[-1]}_pid.png')
# plt.clf()




# plot_scatter_by_res(X_embedded, Y, area, colors, subset_HR=subset_HR )
# plt.savefig(f'{pickle_path.split("/")[-1]}_only_res.png')
# plt.clf()

plot_histogram_intra_class(X_embedded, Y, area, colors)
plt.clf()

# python -W ignore Scripts/analysis/low_res_analysis.py  BM_28_1_LTCC      
# 59127.0
# python -W ignore Scripts/analysis/low_res_analysis.py  BM_28_2_TS_LTCC
# 14736.0
