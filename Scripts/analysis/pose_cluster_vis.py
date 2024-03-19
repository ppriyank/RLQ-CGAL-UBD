import pandas as pd 
import os 
from PIL import Image

def read_image(path):
    mask = Image.open(path).convert('RGB')
    mask = mask.resize((192,384))
    return mask 


def make_folder(name):
    try: 
        os.mkdir(name) 
    except OSError as error: 
        _ = 0 
    

src = "/data/priyank/synthetic/LTCC/LTCC_ReID/train/"

df = pd.read_csv('Scripts/Helper/LTCC_Pose_Cluster.csv')
pose_clusters = df['R_LA_15']
clusters = set(list(pose_clusters))

make_folder("Samples/")
for cl in clusters:
    make_folder(f"Samples/{cl}")
    samples = df[df.R_LA_15 == cl].Image.sample(frac=0.5)
    samples = [read_image(os.path.join(src , x + ".png")).save(f"Samples/{cl}/{x}.png") for x in samples]
    
# cd ~/CCReID
# python Scripts/analysis/pose_cluster_vis.py