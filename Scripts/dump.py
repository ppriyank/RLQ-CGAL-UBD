import os 
import sys
currentdir = os.getcwd() 
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)


import random 
from PIL import Image
from data.utils import read_image , colorname_to_rgb, create_image
import numpy as np 
import pickle
import cv2 
def load_sil(sil):
    sil = read_image(sil)
    sil = np.array(sil)
    sil[sil != 0 ] = 1
    return sil 


colors = ['green', 'blue', 'purple', 'red', 'black', 'white', 'orange', 'yellow', 'pink']
color_to_rgb = {e : colorname_to_rgb(e) for e in colors}
resize_dim=(192,384)    

l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]


def load_pickle(name):
    # Load data (deserialize)
    with open(f'{name}.pkl', 'rb') as handle:
        data = pickle.load(handle)
    return data


def draw_line(coordinates):
    part_line = {}
    for i in range(coordinates.shape[0]):
        part_line[i] = (int(coordinates[i][0])), int(coordinates[i][1])
    return part_line

def draw_circles(img, coordinates, part_line={}, l_pair=None, draw_without_line=None, rgb_opacity=1, save_as_png=None):
    img = img.astype(np.uint8)
    for i in range(coordinates.shape[0]):
        cv2.circle(img, coordinates[i].astype(int), radius=2, color=(255,255,255), thickness=2)
        cv2.circle(img, coordinates[i].astype(int), radius=4, color=(0,0,0), thickness=1)
    if not draw_without_line:
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, (255,0,0), 1)
    img = Image.fromarray(img)
    return img 
    
def draw_circles_on_png(rgb_image, coordinates, part_line={}, l_pair=None, draw_without_line=None, rgb_opacity=1, save_as_png=None):
    img = (np.zeros_like(rgb_image)+ 255).astype(np.uint8)
    for i in range(coordinates.shape[0]):
        cv2.circle(img, coordinates[i].astype(int), radius=2, color=(255,0,0), thickness=2)
        cv2.circle(img, coordinates[i].astype(int), radius=4, color=(0,0,0), thickness=1)
    if not draw_without_line:
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, (255,0,0), 1)
    img = Image.fromarray(img)
    img = img.convert('RGBA')
    datas = img.getdata()
    new_data = []
    newData = []
    for item in datas:
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img
    



def resize_coordinates(x, resize_dim, original_dim):
    return (x * (resize_dim / original_dim)).astype(int)

def process_coordinates(pickle_path, resize_dim, original_dim):
    img = load_pickle( pickle_path )
    scores = img["yolo_scores"]
    if scores is None :
        return None, None, None 
    index = scores.argmax()

    coordinates = img["coordinates"][index]
    # feat = img["feat"][index]

    normalized_cood = img["normalized_coords"]
    if normalized_cood.shape != (17,2):
        normalized_cood = normalized_cood[index]
    # print(normalized_cood.shape)
    # (17,2)
            
    cor_x, cor_y = coordinates[:, 0].astype(int), coordinates[:, 1].astype(int)

    cor_x = resize_coordinates(cor_x, resize_dim[0], original_dim[0])
    cor_y = resize_coordinates(cor_y, resize_dim[1], original_dim[1])
    resize_cordinate = np.column_stack((cor_x, cor_y))
    
    return coordinates, resize_cordinate, normalized_cood

def change_clothes_color(shirt_sil, pant_sil, img, alpha):
    shirt_color, pant_color = random.sample(colors, k=2)
    shirt = create_image(shirt_sil.shape[0], shirt_sil.shape[1], color_to_rgb[shirt_color])
    img = np.array(img)
    pant = create_image(pant_sil.shape[0], shirt_sil.shape[1], color_to_rgb[pant_color])
    sil = pant_sil | shirt_sil
    masked_image = img * (1 - sil) + img * sil * (1 - alpha) + alpha * ( (pant_sil * pant) + (shirt_sil * shirt) )
    return masked_image.astype(np.uint8) 



ltcc="/data/priyank/synthetic/LTCC/"
ltcc_sil="/data/priyank/synthetic/LTCC/masks/ltcc/" 
ltcc_pose="/data/priyank/synthetic/LTCC/AlphaPose/2DPose/"



images = os.listdir(os.path.join(ltcc, "LTCC_ReID", "train"))
images_sils = os.listdir(os.path.join(ltcc_sil, "train"))

selected = random.sample(images, k=5)




for img in selected:
    img_colored = img.replace(".png", "_colored.png")
    img_pose = img.replace(".png", "_pose.png")
    
    img_path = os.path.join(ltcc, "LTCC_ReID", "train", img)
    pose_path = os.path.join(ltcc_pose, "train", img.replace(".png", ""))
    Image.open(img_path).convert('RGB').resize((192, 392)).save(img)    
    sil_name = img.replace(".png", "")
    sil_shirt = os.path.join(ltcc_sil, "train", sil_name + "_shirt.png")
    sil_pant = os.path.join(ltcc_sil, "train", sil_name + "_pant.png")
    assert os.path.exists(sil_pant) and os.path.exists(sil_shirt)
    
    img = Image.open(img_path).convert('RGB')
    width,height = img.size
    original_dim = (width,height)

    coordinates, resize_cordinate, normalized_cood = process_coordinates(pose_path, resize_dim, original_dim)

    pant_sil = load_sil(sil_pant)
    shirt_sil = load_sil(sil_shirt)
    masked_image = change_clothes_color(shirt_sil, pant_sil, img, alpha=0.7)        
    masked_image = Image.fromarray(masked_image)
    masked_image.resize((192, 392)).save(img_colored)
    
    pose = draw_circles_on_png(np.array(img), coordinates, part_line=draw_line(coordinates), l_pair=l_pair, draw_without_line=False, rgb_opacity=0.1, save_as_png=True)
    pose.resize((192, 392)).save(img_pose, "PNG")

    
    



# conda activate bert2
# cd ~/RLQ-CGAL-UBD/
# python Scripts/dump.py