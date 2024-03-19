
import pandas as pd 
import os 
from PIL import Image
import random 
import numpy as np 
import cv2
import torchvision.transforms as transforms

def read_image(path):
    mask = Image.open(path).convert('RGB')
    print(mask)
    mask = mask.resize((192,384))
    return mask 


def make_folder(name):
    try: 
        os.mkdir(name) 
    except OSError as error: 
        _ = 0 
    

def create_low_res(img_hr, low_res):
    H,W = img_hr.size    
    ratio = random.choice(range(*low_res)) / min(H,W) 
    print(ratio, (round( H * ratio), round( W *ratio)) )
    img_lr = img_hr.resize((round( H * ratio), round( W *ratio)))
    img_lr = img_lr.resize( (H , W) )
    return img_lr

def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
    k = k * ( 1.0 / np.sum(k) )        
    return cv2.filter2D(image, -1, k) 

def create_blur_motion(img_hr, motion_blur, motion_blur_angle):
    blur_strength = random.choice(range(*motion_blur))
    blur_angle = random.choice(range(*motion_blur_angle))
    img_lr = apply_motion_blur(np.array(img_hr), blur_strength, blur_angle)
    img_lr = Image.fromarray( img_lr )
    return img_lr

def create_g_blur(img_hr, g_blur):
    blur_strength = random.choice(range(*g_blur, 2)) + 1
    img_lr = transforms.GaussianBlur(blur_strength)(img_hr)
    return img_lr



src = "/data/priyank/synthetic/Celeb-reID/train/"
selected = random.sample(os.listdir(src) , k =4) 
make_folder("Samples/")
low_res=(16,64)
motion_blur_angle=(0,180)
motion_blur=(8,20)
g_blur=[4,22]
for img_name in selected:
    img = read_image(os.path.join(src , img_name))
    img_name = img_name.replace(".jpg", "").replace(".png", "")
    img.save(f"Samples/{img_name}.png")
    lr = create_low_res(img, low_res)
    lr.save(f"Samples/{img_name}_pix.png")
    mb = create_blur_motion(img, motion_blur, motion_blur_angle)
    mb.save(f"Samples/{img_name}_mb.png")
    gb = create_g_blur(img, g_blur)
    gb.save(f"Samples/{img_name}_gb.png")




# cd ~/CCReID
# python Scripts/analysis/vis_synthetic_lq.py