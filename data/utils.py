from PIL import Image
import random 
import matplotlib.colors as mcolors
import numpy as np 
import os 
import os.path as osp
import h5py
try:
    from mmcv.fileio import FileClient
except:
    from mmengine.fileio import FileClient
try:
    from mmcv.runner.utils import set_random_seed
except:
    from mmengine.runner import set_random_seed

import decord
import io 
io_backend='disk'
file_client = FileClient(io_backend)
import cv2

def random_crop(img, H, W):
    w, h = img.size 
    if w - W != 0 :
        x1 = random.randrange(0, w - W)
    else:
        x1 = 0 
    if h - H != 0 :
        y1 = random.randrange(0, h - H)
    else:
        y1 = 0

    # img = np.array(img)
    img = img.crop((x1, y1, x1 + W, y1 + H))
    return img

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def middle_frame(x):
    return x[len(x) // 2]

def random_frame(x):
    return random.choice(x)

def read_h5py(img_paths):
    clip = np.array(h5py.File(osp.join(img_paths, 'clip.h5'), 'r')['data'])
    return clip

def video_without_img_paths(video_path, num_threads=1):
    file_obj = io.BytesIO(file_client.get(video_path))
    container = decord.VideoReader(file_obj, num_threads=num_threads)
    return container

def read_image(img_path, illumination= None):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    if illumination:
        img = fix_illumination(img)
    return img

def create_image(height, width, color):
    array = np.zeros([height, width, 3], dtype=np.uint8)
    array[:,:] = color  # Fill array with color
    return array

def create_mask(height, width):
    array = np.zeros([height, width, 3], dtype=np.uint8)
    return array

def colorname_to_rgb(colorname):
    rgb_float = mcolors.to_rgb(colorname)
    rgb_uint8 = tuple((np.array(rgb_float)*255).astype(np.uint8))
    return rgb_uint8

def colorname_to_hex(colorname):
    return mcolors.to_hex(colorname)

def crop_img(sil, rgb, padding=10):
    W,H,_ = sil.shape
    
    y_sum = sil.sum(axis=0).sum(axis=1) # y cropping is actually x xropping 
    y_top = (y_sum != 0).argmax(axis=0)
    y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
    y_mid = (y_top + y_btm) // 2
    y_dist = y_btm - y_top
    
    x_sum = sil.sum(axis=1).sum(axis=1)
    x_left = (x_sum != 0).argmax(axis=0)
    x_right = (x_sum != 0).cumsum(axis=0).argmax(axis=0)
    x_mid = (x_left + x_right) // 2
    x_dist = x_right - x_left
    
    if y_dist < 100 or x_dist < 100:return sil, rgb

    sil = sil[ max(0, x_mid - x_dist//2 - padding): min(W, x_mid + x_dist//2 + padding), max(0, y_mid - y_dist//2 - padding): min(H, y_mid + y_dist//2 + padding)]
    rgb = rgb[ max(0, x_mid - x_dist//2 - padding): min(W, x_mid + x_dist//2 + padding), max(0, y_mid - y_dist//2 - padding): min(H, y_mid + y_dist//2 + padding)]
    return sil, rgb


def crop_sil(sil, padding=10):
    W,H,_ = sil.shape
    
    y_sum = sil.sum(axis=0).sum(axis=1) # y cropping is actually x xropping 
    y_top = (y_sum != 0).argmax(axis=0)
    y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
    y_mid = (y_top + y_btm) // 2
    y_dist = y_btm - y_top
    
    x_sum = sil.sum(axis=1).sum(axis=1)
    x_left = (x_sum != 0).argmax(axis=0)
    x_right = (x_sum != 0).cumsum(axis=0).argmax(axis=0)
    x_mid = (x_left + x_right) // 2
    x_dist = x_right - x_left
    
    if y_dist < 100 or x_dist < 100:return sil

    sil = sil[ max(0, x_mid - x_dist//2 - padding): min(W, x_mid + x_dist//2 + padding), max(0, y_mid - y_dist//2 - padding): min(H, y_mid + y_dist//2 + padding)]
    return sil

