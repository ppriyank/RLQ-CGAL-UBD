import os
import sys
import shutil
import errno
import json
import os.path as osp
import torch
import random
import logging
import numpy as np
import pickle
from einops import rearrange, repeat
from torchvision.utils import save_image
from PIL import Image
import cv2 

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def rearrange_mlr(x):
    x = rearrange(x, "B N ... -> (B N) ... ")
    return x

def expand_input(x, N):
    x = repeat(x, "B ... -> B N ... ", N= N)
    return rearrange_mlr(x)

def reverse_arrange(x, B, N):
    x = rearrange(x, "(B N) ... -> B N ... ", B=B, N= N)
    return x


def set_seed(seed=None):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = ("%s" % seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar', normal_save=True):
    mkdir_if_missing(osp.dirname(fpath))
    if normal_save:
        torch.save(state, fpath)
    if is_best:
        if not normal_save:
            torch.save(state,  osp.join(osp.dirname(fpath), 'best_model.pth.tar'))
        else:
            shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

'''
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
'''


def get_logger(fpath, local_rank=0, name=''):
    # Creat logger
    logger = logging.getLogger(name)
    level = logging.INFO if local_rank in [-1, 0] else logging.WARN
    logger.setLevel(level=level)

    # Output to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level=level) 
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Output to file
    if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
    file_handler = logging.FileHandler(fpath, mode='w')
    file_handler.setLevel(level=level)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    return logger




def save_pickle(data, name):
    # Store data (serialize)
    with open(f'{name}.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(name):
    # Load data (deserialize)
    with open(f'{name}.pkl', 'rb') as handle:
        data = pickle.load(handle)
    return data


def gif_generator(x , name="temp"):
    images = [] 
    # for img in x:y=Image.fromarray(img);images.append(y)
    for img in x:y=Image.open(img).convert('RGB').resize((192, 384));images.append(y)
    images[0].save(f"{name}.gif", save_all=True, append_images=images[1:], duration=50, loop=0)

def video_generator(x , name="temp", fps=30):
    images = [] 
    # for img in x:y=Image.fromarray(img);images.append(y)
    for img in x:y=Image.open(img).convert('RGB').resize((192,384));images.append(np.array(y))
    out = cv2.VideoWriter(f'{name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, (192, 384))
    # out = cv2.VideoWriter(f'{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (192, 384))
    for frame in images:out.write(frame)
    out.release()

