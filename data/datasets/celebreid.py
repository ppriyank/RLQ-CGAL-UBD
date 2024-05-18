import os
import re
from glob import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat
from collections import defaultdict

import pandas as pd 

class CelebreID(object):

    def __init__(self, root, name_pattern = "celeb", dataset_sampling=None, **kwargs):
        self.root = root

        self.fnames = []
        self.pids = []
        self.ret = []
        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes, num_camera = \
            self.preprocess(name_pattern = name_pattern, category="train")

        if dataset_sampling:
            N = len(train)
            train = random.sample(train, int((N * dataset_sampling)/100) )
            num_train_imgs = len(train)
            
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes, num_test_cam = \
            self.preprocess_test(name_pattern = name_pattern)

        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes

        logger = logging.getLogger('reid.dataset')
        logger.info("=> CelebreID loaded")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        logger.info("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
        logger.info("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  ----------------------------------------")
        logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  ----------------------------------------")

        logger.info("  ----------------------------------------")
        logger.info("  subset   |  '<128' | '128-256' | '>256' ")
        logger.info("  ----------------------------------------")
        logger.info("  train    |  {:5d}  |   {:5d}   |  {:5d} | ".format(0, 20208, 0 ))
        logger.info("  query    |  {:5d}  |   {:5d}   |  {:5d} | ".format(0, 2972,  0))
        logger.info("  gallery  |  {:5d}  |   {:5d}   |  {:5d} | ".format(0, 11006, 0))
        logger.info("  ----------------------------------------")
        
        self.clothes_dict = None
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_cam = num_camera
        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes
        print(pid2clothes.shape)
        self.original_num_clothes = self.num_train_clothes 

    def preprocess(self, name_pattern, category = "train"):
        if name_pattern=='celeb':
            pattern = re.compile(r'([-\d]+)_(\d)')
        else:
            pattern = re.compile(r'([-\d]+)_c(\d)')
        
        fpaths = sorted(glob(osp.join(self.root, category, '*.jpg')))
        all_pids = {}

        clothes_counter = -1 
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            clothes_counter += 1

        clothes_counter += 1
        num_clothes = clothes_counter
        num_pids = len(all_pids)
        dataset = []
        pid2clothes = np.zeros((num_pids, clothes_counter))
        
        clothes_counter = -1
        cam_container = set()        
        
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            cam -= 1
            cam_container.add(cam)
            pid = all_pids[pid]
            clothes_counter += 1
            dataset.append((fpath, pid, cam, clothes_counter))
            pid2clothes[pid, clothes_counter] = 1
        
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs, num_clothes, pid2clothes, len(cam_container)

    def preprocess_test(self, name_pattern):
        if name_pattern=='celeb':
            pattern = re.compile(r'([-\d]+)_(\d)')
        else:
            pattern = re.compile(r'([-\d]+)_c(\d)')

        fpaths = sorted(glob(osp.join(self.root, "query", '*.jpg')))
        all_pids = {}

        query = []
        cam_container = set()        
        clothes_counter = -1

        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            cam -= 1
            cam_container.add(cam)
            pid = all_pids[pid]
            clothes_counter += 1
            query.append((fpath, pid, cam, clothes_counter))
        
        num_query_imgs = len(query)

        fpaths = sorted(glob(osp.join(self.root, "gallery", '*.jpg')))
        gallery = [] 
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            cam -= 1
            cam_container.add(cam)
            pid = all_pids[pid]
            clothes_counter += 1
            gallery.append((fpath, pid, cam, clothes_counter))
        
        num_gallery_imgs = len(gallery)

        num_pids = len(all_pids)
        num_clothes = 1
        return query, gallery, num_pids, num_query_imgs, num_gallery_imgs, num_clothes, len(cam_container)


class CelebreID_custom_colors(CelebreID):
    def __init__(self, root='data', new_clothes_pids = True, **kwargs):
        super().__init__(root=root, **kwargs)
        
        colors_full = [('green', 1), ('blue',1), ('purple',1),
            ('red',1), ('black',1), ('white',1), ('orange',1), ('yellow',1), ('pink',1)]

        new_colors = []
        label = 0 
        self.clothes_dict = {}
        for top_color in colors_full:
            for bottom_color in colors_full:
                self.clothes_dict[(top_color[0],bottom_color[0])] = label
                label += 1

        self.original_num_clothes = self.num_train_clothes 
        if new_clothes_pids:
            self.num_train_clothes = self.original_num_clothes + label * self.num_train_pids 
            self.new_pid_2_clothes(num_clothes=self.num_train_clothes, label=label)
        
    def new_pid_2_clothes(self, num_clothes, label):

        pid2clothes = np.zeros((self.num_train_pids, num_clothes))
        for pid in range(self.num_train_pids ):
            start_index = self.original_num_clothes + pid * label
            end_index = self.original_num_clothes + (pid + 1) * label
            pid2clothes[pid, start_index : end_index ] = 1
        
        pid2clothes[:, :self.original_num_clothes] = self.pid2clothes
        self.pid2clothes = pid2clothes
        
class CelebreID_CC_gender(CelebreID_custom_colors):
    def __init__(self, root='data', gender_file=None, **kwargs):
        
        df = pd.read_csv(gender_file)
        self.gender_mapping = {i[1]:i[2] for i in df.values}
        
        super().__init__(root=root, **kwargs)
        
    def preprocess(self, name_pattern, category = "train"):
        if name_pattern=='celeb':
            pattern = re.compile(r'([-\d]+)_(\d)')
        else:
            pattern = re.compile(r'([-\d]+)_c(\d)')
        
        fpaths = sorted(glob(osp.join(self.root, category, '*.jpg')))
        all_pids = {}

        clothes_counter = -1 
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            clothes_counter += 1

        clothes_counter += 1
        num_clothes = clothes_counter
        num_pids = len(all_pids)
        dataset = []
        pid2clothes = np.zeros((num_pids, clothes_counter))
        
        clothes_counter = -1
        cam_container = set()        
        
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            cam -= 1
            gender = self.gender_mapping[pid]
            cam_container.add(cam)
            pid = all_pids[pid]
            clothes_counter += 1
            dataset.append((fpath, pid, cam, clothes_counter, gender))
            pid2clothes[pid, clothes_counter] = 1
        
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs, num_clothes, pid2clothes, len(cam_container)

    


if __name__ == "__main__":
    model = CelebreID(root="/home/c3-0/datasets/ID-Dataset/Celeb-reID/")

# srun --pty --cpus-per-task=8 bash
# cd ~/CCReID/data/datasets/
# python celebreid.py
