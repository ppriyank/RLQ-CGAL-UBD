import os
import re
import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat
from tools.utils import mkdir_if_missing, write_json, read_json

import pandas as pd 

class PRCC(object):
    """ PRCC

    Reference:
        Yang et al. Person Re-identification by Contour Sketch under Moderate Clothing Change. TPAMI, 2019.

    URL: https://drive.google.com/file/d/1yTYawRm4ap3M-j0PjLQJ--xmZHseFDLz/view
    """
    def __init__(self, root='data', **kwargs):
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'rgb/train')
        self.val_dir = osp.join(self.dataset_dir, 'rgb/val')
        self.test_dir = osp.join(self.dataset_dir, 'rgb/test')
        self._check_before_run()

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        val, num_val_pids, num_val_imgs, num_val_clothes, _ = \
            self._process_dir_train(self.val_dir)

        query_same, query_diff, gallery, num_test_pids, \
            num_query_imgs_same, num_query_imgs_diff, num_gallery_imgs, \
            num_test_clothes, gallery_idx = self._process_dir_test(self.test_dir)

        num_total_pids = num_train_pids + num_test_pids
        num_test_imgs = num_query_imgs_same + num_query_imgs_diff + num_gallery_imgs
        num_total_imgs = num_train_imgs + num_val_imgs + num_test_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        logger = logging.getLogger('reid.dataset')
        logger.info("=> PRCC loaded")
        logger.info("Dataset statistics:")
        logger.info("  --------------------------------------------")
        logger.info("  subset      | # ids | # images | # clothes")
        logger.info("  --------------------------------------------")
        logger.info("  train       | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  val         | {:5d} | {:8d} | {:9d}".format(num_val_pids, num_val_imgs, num_val_clothes))
        logger.info("  test        | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        logger.info("  query(same) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_same))
        logger.info("  query(diff) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_diff))
        logger.info("  gallery     | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  --------------------------------------------")
        logger.info("  total       | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  --------------------------------------------")

        logger.info("  --------------------------------------------------------------- ")
        logger.info("   Partition |  <32  | '32-64' | '64-128' | '128-256' | '>256' ")
        logger.info("    train    |     0 |    125  |   15101  |    2597   |     73 ")
        logger.info("    val      |     0 |     20  |    4359  |     622   |      1 ") 
        logger.info("    test     |     0 |     23  |    8510  |    2247   |     20 ") 
        logger.info("  --------------------------------------------------------------- ")

        self.train = train
        self.val = val
        self.query_same = query_same
        self.query_diff = query_diff
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes
        self.gallery_idx = gallery_idx

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir_train(self, dir_path):
        pdirs = glob.glob(osp.join(dir_path, '*'))
        pdirs.sort()

        pid_container = set()
        clothes_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0] # 'A' or 'B' or 'C'
                if cam in ['A', 'B']:
                    clothes_container.add(osp.basename(pdir))
                else:
                    clothes_container.add(osp.basename(pdir)+osp.basename(img_dir)[0])
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0] # 'A' or 'B' or 'C'
                label = pid2label[pid]
                camid = cam2label[cam]
                if cam in ['A', 'B']:
                    clothes_id = clothes2label[osp.basename(pdir)]
                else:
                    clothes_id = clothes2label[osp.basename(pdir)+osp.basename(img_dir)[0]]
                dataset.append((img_dir, label, camid, clothes_id))
                pid2clothes[label, clothes_id] = 1            
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes

    def _process_dir_test(self, test_path):
        pdirs = glob.glob(osp.join(test_path, '*'))
        pdirs.sort()

        pid_container = set()
        for pdir in glob.glob(osp.join(test_path, 'A', '*')):
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = num_pids * 2

        query_dataset_same_clothes = []
        query_dataset_diff_clothes = []
        gallery_dataset = []
        for cam in ['A', 'B', 'C']:
            pdirs = glob.glob(osp.join(test_path, cam, '*'))
            for pdir in pdirs:
                pid = int(osp.basename(pdir))
                img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
                for img_dir in img_dirs:
                    # pid = pid2label[pid]
                    camid = cam2label[cam]
                    if cam == 'A':
                        clothes_id = pid2label[pid] * 2
                        gallery_dataset.append((img_dir, pid, camid, clothes_id))
                    elif cam == 'B':
                        clothes_id = pid2label[pid] * 2
                        query_dataset_same_clothes.append((img_dir, pid, camid, clothes_id))
                    else:
                        clothes_id = pid2label[pid] * 2 + 1
                        query_dataset_diff_clothes.append((img_dir, pid, camid, clothes_id))

        pid2imgidx = {}
        for idx, (img_dir, pid, camid, clothes_id) in enumerate(gallery_dataset):
            if pid not in pid2imgidx:
                pid2imgidx[pid] = []
            pid2imgidx[pid].append(idx)

        # get 10 gallery index to perform single-shot test
        gallery_idx = {}
        random.seed(3)
        for idx in range(0, 10):
            gallery_idx[idx] = []
            for pid in pid2imgidx:
                gallery_idx[idx].append(random.choice(pid2imgidx[pid]))
                 
        num_imgs_query_same = len(query_dataset_same_clothes)
        num_imgs_query_diff = len(query_dataset_diff_clothes)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset_same_clothes, query_dataset_diff_clothes, gallery_dataset, \
               num_pids, num_imgs_query_same, num_imgs_query_diff, num_imgs_gallery, \
               num_clothes, gallery_idx



class PRCC_custom_colors(PRCC):
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
        for pid in range( self.num_train_pids ):
            start_index = self.original_num_clothes + pid * label
            end_index = self.original_num_clothes + (pid + 1) * label
            pid2clothes[pid, start_index : end_index ] = 1
        
        pid2clothes[:, :self.original_num_clothes] = self.pid2clothes
        self.pid2clothes = pid2clothes
        # self.pid2clothes[0,:256 + 81] 

    
        
class PRCC_CC_gender(PRCC_custom_colors):
    def __init__(self, root='data', gender_file=None, **kwargs):
        df = pd.read_csv(gender_file)
        self.gender_mapping = {i[1]:i[2] for i in df.values}
        
        super().__init__(root=root, **kwargs)
        
    def _process_dir_train(self, dir_path):
        pdirs = glob.glob(osp.join(dir_path, '*'))
        pdirs.sort()

        pid_container = set()
        clothes_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0] # 'A' or 'B' or 'C'
                if cam in ['A', 'B']:
                    clothes_container.add(osp.basename(pdir))
                else:
                    clothes_container.add(osp.basename(pdir)+osp.basename(img_dir)[0])
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            gender = self.gender_mapping[pid]
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0] # 'A' or 'B' or 'C'
                label = pid2label[pid]
                camid = cam2label[cam]
                if cam in ['A', 'B']:
                    clothes_id = clothes2label[osp.basename(pdir)]
                else:
                    clothes_id = clothes2label[osp.basename(pdir)+osp.basename(img_dir)[0]]
                dataset.append((img_dir, label, camid, clothes_id, gender))
                pid2clothes[label, clothes_id] = 1            
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes
