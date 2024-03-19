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

class DeepChange(object):
    """ DeepChange

    Reference:
        Xu et al. DeepChange: A Long-Term Person Re-Identification Benchmark. arXiv:2105.14685, 2021.

    URL: https://github.com/PengBoXiangShang/deepchange
    """
    dataset_dir = 'DeepChangeDataset'
    def __init__(self, root='data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train-set')
        self.train_list = osp.join(self.dataset_dir, 'train-set-bbox.txt')
        self.val_query_dir = osp.join(self.dataset_dir, 'val-set-query')
        self.val_query_list = osp.join(self.dataset_dir, 'val-set-query-bbox.txt')
        self.val_gallery_dir = osp.join(self.dataset_dir, 'val-set-gallery')
        self.val_gallery_list = osp.join(self.dataset_dir, 'val-set-gallery-bbox.txt')
        self.test_query_dir = osp.join(self.dataset_dir, 'test-set-query')
        self.test_query_list = osp.join(self.dataset_dir, 'test-set-query-bbox.txt')
        self.test_gallery_dir = osp.join(self.dataset_dir, 'test-set-gallery')
        self.test_gallery_list = osp.join(self.dataset_dir, 'test-set-gallery-bbox.txt')
        self._check_before_run()

        train_names = self._get_names(self.train_list)
        val_query_names = self._get_names(self.val_query_list)
        val_gallery_names = self._get_names(self.val_gallery_list)
        test_query_names = self._get_names(self.test_query_list)
        test_gallery_names = self._get_names(self.test_gallery_list)

        pid2label, clothes2label, pid2clothes = self.get_pid2label_and_clothes2label(train_names)
        train, num_train_pids, num_train_clothes = self._process_dir(self.train_dir, train_names, clothes2label, pid2label=pid2label, train=True)

        pid2label, clothes2label = self.get_pid2label_and_clothes2label(val_query_names, val_gallery_names)
        val_query, num_val_query_pids, num_val_query_clothes  = self._process_dir(self.val_query_dir, val_query_names, clothes2label)
        val_gallery, num_val_gallery_pids, num_val_gallery_clothes = self._process_dir(self.val_gallery_dir, val_gallery_names, clothes2label)
        num_val_pids = len(pid2label)
        num_val_clothes = len(clothes2label)

        pid2label, clothes2label = self.get_pid2label_and_clothes2label(test_query_names, test_gallery_names)
        test_query, num_test_query_pids, num_test_query_clothes = self._process_dir(self.test_query_dir, test_query_names, clothes2label)
        test_gallery, num_test_gallery_pids, num_test_gallery_clothes = self._process_dir(self.test_gallery_dir, test_gallery_names, clothes2label)
        num_test_pids = len(pid2label)
        num_test_clothes = len(clothes2label)

        num_total_pids = num_train_pids + num_val_pids + num_test_pids
        num_total_clothes = num_train_clothes + num_val_clothes + num_test_clothes
        num_total_imgs = len(train) + len(val_query) + len(val_gallery) + len(test_query) + len(test_gallery)

        logger = logging.getLogger('reid.dataset')
        logger.info("=> DeepChange loaded")
        logger.info("Dataset statistics:")
        logger.info("  --------------------------------------------")
        logger.info("  subset        | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train         | {:5d} | {:8d} | {:9d} ".format(num_train_pids, len(train), num_train_clothes))
        logger.info("  query(val)    | {:5d} | {:8d} | {:9d} ".format(num_val_query_pids, len(val_query), num_val_query_clothes))
        logger.info("  gallery(val)  | {:5d} | {:8d} | {:9d} ".format(num_val_gallery_pids, len(val_gallery), num_val_gallery_clothes))
        logger.info("  query         | {:5d} | {:8d} | {:9d} ".format(num_test_query_pids, len(test_query), num_test_query_clothes))
        logger.info("  gallery       | {:5d} | {:8d} | {:9d} ".format(num_test_gallery_pids, len(test_gallery), num_test_gallery_clothes))
        logger.info("  --------------------------------------------")
        logger.info("  total         | {:5d} | {:8d} | {:9d} ".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  --------------------------------------------")

        logger.info("  --------------------------------------------------------------------")
        logger.info("  Partition        |  <32  | '32-64' | '64-128' | '128-256' | '>256'  ")
        logger.info("  train-set        |     0 |      0  |   75083  |       0   |      0  ")
        logger.info("  test-set-query   |     0 |      0  |   17527  |       0   |      0  ")
        logger.info("  test-set-gallery |     0 |      0  |   62956  |       0   |      0  ")
        logger.info("  val-set-gallery  |     0 |      0  |   17865  |       0   |      0  ")
        logger.info("  val-set-query    |     0 |      0  |    4976  |       0   |      0  ")
        logger.info("  --------------------------------------------------------------------")


        self.train = train
        self.val_query = val_query
        self.val_gallery = val_gallery
        self.query = test_query
        self.gallery = test_gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def get_pid2label_and_clothes2label(self, img_names1, img_names2=None):
        if img_names2 is not None:
            img_names = img_names1 + img_names2
        else:
            img_names = img_names1

        pid_container = set()
        clothes_container = set()
        for img_name in img_names:
            names = img_name.split('.')[0].split('_')
            clothes = names[0] + names[2]
            pid = int(names[0][1:])
            pid_container.add(pid)
            clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

        if img_names2 is not None:
            return pid2label, clothes2label

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_name in img_names:
            names = img_name.split('.')[0].split('_')
            clothes = names[0] + names[2]
            pid = int(names[0][1:])
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            pid2clothes[pid, clothes_id] = 1

        return pid2label, clothes2label, pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_query_dir):
            raise RuntimeError("'{}' is not available".format(self.val_query_dir))
        if not osp.exists(self.val_gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.val_gallery_dir))
        if not osp.exists(self.test_query_dir):
            raise RuntimeError("'{}' is not available".format(self.test_query_dir))
        if not osp.exists(self.test_gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.test_gallery_dir))

    def _process_dir(self, home_dir, img_names, clothes2label, pid2label=None, train=None):
        dataset = []
        pid_container = set()
        clothes_container = set()
        for img_name in img_names:
            img_path = osp.join(home_dir, img_name.split(',')[0])
            names = img_name.split('.')[0].split('_')
            tracklet_id = int(img_name.split(',')[1])
            clothes = names[0] + names[2]
            clothes_id = clothes2label[clothes]
            clothes_container.add(clothes_id)
            pid = int(names[0][1:])
            pid_container.add(pid)
            camid = int(names[1][1:])
            if pid2label is not None:
                pid = pid2label[pid]
            # on DeepChange, we allow the true matches coming from the same camera 
            # but different tracklets as query following the original paper.
            # So we use tracklet_id to replace camid for each sample.
            dataset.append((img_path, pid, tracklet_id, clothes_id))
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        return dataset, num_pids, num_clothes

class DeepChange_custom_colors(DeepChange):
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
        

class DeepChange_CC_gender(DeepChange_custom_colors):
    def __init__(self, root='data', gender_file=None, **kwargs):
        df = pd.read_csv(gender_file)
        self.gender_mapping = {i[1]:i[2] for i in df.values}
        
        super().__init__(root=root, **kwargs)
        
    def _process_dir(self, home_dir, img_names, clothes2label, pid2label=None, train=None):
        dataset = []
        pid_container = set()
        clothes_container = set()
        # sorted(set([img_name.split('.')[0].split('_')[0] for img_name in img_names]))
        for img_name in img_names:
            img_path = osp.join(home_dir, img_name.split(',')[0])
            names = img_name.split('.')[0].split('_')
            tracklet_id = int(img_name.split(',')[1])
            clothes = names[0] + names[2]
            clothes_id = clothes2label[clothes]
            clothes_container.add(clothes_id)
            pid = int(names[0][1:])
            pid_container.add(pid)
            camid = int(names[1][1:])
            if pid2label is not None:
                pid = pid2label[pid]
            # on DeepChange, we allow the true matches coming from the same camera 
            # but different tracklets as query following the original paper.
            # So we use tracklet_id to replace camid for each sample.
            if train:
                gender = self.gender_mapping[names[0]]
                dataset.append((img_path, pid, tracklet_id, clothes_id, gender))
            else:
                dataset.append((img_path, pid, tracklet_id, clothes_id))

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        return dataset, num_pids, num_clothes
