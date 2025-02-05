# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import logging
import os.path as osp
import numpy as np

import os 
from collections import defaultdict
import pandas as pd 
import random 

def make_folder(name):
    try: 
        os.mkdir(name) 
    except OSError as error: 
        _ = 0 
    return 



class Market1501(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

        self.pid2clothes = np.eye(self.num_train_pids)
        self.num_train_clothes = self.num_train_pids
        # self.gender_csv(train)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        num_test_pids = num_gallery_pids
        logger = logging.getLogger('reid.dataset')
        logger.info("=> Market1501 loaded")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_pids))
        logger.info("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_query_imgs+num_gallery_imgs, num_test_pids))
        logger.info("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
        logger.info("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  ----------------------------------------")
        logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_train_pids+num_test_pids, num_train_imgs+num_query_imgs+num_gallery_imgs, num_train_pids+num_test_pids))
        logger.info("  ----------------------------------------")

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views
        
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            if relabel:
                dataset.append((img_path, self.pid_begin + pid, camid, pid))
            else:
                dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset
    
    def gender_csv(self, train):
        pid_to_path = defaultdict(list)
        images_database = []
        for e in train:
            pid_to_path[ e[1] ].append( e[0] ) 
            
        src = "Dump/"
        make_folder(src)
        for person in pid_to_path:
            dest = f"{src}/{person}"
            make_folder(dest)
            selected = random.choices(pid_to_path[ person ], k = 2)
            for img_path in selected:
                os.system(f'cp {img_path} {dest}')
            images_database.append(["train", person, "0"])

        df = pd.DataFrame(images_database, columns = ["Category","ID","Gender"]) 
        assert os.path.exists(f"Scripts/Helper/market_Gender.csv") == False, "Market Gender csv will be overwritten!!"
        df.to_csv(f"Scripts/Helper/market_Gender.csv", index=False,)  

       
class Market1501_CC_gender(Market1501):
    def __init__(self, root='data', gender_file=None, **kwargs):
        
        df = pd.read_csv(gender_file)
        self.gender_mapping = {i[1]:i[2] for i in df.values}
        
        super().__init__(root=root, **kwargs)
        
    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for e in data:
            pid, camid, trackid = e[1], e[2], e[3]
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views
        
    
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: 
                pid = pid2label[pid]
                gender = self.gender_mapping[pid]

            if relabel:
                dataset.append((img_path, self.pid_begin + pid, camid, pid, gender))
            else:
                dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset
    
    