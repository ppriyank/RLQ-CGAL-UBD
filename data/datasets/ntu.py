import os
import glob
import random
import logging
import numpy as np
import os.path as osp
import pandas as pd 
from collections import defaultdict

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise



class NTU(object):
    def __init__(self, root='data', pid_separator = 60, **kwargs):
        self.root = root
        
        split_file = '~/RLQ-CGAL-UBD/Scripts/Helper/ntu_subset.csv'
        split = pd.read_csv(split_file, names=["names"])

        file_names = split.names.tolist()
        pid2label, clothes2label, pid2clothes, cam2label, action2label = self.get_pid2label_and_clothes2label(file_names, pid_separator=pid_separator)
        
        train, num_train_pids, num_train_clothes = self._process_dir(file_names, clothes2label, action2label=action2label, cam2label=cam2label, pid2label=pid2label, mode='train', pid_separator=pid_separator)

        all_test, all_num_test_pids, all_num_test_clothes  = self._process_dir(file_names, clothes2label, action2label=action2label, cam2label=cam2label, pid2label=pid2label, mode='all_test')
        
        test_query, num_test_query_pids, num_test_query_clothes  = self._process_dir(file_names, clothes2label, action2label=action2label, cam2label=cam2label, pid2label=pid2label, mode='query', pid_separator=pid_separator)
        test_gallery, num_test_gallery_pids, num_test_gallery_clothes  = self._process_dir(file_names, clothes2label, action2label=action2label, cam2label=cam2label, pid2label=pid2label, mode='gallery', pid_separator=pid_separator)

        # self.various_cloth_id(train) 
        # self.various_cloth_id(all_test) 
        # self.various_test_actions(all_test, sampling=True )
        # self.various_test_actions(test_query, sampling=None)
        # self.various_test_actions(test_gallery, sampling=None)

        num_total_pids = num_train_pids + all_num_test_pids
        num_total_clothes = num_train_clothes + all_num_test_clothes
        num_total_imgs = len(train) + len(test_query) + len(test_gallery)

        logger = logging.getLogger('reid.dataset')
        logger.info("=> NTU loaded")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, len(train), num_train_clothes))
        logger.info("  query    | {:5d} | {:8d} | {:9d}".format(all_num_test_pids, len(test_query), num_test_query_clothes))
        logger.info("  gallery  | {:5d} | {:8d} | {:9d}".format(all_num_test_pids, len(test_gallery), num_test_gallery_clothes))
        logger.info("  ----------------------------------------")
        logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  ----------------------------------------")

        self.clothes_dict = None
        self.train = train
        self.query = test_query
        self.gallery = test_gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes
        
        self.original_num_clothes = self.num_train_clothes 

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def get_pid2label_and_clothes2label(self, img_names, pid_separator=60):
        
        pid_container = set()
        clothes_container = set()
        cam_container = set()
        action_container = set()
        train_no_clothes = set()

        for img_name in img_names:
            pid = img_name.split("/")[-4]
            pid_container.add(pid)

            # clothes = "_".join(img_name.split("/")[-4:-1])
            # clothes = img_name.split("/")[-4] + "_" +  img_name.split("/")[-2]
            clothes = img_name.split("/")[-4] + "_" +  img_name.split("/")[-1][:4]
            clothes_container.add(clothes)

            cam_id = img_name.split("/")[-2]
            cam_container.add(cam_id)

            action_id= img_name.split("/")[-3]
            action_container.add(action_id)
            
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        cam_container = sorted(cam_container)
        action_container = sorted(action_container)
        
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        cam2label = {cam: label for label, cam in enumerate(cam_container)}
        action2label = {action: label for label, action in enumerate(action_container)}
        clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

        for img_name in img_names:
            pid = img_name.split("/")[-4]
            pid = pid2label[pid]
            clothes = img_name.split("/")[-4] + "_" +  img_name.split("/")[-1][:4]
            if pid <= pid_separator:
                train_no_clothes.add(clothes)

        pid2clothes = np.zeros((pid_separator + 1, len(train_no_clothes)))
        for img_name in img_names:
            pid = img_name.split("/")[-4]
            clothes = img_name.split("/")[-4] + "_" +  img_name.split("/")[-1][:4]

            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]

            if pid > pid_separator:
                continue 
            pid2clothes[pid, clothes_id] = 1

        return pid2label, clothes2label, pid2clothes, cam2label, action2label

    def various_cloth_id(self, train):
        home_directory = os.path.expanduser('~')
        sample_fodler = os.path.join(home_directory, 'RLQ-CGAL-UBD', 'Samples')
        os.system(f'rm -rf {sample_fodler}/' )

        cloth_imgs = defaultdict(list)

        for ele in train: cloth_imgs[ele[3]].append(ele[0])
        
        for key in cloth_imgs: 
            if len(cloth_imgs[key]) > 1:
                dest_folder = os.path.join(home_directory, 'RLQ-CGAL-UBD', 'Samples',str(key))
                mkdir_if_missing(dest_folder)
                [os.system(f'cp {e} {dest_folder}' ) for e in cloth_imgs[key]]
                # img_name = cloth_imgs[key][0]

    def various_test_actions(self, all_test, sampling=None):
        actions_imgs = defaultdict(list)
        for ele in all_test: actions_imgs[ele[1]].append(ele[2]) 

        if sampling:
            for key in actions_imgs: print(key, actions_imgs[key][::2])
            for key in actions_imgs: print(key, actions_imgs[key][1::2])
        else:
            for key in actions_imgs: print(key, actions_imgs[key])

    def _process_dir(self, img_names, clothes2label, action2label=None, cam2label=None, pid2label=None, mode=None, pid_separator=60):
        dataset = []
        pid_container = set()
        clothes_container = set()
        for img_name in img_names:
            pid = img_name.split("/")[-4]
            if pid2label is not None:
                pid = pid2label[pid]

            action_id= img_name.split("/")[-3]
            action_id = action2label[action_id]
            
            if mode == 'train':
                if pid > pid_separator:
                    continue 
            else:
                if pid <= pid_separator:
                    continue
                if mode == "query" and action_id > 70:
                    continue 
                if mode == "gallery" and action_id <= 70:
                    continue 

            pid_container.add(pid)
            # clothes = "_".join(img_name.split("/")[-4:-1])
            clothes = img_name.split("/")[-4] + "_" +  img_name.split("/")[-1][:4]
            clothes_id = clothes2label[clothes]
            clothes_container.add(clothes_id)

            cam_id = img_name.split("/")[-2]
            cam_id = cam2label[cam_id]

            img_path = osp.join(self.root, "/".join(img_name.split('/')[-4:]) )
            assert osp.exists(img_path)
            
            dataset.append((img_path, pid, action_id, clothes_id))
            
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        return dataset, num_pids, num_clothes

class NTU_custom_colors(NTU):
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
        




if __name__ == "__main__":
    dataset = NTU(root="/data/priyank/synthetic/NTU/RGB/")
    snippet = dataset.train[1919]
    
    img_path = snippet[0]
    sil_path = img_path.replace('.mp4', '_sil.png' ).replace('RGB', 'Mask' )
    
    
# cd ~/RLQ-CGAL-UBD/
# python data/datasets/ntu.py