import torch
from einops import rearrange
import functools
import os.path as osp
import copy 
from torch.utils.data import Dataset
from .utils import * 
    
import cv2
from PIL import Image

import numpy as np
import os 
import random 
from torchvision.utils import save_image
import torchvision.transforms as transforms
from .img_transforms import RandomErasing_colors
from collections import defaultdict 
import pandas as pd
from tools.utils import load_pickle

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, train=True, illumination=None, return_index=None, splits=None, dataset_name=None, 
        load_as_video=None, **kwargs):
        self.dataset = dataset
        self.transform = transform
        self.illumination = illumination
        self.train = train
        self.return_index = return_index
        self.splits = splits
        self.dataset_name = dataset_name

        self.load_as_video = load_as_video
        # self.__getitem__(0)
        # video_without_img_paths

    def __len__(self):
        return len(self.dataset)

    def split_input(self, img, mode="uniform"):
        img = np.array(img)
        H, W = img.shape[:2]
        if mode == "uniform":            
            splits_height = H // self.splits
            splits = []
            for i in range(self.splits):
                mask = img[i * splits_height : i * splits_height + splits_height]
                mask = Image.fromarray(mask) 
                mask = self.transform(mask)
                splits.append(mask)
            splits = torch.stack(splits)
            # save_image(normalize(splits), "temp.png")
            return splits

    def read_vid_as_image(self, img_path):        
        sil_path = img_path.replace('.mp4', '_sil.png' ).replace('RGB', 'Mask' )
        sil = read_image(sil_path)    
        sil = np.array(sil)        
        
        video = video_without_img_paths(img_path)
        img = video[len(video) // 2].asnumpy()
        sil, img = crop_img(sil, img, padding=10)

        img = Image.fromarray(img) 
        # img.save("temp.png")
        return img
            
    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img = read_image(img_path, self.illumination)
        if self.splits:
            h_imgs = self.split_input(img)
            if self.transform is not None:
                img = self.transform(img)
            if self.return_index:
                return img, pid, camid, clothes_id, h_imgs, "/".join(img_path.split("/")[-2:])
            return img, pid, camid, clothes_id, h_imgs
        else:
            if self.transform is not None:
                img = self.transform(img)
            
        if self.return_index:
            return img, pid, camid, clothes_id, "/".join(img_path.split("/")[-2:])
        return img, pid, camid, clothes_id

    def set_epoch(self, epoch):
        self.epoch = epoch
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageDataset_w_sil(ImageDataset):
    """Image Person ReID Dataset"""
    def __init__(self, bkd_folder=None, silhouettes=None, sil_mode=None, clothes_dict=None, 
        original_clothes=-1, sil_transform_train=None, sil_transform_test=None, splits=None, dataset_name=None,
        **kwargs):
        super().__init__(dataset_name=dataset_name, **kwargs)
        self.dataset_setup(dataset_name)
        self.epoch = 0
        self.bkd_folder = [ ]
        if bkd_folder:
            if os.path.exists(bkd_folder):        
                for e in os.listdir(bkd_folder):
                    bkd = os.path.join(bkd_folder, e)
                    bkd = Image.open(bkd).convert('RGB')
                    self.bkd_folder.append(bkd)
                        
        self.silhouettes = silhouettes
        self.clothes_dict = clothes_dict
        self.original_clothes = original_clothes 
        self.sil_transform_train = sil_transform_train
        self.sil_transform_test = sil_transform_test
        self.splits = splits
        
        if self.clothes_dict:
           self.colors = self.clothes_dict.keys()
           self.colors = sorted(list(set([e[0] for e in self.colors])))
           self.color_to_rgb = {e : colorname_to_rgb(e) for e in self.colors}
           self.offset = (len(self.colors) * len(self.colors))
            # for color in self.color_to_rgb:pant = create_image(shirt_sil.shape[0], shirt_sil.shape[1], self.color_to_rgb[color]);Image.fromarray(pant).save(f'temp_{color}.png')
        if sil_mode is None:
            # self.sil_mode = "background_overlap"
            self.sil_mode = None
        else:
            self.sil_mode = sil_mode
        if self.sil_mode == "foreground_overlap_patch_w_sil":
            self.cropping = RandomErasing_colors()

        self.dataset = sorted(self.dataset)
        print(self.transform)
        # self.__getitem__(0)
    
    def dataset_setup(self, dataset_name):
        if self.train:self.category = "train"
        if "ltcc" in dataset_name :
            pant_faulty = "Scripts/Helper/LTCC_Pant_faulty.csv"
            shirt_faulty = "Scripts/Helper/LTCC_Shirt_faulty.csv"
            self.load_indentifier = self.simple_indentifer
        elif "prcc" in dataset_name :
            pant_faulty = "Scripts/Helper/PRCC_Pant_faulty.csv"
            shirt_faulty = "Scripts/Helper/PRCC_Shirt_faulty.csv"
            self.load_indentifier = self.prcc_indentifier
        elif "celeb" in dataset_name :
            pant_faulty = "Scripts/Helper/Celeb-reID_Pant_faulty.csv"
            shirt_faulty = "Scripts/Helper/Celeb-reID_Shirt_faulty.csv"
            self.load_indentifier = self.simple_indentifer
        elif "deepchange" in dataset_name :
            pant_faulty = "Scripts/Helper/DeepChange_Pant_faulty.csv"
            shirt_faulty = "Scripts/Helper/DeepChange_Shirt_faulty.csv"
            self.load_indentifier = self.simple_indentifer
            if self.train:self.category = "train-set"
        elif "last" in dataset_name :
            pant_faulty = "Scripts/Helper/LaST_Pant_faulty.csv"
            shirt_faulty = "Scripts/Helper/LaST_Shirt_faulty.csv"
            self.load_indentifier = self.prcc_indentifier
        elif 'ntu' in dataset_name:
            self.category = ''
            self.load_indentifier = self.ntu_indentifier
            self.faulty_sil = set()
            return 
        elif 'market' in dataset_name:
            self.category = ''
            self.load_indentifier = self.simple_indentifer
            self.faulty_sil = set()
            return 
        else:
            assert False, f"Dataset {dataset_name} not found for clothes"

        df = pd.read_csv(pant_faulty, names=["images"])
        faulty_sil  = set(df.images)
        df = pd.read_csv(shirt_faulty, names=["images"])
        faulty_sil = faulty_sil.union(set(df.images))
        self.faulty_sil = faulty_sil

    def ntu_indentifier(self, img):
        return "/".join(img.split("/")[-4:])[:-4]

    def simple_indentifer(self, img):
        indentifier = img.split("/")[-1][:-4]
        return indentifier

    def prcc_indentifier(self, img):
        id, image_name = img.split("/")[-2:]
        identifier = id + "_" + image_name[:-4]
        return identifier 
        # identifier = f"{session}_{folder}_{image_name[:-4]}.png"
            
    def load_sil(self, sil):
        sil = read_image(sil)
        sil = np.array(sil)
        sil[sil != 0 ] = 1
        return sil 

    def foreground_overlap_create(self, img, clothes_id, indentifier, pid, full_clothes=False, return_w_sil=False, 
        test_mode=None, return_w_bkd=False, return_w_body=False, retain=False):
        pant_sil = os.path.join(self.silhouettes, self.category, f"{indentifier}_pant.png") 
        shirt_sil = os.path.join(self.silhouettes, self.category, f"{indentifier}_shirt.png") 
        body_sil = os.path.join(self.silhouettes, self.category, f"{indentifier}_sil.png") 
        if full_clothes:
            if random.random() > 0.5:
                pant_sil = os.path.join(self.silhouettes, self.category, f"{indentifier}_full_pant.png") 
            if random.random() > 0.5:
                shirt_sil = os.path.join(self.silhouettes, self.category, f"{indentifier}_full_shirt.png") 
        
        faulty = False
        if indentifier in self.faulty_sil:
            shirt_sil = os.path.join(self.silhouettes, self.category, f"{indentifier}_clothes.png") 
            pant_sil = shirt_sil
            faulty = True 
        
        if return_w_sil:
            # img, clothes_id, pant_sil, shirt_sil, body_sil = self.foreground_overlap_aug(img, pant_sil, shirt_sil, pid, clothes_id)
            # return img, clothes_id, pant_sil.transpose(2,0,1).astype(np.uint8), shirt_sil.transpose(2,0,1).astype(np.uint8), body_sil.transpose(2,0,1).astype(np.uint8) 
            img, clothes_id, src_patch, src_mask = self.foreground_overlap_aug(img, pant_sil, shirt_sil, body_sil, pid, clothes_id)
            return img, clothes_id, src_patch, src_mask.transpose(2,0,1).astype(np.uint8)
        elif return_w_body:
            img, clothes_id, sil = self.foreground_overlap_w_bkd(img, pant_sil, shirt_sil, body_sil, pid, clothes_id, test_mode=test_mode, faulty=faulty)
            return img, clothes_id, sil
        elif retain:
            pant_sil_full = os.path.join(self.silhouettes, category, f"{indentifier}_full_pant.png") 
            shirt_sil_full = os.path.join(self.silhouettes, category, f"{indentifier}_full_shirt.png") 
            if faulty:
                shirt_sil_full = os.path.join(self.silhouettes, category, f"{indentifier}_clothes.png") 
                pant_sil_full = shirt_sil_full
            img, clothes_id = self.foreground_full_half_overlap(img, pant_sil, shirt_sil, shirt_sil_full, pant_sil_full, pid, clothes_id, test_mode=test_mode, faulty=faulty)
        else:
            if return_w_bkd:
                img_bkd = self.background_overlap(img, body_sil, retain_original=False)
                img, clothes_id = self.foreground_overlap(img, pant_sil, shirt_sil, pid, clothes_id, test_mode=test_mode, faulty=faulty)
                img = torch.cat([img, img_bkd.unsqueeze(0)])
            else:
                img, clothes_id = self.foreground_overlap(img, pant_sil, shirt_sil, pid, clothes_id, test_mode=test_mode, faulty=faulty)
        return img, clothes_id

    def create_train_clothes(self, img, indentifier, clothes_id, pid):
        if self.sil_mode == "background_overlap":
            img = self.background_overlap_create(img, indentifier)
        elif self.sil_mode == "foreround_overlap":
            img, clothes_id = self.foreground_overlap_create(img, clothes_id, indentifier, pid)
        elif self.sil_mode == "fore_full_overlap":
            img, clothes_id = self.foreground_overlap_create(img, clothes_id, indentifier, pid, full_clothes=True)
            # save_image(normalize(img), "temp.png")
        elif self.sil_mode == "foreround_overlap_bkd":
            img_bkd = self.background_overlap_create(img, indentifier, retain_original=False)
            img, clothes_id = self.foreground_overlap_create(img, clothes_id, indentifier, pid)
            img = torch.cat([img, img_bkd.unsqueeze(0)])
            clothes_id = torch.cat([clothes_id, clothes_id[0].unsqueeze(0)])
        elif self.sil_mode == "foreround_aug_w_sil":
            assert False, "Outdated, re-implement "
        elif self.sil_mode == "foreground_overlap_w_sil":
            img, clothes_id, sil = self.foreground_overlap_create(img, clothes_id, indentifier, pid, return_w_body=True)
            return img, clothes_id, sil
        elif self.sil_mode == "fore_full_half_overlap":
            img, clothes_id = self.foreground_overlap_create(img, clothes_id, indentifier, pid, retain=True)
            # save_image(normalize(img), "temp.png")
        elif self.sil_mode is None:
            img = self.transform(img)
        return img, clothes_id

    def foreground_overlap(self, img, pant_sil, shirt_sil, pid=-1, clothes_id=-1, alpha =0.70, test_mode=None, faulty=None, indentifier=None):
        pant_sil = self.load_sil(pant_sil)
        # Image.fromarray(pant_sil * 255).save("temp.png")
        shirt_sil = self.load_sil(shirt_sil)
        # Image.fromarray(shirt_sil * 255).save("temp2.png")
        if test_mode:
            labels = []
            imgs = []
            for color in test_mode:
                masked_image, assigned_label = self.change_clothes_color_test(shirt_sil, pant_sil, img, alpha, color=color, faulty=faulty)
                masked_image = Image.fromarray(masked_image)
                # masked_image.save("temp.png")
                effective_label = self.original_clothes + pid * self.offset + assigned_label    
                labels.append(effective_label)
                if self.transform is not None:
                    masked_image = self.transform(masked_image)
                imgs.append(masked_image)
            if self.transform is not None:
                img = self.transform(img)
            img = torch.stack([img] + imgs)   
            effective_label = torch.tensor([clothes_id] + labels)
        else:
            masked_image, assigned_label = self.change_clothes_color(shirt_sil, pant_sil, img, alpha, faulty)
            # print(f'Active CUDA Device: {dist.get_rank()} COLOR: {assigned_label} NAME: {indentifier}' )
            masked_image = Image.fromarray(masked_image)
            # masked_image.save("temp.png")
            effective_label = self.original_clothes + pid * self.offset + assigned_label
            effective_label = torch.tensor([clothes_id, effective_label])
        
            if self.transform is not None:
                img = self.transform(img)
                masked_image = self.transform(masked_image)
                img = torch.stack([img, masked_image])
        return img, effective_label

    def change_clothes_color(self, shirt_sil, pant_sil, img, alpha, faulty=None, shirt_color=None, pant_color=None):
        shirt_color = random.choice(self.colors)
        shirt = create_image(shirt_sil.shape[0], shirt_sil.shape[1], self.color_to_rgb[shirt_color])

        img = np.array(img)
        if faulty:
            assigned_label = self.clothes_dict[(shirt_color, shirt_color)]
            sil = shirt_sil
            masked_image = img * (1 - sil) + img * sil * (1 - alpha) + alpha * (shirt_sil * shirt)
        else:
            pant_color = random.choice(self.colors)
            assigned_label = self.clothes_dict[(pant_color, shirt_color)]
            pant = create_image(pant_sil.shape[0], shirt_sil.shape[1], self.color_to_rgb[pant_color])
            # Image.fromarray(pant * 255).save('temp1.png'), Image.fromarray(shirt * 255).save('temp2.png')
            sil = pant_sil | shirt_sil
        
            masked_image = img * (1 - sil) + img * sil * (1 - alpha) \
                + alpha * ( (pant_sil * pant) + (shirt_sil * shirt) )
        return masked_image.astype(np.uint8) , assigned_label
    
    def __getitem__(self, index):
        try:
            img_path, pid, camid, clothes_id = self.dataset[index]
            if self.load_as_video:
                img = self.read_vid_as_image(img_path)
            else:
                img = read_image(img_path, self.illumination)            
            indentifier = self.load_indentifier(img_path)
            if self.train:
                if self.sil_mode == "foreground_overlap_patch_w_sil":
                    img, clothes_id, src_patch, src_mask = self.foreground_overlap_create(img, clothes_id, indentifier, pid, return_w_sil=True )
                    return img, pid, camid, clothes_id, src_patch, src_mask
                elif self.sil_mode == "foreground_overlap_w_sil":
                    img, clothes_id , body_sil = self.create_train_clothes(img, indentifier, clothes_id, pid)
                    return img, pid, camid, clothes_id, body_sil
                else:
                    img, clothes_id = self.create_train_clothes(img, indentifier, clothes_id, pid)
            else:
                category = img_path.split("/")[-2]
                if "test" in self.sil_mode:
                    img = self.create_test_clothes(img, indentifier, clothes_id, pid, category)
                elif self.sil_mode == "foreground_overlap_w_sil":
                    img, sil = self.test_w_sil(img, indentifier, clothes_id, pid, category)
                    if self.return_index:
                        return img, pid, camid, clothes_id, "/".join(img_path.split("/")[-2:]), sil 
                    return img, pid, camid, clothes_id, sil 
                else:
                    img = self.transform(img)
            if self.return_index:
                return img, pid, camid, clothes_id, "/".join(img_path.split("/")[-2:])
            return img, pid, camid, clothes_id
        except Exception as e:
            print(index)
            print("****", index, e)
            quit()

    def set_epoch(self, epoch):
        self.epoch = epoch

def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
    k = k * ( 1.0 / np.sum(k) )        
    return cv2.filter2D(image, -1, k) 

class ImageDataset_w_res(ImageDataset):
    def __init__(self, low_res=(16,64), motion_blur=(8,20), motion_blur_angle=(0,180), g_blur=[4,22], **kwargs):
        super().__init__(**kwargs)

        self.low_res = low_res
        self.motion_blur = motion_blur
        self.motion_blur_angle = motion_blur_angle
        self.g_blur = g_blur 
        self.__getitem__(0)

    def create_low_res(self, img_hr):
        H,W = img_hr.size    
        ratio = random.choice(range(*self.low_res)) / min(H,W) 
        img_lr = img_hr.resize((round( H * ratio), round( W *ratio)))
        return img_lr

    def create_blur_motion(self, img_hr):
        blur_strength = random.choice(range(*self.motion_blur))
        blur_angle = random.choice(range(*self.motion_blur_angle))
        img_lr = apply_motion_blur(np.array(img_hr), blur_strength, blur_angle)
        img_lr = Image.fromarray( img_lr )
        return img_lr

    def create_g_blur(self, img_hr):
        blur_strength = random.choice(range(*self.g_blur, 2)) + 1
        img_lr = transforms.GaussianBlur(blur_strength)(img_hr)
        return img_lr

    def __getitem__(self, index):
        
        img_path, pid, camid, clothes_id = self.dataset[index]
        if self.load_as_video:
            img_hr = self.read_vid_as_image(img_path)
        else:
            img_hr = read_image(img_path, self.illumination)
        
        if random.random() > 0.5:
            img_lr = self.create_low_res(img_hr)
        else:
            img_lr = self.create_blur_motion(img_hr)
        
        # img_hr, img_lr
        # img_hr.save("temp.png"), img_lr.save("temp2.png")
        if self.transform is not None:
            img_hr = self.transform(img_hr)
            img_lr = self.transform(img_lr)
            
        return img_hr, pid, camid, clothes_id, img_lr

class ImageDataset_w_res_prcc(ImageDataset_w_res):
    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        if self.load_as_video:
            img_hr = self.read_vid_as_image(img_path)
        else:
            img_hr = read_image(img_path, self.illumination)
        if random.random() > 0.5:
            img_lr = self.create_low_res(img_hr)
        else:
            img_lr = self.create_g_blur(img_hr)
        
        # img_hr.save("temp.png"), img_lr.save("temp2.png")
        if self.transform is not None:
            img_hr = self.transform(img_hr)
            img_lr = self.transform(img_lr)
            
        return img_hr, pid, camid, clothes_id, img_lr

class ImageDataset_w_res_ltcc(ImageDataset_w_res):
    def __getitem__(self, index):
        try:
            img_path, pid, camid, clothes_id = self.dataset[index]
            if self.load_as_video:
                img_hr = self.read_vid_as_image(img_path)
            else:
                img_hr = read_image(img_path, self.illumination)
            
            
            flip_coin = random.random()
            # x = [(1 if random.random() < 0.33 else (2 if random.random() > 0.63 else 3)) for i in range(10000)]            
            # values, counts = np.unique(x, return_counts=True)
            if flip_coin < 0.33 :
                img_lr = self.create_low_res(img_hr)
                # img_lr.save("temp1.png")
            elif flip_coin > 0.63:
                img_lr = self.create_g_blur(img_hr)
                # img_lr.save("temp2.png")
            else:
                img_lr = self.create_blur_motion(img_hr)
                # img_lr.save("temp3.png")
            
            if self.transform is not None:
                img_hr = self.transform(img_hr)
                img_lr = self.transform(img_lr)
            # save_image(img_hr, "temp.png"), save_image(img_lr, "temp2.png")
            if self.return_index:
                return img_lr, pid, camid, clothes_id, "/".join(img_path.split("/")[-2:])
            
            return img_hr, pid, camid, clothes_id, img_lr
        
        except Exception as e:
            print("****", index, e)
            quit()


class ImageDataset_w_res_ONLY_LR(ImageDataset_w_res):
    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        if self.load_as_video:
            img_hr = self.read_vid_as_image(img_path)
        else:
            img_hr = read_image(img_path, self.illumination)
        img_lr = self.create_low_res(img_hr)
        if self.transform is not None:
            img_hr = self.transform(img_hr)
            img_lr = self.transform(img_lr)
        if self.return_index:
            return img_lr, pid, camid, clothes_id, "/".join(img_path.split("/")[-2:])
        return img_hr, pid, camid, clothes_id, img_lr

class ImageDataset_w_res_ONLY_MB(ImageDataset_w_res):
    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        if self.load_as_video:
            img_hr = self.read_vid_as_image(img_path)
        else:
            img_hr = read_image(img_path, self.illumination)
        img_lr = self.create_blur_motion(img_hr)
        if self.transform is not None:
            img_hr = self.transform(img_hr)
            img_lr = self.transform(img_lr)
        if self.return_index:
            return img_lr, pid, camid, clothes_id, "/".join(img_path.split("/")[-2:])
        return img_hr, pid, camid, clothes_id, img_lr

class ImageDataset_w_res_ONLY_OOF(ImageDataset_w_res):
    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        if self.load_as_video:
            img_hr = self.read_vid_as_image(img_path)
        else:
            img_hr = read_image(img_path, self.illumination)
        img_lr = self.create_g_blur(img_hr)
        if self.transform is not None:
            img_hr = self.transform(img_hr)
            img_lr = self.transform(img_lr)
        if self.return_index:
            return img_lr, pid, camid, clothes_id, "/".join(img_path.split("/")[-2:])
        return img_hr, pid, camid, clothes_id, img_lr

class ImageDataset_w_gender(ImageDataset_w_sil):
    def __init__(self, gender_file=None, use_adv_sample=None, n_classes=-1, dataset=None, train=True, 
        **kwargs ):
        self.use_adv_sample = use_adv_sample
        if train and use_adv_sample:
            self.setup_adv_samples(n_classes, dataset) 
        super().__init__(dataset=dataset, train=train, **kwargs)            

    def setup_adv_samples(self, n_classes, dataset):
        self.adv_samples = []
        for i in range(n_classes):
            valid = [i for i in range(i)] + [i for i in range(i+1, n_classes)]
            self.adv_samples.append(random.choice(valid))
        self.pids_to_index = {i:[] for i in range(n_classes)}
        for index, data in enumerate(dataset):
            _, pid, _, _, _= data
            self.pids_to_index[pid].append(index)

    def load_adv_samples(self, pid ):
        adv_pid = self.adv_samples[pid]
        adv_index = random.choice( self.pids_to_index[adv_pid] )
        adv_img_path, adv_pid, adv_camid, adv_clothes_id, adv_gender = self.dataset[adv_index]
        adv_img = read_image(adv_img_path, self.illumination)            
        adv_img = self.transform(adv_img)
        return adv_img, adv_clothes_id, adv_pid, adv_camid, adv_gender

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = None, None, None, None
        try:
            if self.train:
                img_path, pid, camid, clothes_id, gender = self.dataset[index]
                img = read_image(img_path, self.illumination)            
                indentifier = img_path.split("/")[-1][:-4]
                img, clothes_id = self.create_train_clothes(img, indentifier, clothes_id, pid)
                if self.use_pose_sample:
                    pose_label = int(self.pose_label[indentifier])       
                    if self.use_adv_sample:
                        assert False, "Adv sample not yet defined"
                    return img, pid, camid, clothes_id, gender, pose_label  
                elif self.use_adv_sample:
                    adv_img, adv_clothes_id, adv_pid, adv_camid, adv_gender = self.load_adv_samples(pid )
                    # save_image(normalize(adv_img), "adv_img.png")
                    return img, pid, camid, clothes_id, gender, adv_img, adv_clothes_id, adv_pid, adv_camid, adv_gender
                return img, pid, camid, clothes_id, gender
            else:
                img_path, pid, camid, clothes_id = self.dataset[index]
                img = read_image(img_path, self.illumination)            
                indentifier = img_path.split("/")[-1][:-4]
                category = img_path.split("/")[-2]
                img = self.transform(img)
                # save_image(normalize(img), "temp.png")
                if self.return_index:
                    return img, pid, camid, clothes_id, "/".join(img_path.split("/")[-2:])
                return img, pid, camid, clothes_id
        except Exception as e:
            print("****", index, e, img_path, pid, camid, clothes_id)
            if self.use_adv_sample:
                print(" --- " , self.adv_samples ) 
            quit()

class ImageDataset_w_pose(ImageDataset_w_sil):
    def __init__(self, pose_mode=None, pose_path=None, **kwargs ):
        POSE_config = "Scripts/Helper/Pose_Cluster.csv"
        df = pd.read_csv(POSE_config)

        self.pose_mode = pose_mode
        self.pose_label = {}
        if pose_mode in df.columns:
            for x ,y in zip(df.Image, df[pose_mode]):
                self.pose_label[x] = y
        elif pose_mode == "embed":
            self.pose_path = pose_path
        else: 
            assert False, "emebdding loading not yet implemented"
        
        super().__init__(**kwargs)

    def load_pose_pickle(self, pickle_path):
        if os.path.exists(pickle_path + ".pkl"):
            pickle_path = load_pickle( pickle_path )
            scores = pickle_path["yolo_scores"]
            index = scores.argmax()
            # (2048, 8, 6)
            feat = pickle_path["feat"][index]
        else:
            feat = np.zeros((2048, 8, 6))
        
        return feat
  
    def __getitem__(self, index):
        try:
            img_path, pid, camid, clothes_id = self.dataset[index]
            img = read_image(img_path, self.illumination)            
            h_imgs = None
            indentifier = img_path.split("/")[-1][:-4]
            if self.train:    
                if self.pose_mode == "embed":
                    pickle_path = os.path.join(self.pose_path, "train", indentifier)
                    pose_label = self.load_pose_pickle(pickle_path)
                else:     
                    pose_label = int(self.pose_label[indentifier])       
                
                img, clothes_id = self.create_train_clothes(img, indentifier, clothes_id, pid)
                return img, pid, camid, clothes_id, pose_label
            else:
                category = img_path.split("/")[-2]
                img = self.transform(img)
                # save_image(normalize(img), "temp.png")
                if self.return_index:
                    return img, pid, camid, clothes_id, "/".join(img_path.split("/")[-2:])
                return img, pid, camid, clothes_id
        except Exception as e:
            print("****", index, e)
            quit()

class ImageDataset_w_gender_Pose(ImageDataset_w_gender):
    def __init__(self, use_body_shape=None, body_mode=None, use_adv_sample=None, dataset_name=None, 
    use_pose_sample=None, pose_mode=None, pose_path=None, **kwargs ):
        self.use_body_shape = use_body_shape
        self.use_pose_sample = use_pose_sample
        if "ltcc" in dataset_name:
            self.setup_ltcc_dataset(body_mode=body_mode, use_body_shape=use_body_shape, use_pose_sample=use_pose_sample, pose_mode=pose_mode)
        elif "prcc" in dataset_name:
            self.setup_prcc_dataset(body_mode=body_mode, use_body_shape=use_body_shape, use_pose_sample=use_pose_sample, pose_mode=pose_mode)
        elif "deepchange" in dataset_name:
            self.setup_deepchange_dataset(body_mode=body_mode, use_body_shape=use_body_shape, use_pose_sample=use_pose_sample, pose_mode=pose_mode)
        elif "celeb" in dataset_name:
            self.setup_celeb_dataset(body_mode=body_mode, use_body_shape=use_body_shape, use_pose_sample=use_pose_sample, pose_mode=pose_mode)
        elif "last" in dataset_name:
            self.setup_last_dataset(body_mode=body_mode, use_body_shape=use_body_shape, use_pose_sample=use_pose_sample, pose_mode=pose_mode)
        elif "market" in dataset_name:
            self.setup_market_dataset(body_mode=body_mode, use_body_shape=use_body_shape, use_pose_sample=use_pose_sample, pose_mode=pose_mode)
        else:
            import pdb
            pdb.set_trace()
        if use_adv_sample:
            assert False, 'Not yet implemented for the class "ImageDataset_w_gender_Pose_Body"'
        super().__init__(use_adv_sample=use_adv_sample, dataset_name=dataset_name, **kwargs)            
    
    def setup_pose(self, pose_mode, POSE_config):
        df = pd.read_csv(POSE_config)
        self.pose_mode = pose_mode
        self.pose_label = {}
        if pose_mode in df.columns:
            for x ,y in zip(df.Image, df[pose_mode]):
                self.pose_label[x] = y
        else:
            assert False, f"{pose_mode} not recognized"

    def setup_ltcc_dataset(self, body_mode=None, use_body_shape=None, use_pose_sample=None, pose_mode=None):
        POSE_config = "Scripts/Helper/LTCC_Pose_Cluster.csv"
        SIL_config = "Scripts/Helper/LTCC_Sil_Cluster.csv"        
        if use_body_shape:
            self.setup_body_shape(body_mode, SIL_config)
        if use_pose_sample:
            self.setup_pose(pose_mode, POSE_config)    
        self.load_indentifier = self.simple_indentifer

    def setup_prcc_dataset(self, body_mode=None, use_body_shape=None, use_pose_sample=None, pose_mode=None):
        POSE_config = "Scripts/Helper/PRCC_Pose_Cluster.csv"
        SIL_config = None        
        if use_body_shape:
            self.setup_body_shape(body_mode, SIL_config)
        if use_pose_sample:
            self.setup_pose(pose_mode, POSE_config)    
        self.load_indentifier = self.prcc_indentifier

    def setup_deepchange_dataset(self, body_mode=None, use_body_shape=None, use_pose_sample=None, pose_mode=None):
        POSE_config = "Scripts/Helper/DeepChange_Pose_Cluster.csv"
        SIL_config = None        
        if use_body_shape:
            self.setup_body_shape(body_mode, SIL_config)
        if use_pose_sample:
            self.setup_pose(pose_mode, POSE_config)    
        self.load_indentifier = self.simple_indentifer

    def setup_celeb_dataset(self, body_mode=None, use_body_shape=None, use_pose_sample=None, pose_mode=None):
        POSE_config = "Scripts/Helper/Celeb-reID_Pose_Cluster.csv"
        if use_body_shape:
            self.setup_body_shape(body_mode, SIL_config)
        if use_pose_sample:
            self.setup_pose(pose_mode, POSE_config)    
        self.load_indentifier = self.simple_indentifer

    def setup_last_dataset(self, body_mode=None, use_body_shape=None, use_pose_sample=None, pose_mode=None):
        POSE_config = "Scripts/Helper/LaST_Pose_Cluster.csv"
        SIL_config = None        
        if use_body_shape:
            self.setup_body_shape(body_mode, SIL_config)
        if use_pose_sample:
            self.setup_pose(pose_mode, POSE_config)    
        self.load_indentifier = self.prcc_indentifier

    def setup_market_dataset(self, body_mode=None, use_body_shape=None, use_pose_sample=None, pose_mode=None):
        POSE_config = "Scripts/Helper/MARKET_Pose_Cluster.csv"
        if use_pose_sample:
            self.setup_pose(pose_mode, POSE_config)    
        self.load_indentifier = self.simple_indentifer

    def __getitem__(self, index):
        pose_label = -1   
        img_path, pid, camid, clothes_id = None, None, None, None
        try:
            if self.train:
                img_path, pid, camid, clothes_id, gender = self.dataset[index]
                img = read_image(img_path, self.illumination)            
                indentifier = self.load_indentifier(img_path)
                img, clothes_id = self.create_train_clothes(img, indentifier, clothes_id, pid)
                # save_image(normalize(img), "temp.png")
                if self.use_pose_sample:
                    pose_label = int(self.pose_label[indentifier])       
                return img, pid, camid, clothes_id, gender, pose_label
            else:
                img_path, pid, camid, clothes_id = self.dataset[index]
                img = read_image(img_path, self.illumination)            
                indentifier = self.load_indentifier(img_path)
                category = img_path.split("/")[-2]
                
                img = self.transform(img)
                if self.return_index:
                    return img, pid, camid, clothes_id, "/".join(img_path.split("/")[-2:])
                return img, pid, camid, clothes_id
                           
        except Exception as e:
            print("****", index, e)
            quit()


class ImageDataset_w_sil_with_lr_aug(ImageDataset_w_sil):
    def __init__(self, **kwargs):
        self.low_res=(16,64)
        self.motion_blur=(8,20)
        self.motion_blur_angle=(0,180)
        self.g_blur=[4,22]
        super().__init__(**kwargs)
        # self.__getitem__(0)

    def create_low_res(self, img_hr):
        H,W = img_hr.size    
        ratio = random.choice(range(*self.low_res)) / min(H,W) 
        img_lr = img_hr.resize((round( H * ratio), round( W *ratio)))
        return img_lr

    def create_g_blur(self, img_hr):
        blur_strength = random.choice(range(*self.g_blur, 2)) + 1
        img_lr = transforms.GaussianBlur(blur_strength)(img_hr)
        return img_lr

    def create_blur_motion(self, img_hr):
        blur_strength = random.choice(range(*self.motion_blur))
        blur_angle = random.choice(range(*self.motion_blur_angle))
        img_lr = apply_motion_blur(np.array(img_hr), blur_strength, blur_angle)
        img_lr = Image.fromarray( img_lr )
        return img_lr

    def __getitem__(self, index):
        try:
            img_path, pid, camid, clothes_id = self.dataset[index]
            img_hr = read_image(img_path, self.illumination)            
            indentifier = self.load_indentifier(img_path)
            if self.train:
                img, clothes_id = self.create_train_clothes(img_hr, indentifier, clothes_id, pid)
                flip_coin = random.random()
                if flip_coin < 0.33 :
                    img_lr = self.create_low_res(img_hr)
                    # img_lr.save("temp1.png")
                elif flip_coin > 0.63:
                    img_lr = self.create_g_blur(img_hr)
                    # img_lr.save("temp2.png")
                else:
                    img_lr = self.create_blur_motion(img_hr)
                    # img_lr.save("temp3.png")
                img_lr = self.transform(img_lr)
                img = torch.cat([img, img_lr.unsqueeze(0)])
                clothes_id = torch.cat([clothes_id, clothes_id[0].unsqueeze(0)])
                # save_image(normalize(img), "temp.png"), save_image(normalize(img_lr), "temp2.png")
            else:
                category = img_path.split("/")[-2]
                img = self.transform(img_hr)
            if self.return_index:
                return img, pid, camid, clothes_id, "/".join(img_path.split("/")[-2:])
            return img, pid, camid, clothes_id
        except Exception as e:
            print(index)
            print("****", index, e)
            quit()


class ImageDataset_w_sil_NTU(ImageDataset_w_sil):
    
    def read_vid_as_image(self, img_path, crop=True):        
        sil_path = img_path.replace('.mp4', '_sil.png' ).replace('RGB', 'Mask' )
        sil = read_image(sil_path)    
        sil = np.array(sil)        
        
        video = video_without_img_paths(img_path)
        img = video[len(video) // 2].asnumpy()
        if crop:
            sil, img = crop_img(sil, img, padding=10)
        else:
            img = Image.fromarray(img)     
            return sil, img
        img = Image.fromarray(img)     
        # img.save("temp.png")
        return img

    def foreground_overlap(self, img, pant_sil, shirt_sil, pid=-1, clothes_id=-1, alpha =0.70, test_mode=None, faulty=None, indentifier=None):
        pant_sil = self.load_sil(pant_sil)
        # Image.fromarray(pant_sil * 255).save("temp.png")
        shirt_sil = self.load_sil(shirt_sil)
        # Image.fromarray(shirt_sil * 255).save("temp2.png")
        
        masked_image, assigned_label = self.change_clothes_color(shirt_sil, pant_sil, img, alpha, faulty)
        # print(f'Active CUDA Device: {dist.get_rank()} COLOR: {assigned_label} NAME: {indentifier}' )
        masked_image = Image.fromarray(masked_image)
        # masked_image.save("temp.png")
        effective_label = self.original_clothes + pid * self.offset + assigned_label
        effective_label = torch.tensor([clothes_id, effective_label])
    
        return masked_image, effective_label


    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]    
        indentifier = self.load_indentifier(img_path)        
        if self.train:
            sil_array, img = self.read_vid_as_image(img_path, crop=False)
            masked_image, clothes_id = self.create_train_clothes(img, indentifier, clothes_id, pid)
            
            _, masked_image = crop_img(sil_array, np.array(masked_image), padding=10)
            _, img = crop_img(sil_array, np.array(img), padding=10)
            img = Image.fromarray(img)     
            masked_image = Image.fromarray(masked_image)     

            # img.save("T1.png"), masked_image.save("T2.png")
            img = self.transform(img)
            masked_image = self.transform(masked_image)
            img = torch.stack([img, masked_image])
        
        else:
            img = self.read_vid_as_image(img_path)
            category = img_path.split("/")[-2]
            img = self.transform(img)
        if self.return_index:
            return img, pid, camid, clothes_id, "/".join(img_path.split("/")[-2:])
        return img, pid, camid, clothes_id
    