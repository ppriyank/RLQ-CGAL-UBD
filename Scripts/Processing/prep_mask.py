import os 
from PIL import Image
import numpy as np 
import cv2
import math 
import glob
from multiprocessing.pool import ThreadPool as Pool
pool_size = 8
pool = Pool(pool_size)


def create_mask(mask_file, exclude=[0]):
    misc = np.load(mask_file)
    misc = np.argmax(misc, axis=2)
    for i,row in enumerate(misc):
        for j, pixel in enumerate(row):
            if pixel in exclude:
                misc[i][j] = 0 
            else:
                misc[i][j] = 1
    misc = np.stack([misc,misc,misc,], 2)
    return misc

def create_sil(mask_file, dest_folder, img_name, delete_mask=None):
    mask = create_mask(mask_file)
    
    ##### Background 
    # background_mask = (1 - mask).astype(np.uint8) * 255
    background_mask = mask.astype(np.uint8) * 255
    background_mask = Image.fromarray(background_mask)
    # background_mask.save("temp.png")
    background_mask.save( os.path.join(dest_folder, img_name.replace(".png", "_sil.png").replace(".jpg", "_sil.png") ) )

    ##### PANTs 
    pants = [9,12,]
    lower_mask = create_mask(mask_file, exclude=[i for i in range(20) if i not in pants ])
    lower_mask = lower_mask.astype(np.uint8) * 255
    lower_mask = Image.fromarray(lower_mask)
    # lower_mask.save( "temp.png" ) 
    lower_mask.save( os.path.join(dest_folder, img_name.replace(".png", "_pant.png")).replace(".jpg", "_pant.png")  )

    ##### SHIRTs 
    shirts = [5,7,10]
    upper_mask = create_mask(mask_file, exclude=[i for i in range(20) if i not in shirts ])
    upper_mask = upper_mask.astype(np.uint8) * 255
    upper_mask = Image.fromarray(upper_mask)
    # upper_mask.save( "temp2.png" )
    upper_mask.save( os.path.join(dest_folder, img_name.replace(".png", "_shirt.png")).replace(".jpg", "_shirt.png") )

    
    ##### FULL PANTs 
    pants = [8,9,12,16,17,18,19]
    lower_mask = create_mask(mask_file, exclude=[i for i in range(20) if i not in pants ])
    lower_mask = lower_mask.astype(np.uint8) * 255
    lower_mask = Image.fromarray(lower_mask)
    # lower_mask.save( "temp.png" ) 
    lower_mask.save( os.path.join(dest_folder, img_name.replace(".png", "_full_pant.png")).replace(".jpg", "_full_pant.png")  )

    ##### FULL SHIRTs 
    shirts = [5,6,7,10,11,15,14]
    upper_mask = create_mask(mask_file, exclude=[i for i in range(20) if i not in shirts ])
    upper_mask = upper_mask.astype(np.uint8) * 255
    upper_mask = Image.fromarray(upper_mask)
    # upper_mask.save( "temp2.png" )
    upper_mask.save( os.path.join(dest_folder, img_name.replace(".png", "_full_shirt.png")).replace(".jpg", "_full_shirt.png") )
    
    ##### Clothes
    clothes = [1,3,5,6,7,8,9,10,11,12,18,19]
    clothes_mask = create_mask(mask_file, exclude=[i for i in range(20) if i not in clothes ])
    clothes_mask = clothes_mask.astype(np.uint8) * 255
    clothes_mask = Image.fromarray(clothes_mask)
    # clothes_mask.save( "temp.png" ) 
    clothes_mask.save( os.path.join(dest_folder, img_name.replace(".png", "_clothes.png")).replace(".jpg", "_clothes.png")  )

    ##### Head
    heads = [1,2,4, 13]
    heads = create_mask(mask_file, exclude=[i for i in range(20) if i not in heads ])
    heads = heads.astype(np.uint8) * 255
    heads = Image.fromarray(heads)
    # clothes_mask.save( "temp.png" ) 
    heads.save( os.path.join(dest_folder, img_name.replace(".png", "_head.png")).replace(".jpg", "_head.png")  )
    
    if delete_mask:
        os.remove(mask_file) 

def generate_sil_mask(root=None, dest=None):
    npys = os.listdir(root)
    print(f" *** {len(npys)} ***" )
    dest_folder = os.path.join(dest, "train")
    if not os.path.exists(dest_folder): os.makedirs(dest_folder, exist_ok=True)
    for mask_file in npys:
        mask_file = os.path.join(root, mask_file) 
        img_name = (mask_file[:-4] + ".png").split("/")[-1]
        # create_sil(mask_file, dest_folder, img_name)
        pool.apply_async(create_sil, (mask_file, dest_folder, img_name))     

def generate_sil_mask_mevid(root=None, dest=None):
    ids = sorted(os.listdir(root))
    for id in ids:
        id_path = os.path.join(root, id)
        npys = glob.glob(f"{id_path}/*.npy")
        print(f" {id} *** {len(npys)} ***" )
        if len(npys) == 0 :
            continue 
        for np_file in npys:
            mask_file = os.path.join(id_path, np_file) 
            img_name = (mask_file[:-4] + ".png").split("/")[-1]
            # create_sil(mask_file, id_path, img_name, delete_mask=True)
            pool.apply_async(create_sil, (mask_file, id_path, img_name, True))     
            
            


# LTCC
# generate_sil_mask(root_ltcc, src=src_ltcc, dest= dest_ltcc, categories=["query"], query_path=query_path_ltcc)

# PRCC
# prcc_npy="/data/priyank/synthetic/PRCC/masks/NPY"
# prcc_sil="/data/priyank/synthetic/PRCC/masks/jpgs" 
# generate_sil_mask(prcc_npy, dest= prcc_sil)

# Celeb
# celeb_npy="/data/priyank/synthetic/Celeb-reID/masks/NPY"
# celeb_sil="/data/priyank/synthetic/Celeb-reID/masks/jpgs" 
# generate_sil_mask(celeb_npy, dest= celeb_sil)

# Deepchange
# deepchange_npy="/data/priyank/synthetic/DeepChangeDataset/masks/NPY"
# deepchange_sil="/data/priyank/synthetic/DeepChangeDataset/masks/jpgs" 
# generate_sil_mask(deepchange_npy, dest= deepchange_sil)

# last
# last_npy="/data/priyank/synthetic/LaST/masks/NPY"
# last_sil="/data/priyank/synthetic/LaST/masks/jpgs" 
# generate_sil_mask(last_npy, dest= last_sil)

pool.close()
pool.join()






# https://github.com/GoGoDuck912/Self-Correction-Human-Parsing
# 0 background 
# 1 hat
# 2 hair
# 3 gloves
# 4 sunglasses
# 5 upper cloth
# 6 dress
# 7 coat
# 8 sock 
# 9 pant
# 10 jumpsuit
# 11 scarf 
# 12 skirt 
# 13 face 
# 14 left leg (its actually left arm)
# 15 right leg (its actually right arm)
# 16 left arm (its actually left leg)
# 17 right arm (its actually right leg) 
# 18 left shoe
# 19 right shoe  


# cd ~/CCReID/
# python Scripts/Processing/prep_mask.py
