import data.img_transforms as T
import data.spatial_transforms as ST
import data.temporal_transforms as TT
from torch.utils.data import DataLoader
from data.dataloader import DataLoaderX
from data.dataset_loader import *
from data.samplers import DistributedRandomIdentitySampler, DistributedInferenceSampler, DistributedRandomIdentitySampler_Percent

from data.datasets.ltcc import *
from .datasets.celebreid import *
from data.datasets.prcc import *
from data.datasets.deepchange import *
from data.datasets.last import *

import copy 

__factory = {
    'ltcc': LTCC,
    'ltcc_colors': LTCC_custom_colors,    
    'ltcc_cc_gender': LTCC_CC_gender,    
    
    'prcc': PRCC,
    'prcc_colors': PRCC_custom_colors,    
    'prcc_cc_gender': PRCC_CC_gender,

    'celeb': CelebreID,    
    'celeb_colors': CelebreID_custom_colors,    
    'celeb_cc_colors': CelebreID_CC_gender,    

    'deepchange': DeepChange,
    'deepchange_colors': DeepChange_custom_colors,
    'deepchange_cc_gender': DeepChange_CC_gender,

    'last': LaST,
    'last_colors': LaST_custom_colors,
    'last_cc_gender': LaST_CC_gender,

}

def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(config.DATA.DATASET, __factory.keys()))

    additional_args = {}
    if config.DATA.SUBSET:
        additional_args["subset"] = config.DATA.SUBSET
    if config.TEST.VALIDATION :
        additional_args["validation"] = config.TEST.VALIDATION 
    if config.DATA.GENDER_FILE:
        additional_args["gender_file"] = config.DATA.GENDER_FILE
    if config.MODEL.CAL_ON_ORIG:
        additional_args["new_clothes_pids"] = False 
        
    dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT, **additional_args)

    return dataset


def build_img_transforms(config):
    transform_train = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.RandomCroping(p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=config.AUG.RE_PROB)
    ])
    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test



def build_dataloader(config, local_rank=None, sampling = None ):
    dataset = build_dataset(config)
    additional_args = {}
    additional_args["dataset_name"] = config.DATA.DATASET
    
    data_loader_kwargs = {}
    if local_rank is not None:
        data_loader_kwargs["local_rank"] = local_rank

    # image dataset
    train_sampler = DistributedRandomIdentitySampler(dataset.train,  num_instances=config.DATA.NUM_INSTANCES,  seed=config.SEED)
    if sampling:
        train_sampler = DistributedRandomIdentitySampler_Percent(dataset.train,  percent=sampling, num_instances=config.DATA.NUM_INSTANCES,  seed=config.SEED)
    transform_train, transform_test = build_img_transforms(config)
    IMG_dataset = ImageDataset
    if config.EVAL_MODE:
        additional_args["return_index"] = True

    if config.DATA.SILHOUETTES:
        IMG_dataset = ImageDataset_w_sil
        additional_args["original_clothes"] = dataset.original_num_clothes
        additional_args["silhouettes"] = config.DATA.SILHOUETTES   
        additional_args["sil_mode"] = config.DATA.SIL_MODE
        additional_args["clothes_dict"] = dataset.clothes_dict
    
    if config.DATA.LR_MODE:
        IMG_dataset = ImageDataset_w_res
        if config.DATA.LR_TYPE:
            if config.DATA.LR_TYPE == "LR":
                IMG_dataset = ImageDataset_w_res_ONLY_LR 
            elif config.DATA.LR_TYPE == "MB":
                IMG_dataset = ImageDataset_w_res_ONLY_MB
            elif config.DATA.LR_TYPE == "OOF":
                IMG_dataset = ImageDataset_w_res_ONLY_OOF
        else:
            if config.DATA.DATASET_SPECIFIC:
                if "prcc" in config.DATA.DATASET_SPECIFIC or "deepchange" in config.DATA.DATASET_SPECIFIC:
                    IMG_dataset = ImageDataset_w_res_prcc
                elif "ltcc" in config.DATA.DATASET_SPECIFIC or "last" in config.DATA.DATASET_SPECIFIC:
                    IMG_dataset = ImageDataset_w_res_ltcc
                elif "celeb" in config.DATA.DATASET_SPECIFIC:
                    IMG_dataset = ImageDataset_w_res_ltcc
                else:
                    import pdb
                    pdb.set_trace()
        
    if config.MODEL.TEACHER_MODE:
        IMG_dataset = ImageDataset_w_res
        if config.DATA.DATASET_SPECIFIC:
            if "prcc" in config.DATA.DATASET_SPECIFIC or "deepchange" in config.DATA.DATASET_SPECIFIC:
                IMG_dataset = ImageDataset_w_res_prcc
            elif "ltcc" in config.DATA.DATASET_SPECIFIC or "last" in config.DATA.DATASET_SPECIFIC:
                IMG_dataset = ImageDataset_w_res_ltcc
            else:
                import pdb
                pdb.set_trace()
    
    elif ("gender" in config.DATA.DATASET) or (config.DATA.POSE):
        IMG_dataset = ImageDataset_w_gender_Pose
        additional_args["pose_mode"] = config.DATA.POSE_MODE
        additional_args["pose_path"] = config.DATA.POSE
        additional_args["gender_file"] = config.DATA.GENDER_FILE
        additional_args["use_pose_sample"] = True if config.DATA.POSE else None 
    elif config.DATA.GENDER_FILE:
        IMG_dataset = ImageDataset_w_gender
        additional_args["gender_file"] = config.DATA.GENDER_FILE
    elif config.DATA.POSE:
        IMG_dataset = ImageDataset_w_pose
        additional_args["pose_mode"] = config.DATA.POSE_MODE
        additional_args["pose_path"] = config.DATA.POSE
    
    if config.MODEL.TEACHER_MODE and \
    ("ltcc" in config.DATA.DATASET or "prcc" in config.DATA.DATASET or "deepchange" in config.DATA.DATASET or "last" in config.DATA.DATASET):
        
        IMG_dataset_Teacher = ImageDataset_w_res_ltcc
        
        trainloader = DataLoaderX(dataset=IMG_dataset_Teacher(dataset=dataset.train, transform=transform_train, train=True, **additional_args),
        # trainloader = DataLoaderX(dataset=IMG_dataset_Teacher(dataset=(dataset.train + dataset.gallery + dataset.query), transform=transform_train, train=True, **additional_args),
            sampler=train_sampler, batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, drop_last=True, **data_loader_kwargs)
        queryloader, galleryloader = None, None
    else:
        trainloader = DataLoaderX(dataset=IMG_dataset(dataset=dataset.train, transform=transform_train, train=True, **additional_args),
                                sampler=train_sampler,
                                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn, **data_loader_kwargs)
        galleryloader = DataLoaderX(dataset=IMG_dataset(dataset=dataset.gallery, transform=transform_test, train=False, **additional_args),
                                sampler=DistributedInferenceSampler(dataset.gallery),
                                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=False, shuffle=False, **data_loader_kwargs)
        if 'prcc' in config.DATA.DATASET :
            queryloader_same = DataLoaderX(dataset=IMG_dataset(dataset=dataset.query_same, transform=transform_test, train=False, **additional_args),
                                    sampler=DistributedInferenceSampler(dataset.query_same),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False, **data_loader_kwargs)
            queryloader_diff = DataLoaderX(dataset=IMG_dataset(dataset=dataset.query_diff, transform=transform_test, train=False, **additional_args),
                                    sampler=DistributedInferenceSampler(dataset.query_diff),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False, **data_loader_kwargs)

            return trainloader, [queryloader_same, queryloader_diff], galleryloader, dataset, train_sampler
        else:
            queryloader = DataLoaderX(dataset=IMG_dataset(dataset=dataset.query, transform=transform_test, train=False, **additional_args),
                                    sampler=DistributedInferenceSampler(dataset.query),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False , **data_loader_kwargs)

    return trainloader, queryloader, galleryloader, dataset, train_sampler



def worker_init_fn(worker_id): 
    random.seed(worker_id)     
