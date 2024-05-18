import os
import yaml
from yacs.config import CfgNode as CN



_F = CN()                                                                                                                                                  

_F.DATA = CN()
_F.DATA.NUM_CLASS = -1
_F.DATA.SAMPLER = None 
_F.DATA.GENDER_FILE = None 
_F.DATA.POSE=None
_F.DATA.POSE_MODE=None

_F.DATA.LR_MODE = None
_F.DATA.LR_TYPE = None
_F.DATA.DATASET_SPECIFIC = None
_F.DATA.LR_MODE_W_SIL = None
_F.DATA.DATASET_SAMPLING = None
# sampler

_F.AUG = CN()

_F.MODEL = CN()
# Model name. All supported model can be seen in models/__init__.py
_F.MODEL.RESUME_NO_CLASSIFIER = None 

_F.MODEL.TEACHER_WT = None 
_F.MODEL.TEACHER_MODE = None 
_F.MODEL.CAL_ON_ORIG = None 

_F.MODEL.OVERLAP = -2
_F.MODEL.OVERLAP_2 = -2
_F.MODEL.Class_2 = -1

_F.MODEL.OVERLAP_3 = -2
_F.MODEL.Class_3 = -1

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_F.TEST = CN()
# Perform evaluation after every N epochs (set to -1 to test after training)
_F.TEST.EVAL_STEP = 10
# Start to evaluate after specific epoch
_F.TEST.START_EVAL = 0
_F.TEST.VALIDATION = None

_F.TEST.SAVE_WT = True
# -----------------------------------------------------------------------------
# Losses for training 
# -----------------------------------------------------------------------------
_F.LOSS = CN()
# Classification loss
_F.LOSS.CLA_LOSS = 'crossentropy'
# Clothes classification loss
_F.LOSS.CLOTHES_CLA_LOSS = 'cosface'
# Scale for classification loss
_F.LOSS.CLA_S = 16.
# Margin for classification loss
_F.LOSS.CLA_M = 0.
# Pairwise loss
_F.LOSS.PAIR_LOSS = 'triplet'
# The weight for pairwise loss
_F.LOSS.PAIR_LOSS_WEIGHT = 0.0
# Scale for pairwise loss
_F.LOSS.PAIR_S = 16.
# Margin for pairwise loss
_F.LOSS.PAIR_M = 0.3
# Clothes-based adversarial loss
_F.LOSS.CAL = 'cal'
# Epsilon for clothes-based adversarial loss
_F.LOSS.EPSILON = 0.1
# Momentum for clothes-based adversarial loss with memory bank
_F.LOSS.MOMENTUM = 0.
_F.LOSS.LOSS_MODE = None
_F.LOSS.ADDITIONAL_LOSS = None

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Fixed random seed
_F.SEED = 1
# Perform evaluation only
_F.EVAL_MODE = False
# GPU device ids for CUDA_VISIBLE_DEVICES
_F.GPU = '0'
# Path to output folder, overwritten by command line argument
_F.OUTPUT = '/data/guxinqian/logs/'
# Tag of experiment, overwritten by command line argument
_F.TAG = 'res50-ce-cal'

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_F.TRAIN = CN()
_F.TRAIN.START_EPOCH = 0
_F.TRAIN.MAX_EPOCH = 150
_F.TRAIN.ONLY_CAL = None
_F.TRAIN.FN= "None"

# Using amp for training
_F.TRAIN.AMP = False

#POSE DISENTANGLE
_F.TRAIN.POSE_ID = None
_F.TRAIN.POSE_CL = None
_F.TRAIN.POSE_BOTH = None

# Optimizer
_F.TRAIN.OPTIMIZER = CN()
_F.TRAIN.OPTIMIZER.NAME = 'adam'
# Learning rate
_F.TRAIN.OPTIMIZER.LR = 0.00035
_F.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4


def common_update_config_via_args(config, args):
    config.defrost()
    
    # merge from specific arguments
    #### Misc
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.strict:
        config.MODEL.STRICT = True
    if "amp" in args and args.amp:
        config.TRAIN.AMP = True
    if args.gpu:
        config.GPU = args.gpu
    if args.use_validation:
        config.TEST.VALIDATION = args.use_validation
    if args.eval_step:
        config.TEST.EVAL_STEP = args.eval_step
    if args.seed:
        config.SEED =  args.seed
    if args.no_save:
        config.TEST.SAVE_WT = False 
    if args.LR_MODE:
        config.DATA.LR_MODE = True
    if args.LR_TYPE:
        config.DATA.LR_TYPE = args.LR_TYPE
    if args.dataset_specific:
        config.DATA.DATASET_SPECIFIC = args.dataset
    
    #### Data
    if args.root:
        config.DATA.ROOT = args.root
    if args.silhouettes:
        config.DATA.SILHOUETTES = args.silhouettes
    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.batch_size:
        config.DATA.TRAIN_BATCH = args.batch_size
    if args.size:
        print(args.size)
        config.DATA.HEIGHT, config.DATA.WIDTH = args.size[0], args.size[1]
    if args.subset_dataset:
        config.DATA.SUBSET = args.subset_dataset
    if args.sil_mode:
        config.DATA.SIL_MODE = args.sil_mode
    if args.sampler:
        config.DATA.SAMPLER = args.sampler
    if args.use_gender:
        config.DATA.GENDER_FILE = args.use_gender

    #### Training
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.no_classifier:
        config.MODEL.RESUME_NO_CLASSIFIER = True
    if "eval" in args and args.eval:
        config.EVAL_MODE = True
    if "max_epochs" in args and args.max_epochs:
        config.TRAIN.MAX_EPOCH = args.max_epochs
    if args.train_fn:
        config.TRAIN.FN = args.train_fn
    if args.Pose:
        config.DATA.POSE = args.Pose
        config.DATA.POSE_MODE = args.pose_mode  
    if args.only_CAL:
        config.TRAIN.ONLY_CAL = True 
    if args.pose_id:
        config.TRAIN.POSE_ID = True
    if args.pose_cl:
        config.TRAIN.POSE_CL = True
    if args.pose_both:
        config.TRAIN.POSE_BOTH = True
    if args.lr_aug:
        config.DATA.LR_MODE_W_SIL = True 

    #### Loss
    if args.pair_loss:
        config.LOSS.PAIR_LOSS = args.pair_loss
    if args.loss_mode:
        config.LOSS.LOSS_MODE = args.loss_mode
    if args.additional_loss:
        config.LOSS.ADDITIONAL_LOSS = args.additional_loss
    
    #### Model
    if args.backbone:
        config.MODEL.NAME = args.backbone
    if "teacher_wt" in args and args.teacher_wt:
        config.MODEL.TEACHER_WT = args.teacher_wt
        config.MODEL.TEACHER_MODE = True 
    if args.cal_on_orig:
        config.MODEL.CAL_ON_ORIG = True
    if args.class_2:
        config.MODEL.Class_2 = args.class_2
    if args.class_3:
        config.MODEL.Class_3 = args.class_3    
    if args.overlap_1:
        config.MODEL.OVERLAP = args.overlap_1
    if args.overlap_2:
        config.MODEL.OVERLAP_2 = args.overlap_2
    if args.overlap_3:
        config.MODEL.OVERLAP_3 = args.overlap_3
    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, config.TAG)
    
    config.freeze()

