import logging
from models.classifier import Classifier, NormalizedClassifier
from models.img_resnet import *


__factory = {
    'resnet50': ResNet50,

    'resnet50_separate': ResNet50_SEP,
    'resnet50_separate2': ResNet50_SEP2,

    'resnet50_joint': ResNet50_JOINT,
    'resnet50_joint2': ResNet50_JOINT2,

    'resnet50_joint3_3': ResNet50_JOINT3_3,   # Pose 
    'resnet50_joint3_8': ResNet50_JOINT3_8,   
    
}



def build_model(config, num_identities, num_clothes):
    logger = logging.getLogger('reid.model')
    # Build backbone
    logger.info("Initializing model: {}".format(config.MODEL.NAME))
    if config.MODEL.NAME not in __factory.keys():
        raise KeyError("Invalid model: '{}'".format(config.MODEL.NAME))
    else:
        logger.info("Init model: '{}'".format(config.MODEL.NAME))
        model = __factory[config.MODEL.NAME](config)
        
    logger.info("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    identity_classifier = None
    if num_identities:
        # Build classifier
        if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
            identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
        else:
            identity_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)

    clothes_classifier = None
    if num_clothes:
        clothes_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)

    return model, identity_classifier, clothes_classifier

def build_extra_id_classifier(config, num_identities, input_dim=-1):
    logger = logging.getLogger('reid.model')
    # Build classifier
    if input_dim == -1:
        if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
            identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
        else:
            identity_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
    else:
        if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
            identity_classifier = Classifier(feature_dim=input_dim, num_classes=num_identities)
        else:
            identity_classifier = NormalizedClassifier(feature_dim=input_dim, num_classes=num_identities)
    return identity_classifier
