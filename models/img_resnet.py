import torchvision
import torch
from torch import nn
from torch.nn import init
from models.utils import pooling
from models.classifier import Classifier, NormalizedClassifier        
from tools.utils import save_image, rearrange_mlr, reverse_arrange
from losses.custom import features_avg_pid

class Non_Local_Block(nn.Module):
    def __init__(self, in_channels=2048, intermediate=512, **kwargs):
        super().__init__()
        self.intermediate = intermediate
        
        self.g = nn.Conv2d(in_channels=in_channels, out_channels=intermediate, kernel_size=1)
        self.theta = nn.Conv2d(in_channels=in_channels, out_channels=intermediate, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=in_channels, out_channels=intermediate, kernel_size=1)
    
        self.W_z = nn.Sequential(
            nn.Conv2d(in_channels=self.intermediate, out_channels=in_channels, kernel_size=1),
                    nn.BatchNorm2d(in_channels)
                )
        # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

    def forward(self, query, key, value, B):
        shape = key.size()
        # query , key , value : (B C H W)
        query = self.phi(query).view(B, self.intermediate, -1)

        value = self.g(value).view(B, self.intermediate, -1)
        value = value.permute(0, 2, 1)

        key = self.theta(key).view(B, self.intermediate, -1)
        key = key.permute(0, 2, 1)
        
        inner_prod = torch.matmul(key, query)
        inner_prod = inner_prod.softmax(-1)

        y = torch.matmul(inner_prod, value)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.intermediate, *shape[2:])
        W_y = self.W_z(y)
        return W_y

class ResNet50(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        
    def forward(self, x):
        x = self.base(x)
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        return f

#######  2 Branch   #######
# B1 & B2 (joint - 2)
class ResNet50_JOINT(nn.Module):
    def __init__(self, config, define=True, **kwargs):
        super().__init__()
        self.student_mode =  None
        if define:
            resnet50 = torchvision.models.resnet50(pretrained=True)
            resnet50_2 = torchvision.models.resnet50(pretrained=True)
            if config.MODEL.RES4_STRIDE == 1:
                resnet50.layer4[0].conv2.stride=(1, 1)
                resnet50.layer4[0].downsample[0].stride=(1, 1) 
                resnet50_2.layer4[0].conv2.stride=(1, 1)
                resnet50_2.layer4[0].downsample[0].stride=(1, 1) 

            true_overlap = config.MODEL.OVERLAP - 2
            print(f"\nOverlap at -{2 + true_overlap}th Layer\n")
            # -3 ==> -1   -3 a + b = -1
            # -4 ==> -2   -4 a + b = -2 ==> a = 1 ==> -3 + b = -1 ==> b = 2
            # -5 ==> -3   x + 2 = y 

            self.common = nn.Sequential(*list(resnet50.children())[:true_overlap])
            self.branch1 = nn.Sequential(*list(resnet50.children())[true_overlap:-2])
            self.branch2 = nn.Sequential(*list(resnet50_2.children())[true_overlap:-2])
            del resnet50, resnet50_2
            self.aux_classes(config)

    def temporal_layers(self, config):
        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))

    def aux_classes(self, config):
        self.temporal_layers( config )

        self.bn1 = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        self.bn2 = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)

        init.normal_(self.bn1.weight.data, 1.0, 0.02)
        init.normal_(self.bn2.weight.data, 1.0, 0.02)

        init.constant_(self.bn1.bias.data, 0.0)
        init.constant_(self.bn2.bias.data, 0.0)

    def feature_generator(self,x):
        x = self.common(x)
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        return f1, f2
    
    def pooling(self,f1, f2, B):
        f1 = self.globalpooling(f1)
        f1 = f1.view(B, -1)
        f1 = self.bn1(f1)

        f2 = self.globalpooling(f2)
        f2 = f2.view(B, -1)
        f2 = self.bn2(f2)
        return f1, f2

    def forward(self, x):
        f1, f2 = self.feature_generator(x)

        B = x.size(0)
        f1, f2 = self.pooling(f1, f2, B)
        
        if self.training:
            return f1, f2
        else:
            return torch.cat([f1, f2],-1)
        
# B1 & B2 (Sep - 2)        
class ResNet50_SEP(ResNet50_JOINT):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, define=False, **kwargs)

        resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50_2 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
            resnet50_2.layer4[0].conv2.stride=(1, 1)
            resnet50_2.layer4[0].downsample[0].stride=(1, 1) 

        self.branch1 = nn.Sequential(*list(resnet50.children())[:-2])
        self.branch2 = nn.Sequential(*list(resnet50_2.children())[:-2])
        del resnet50, resnet50_2
        self.aux_classes(config)

    def feature_generator(self,x):        
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        return f1, f2
     
# B1 & B2 (joint - 2) + Classifier    
class ResNet50_JOINT2(ResNet50_JOINT):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=config.DATA.NUM_CLASS)
        self.teacher_mode = False
        if config.MODEL.TEACHER_MODE:
            self.teacher_mode = True 
            del self.identity_classifier

    def forward(self, x):
        f1, f2 = self.feature_generator(x)
        B = x.size(0)
        f1, f2 = self.pooling(f1, f2, B)
        
        # features_clothes, features_id = f1, f2
        if self.training:
            o2 = self.identity_classifier(f2)
            return f1, f2, o2
        elif self.teacher_mode:
            return f1, f2
        else:
            return torch.cat([f1, f2],-1)

# B1 & B2 (Sep - 2) + Classifier    
class ResNet50_SEP2(ResNet50_SEP):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=config.DATA.NUM_CLASS)

    def forward(self, x):
        f1, f2 = self.feature_generator(x)

        B = x.size(0)
        f1, f2 = self.pooling(f1, f2, B)
        o2 = self.identity_classifier(f2)
        if self.training:
            return f1, f2, o2
        else:
            return torch.cat([f1, f2],-1)


#######  3 Branch   #######
# B1 & B2 & B3 (joint - 3) + Classifier            
class ResNet50_JOINT3(ResNet50_JOINT2):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

        resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50_2 = torchvision.models.resnet50(pretrained=True)    
        resnet50_3 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50_3.layer4[0].conv2.stride=(1, 1)
            resnet50_3.layer4[0].downsample[0].stride=(1, 1) 
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
            resnet50_2.layer4[0].conv2.stride=(1, 1)
            resnet50_2.layer4[0].downsample[0].stride=(1, 1) 
        
        self.split_common = False
        self.split_branch = False
        # list(resnet50.children()) == 10 (last two classifiers)
        # 4 == first four basic layers 
        true_overlap = config.MODEL.OVERLAP - 2
        if config.MODEL.OVERLAP_2 > config.MODEL.OVERLAP:
            self.split_branch = True
            self.branch2 = nn.ModuleList()
            for e in list(resnet50_2.children())[true_overlap:-2]:
                self.branch2.append(e)
            self.split_index = 8 - len(self.common) + config.MODEL.OVERLAP_2
        elif config.MODEL.OVERLAP_2 == config.MODEL.OVERLAP:
            _ = 0 # do nothing
        else:
            self.split_common = True 
            self.common = nn.ModuleList()
            for e in list(resnet50.children())[:true_overlap]:
                self.common.append(e)
            # -3 ==> 5 
            # -2 ==> 6
            # -1 ==> 7 ==> x + 8 
            self.split_index = (config.MODEL.OVERLAP_2 + 8)
                
        true_overlap = config.MODEL.OVERLAP_2 - 2
        print(f"\n2nd Overlap at -{2 + true_overlap}th Layer with {config.MODEL.Class_2} Classes\n")
        self.branch3 = nn.Sequential(*list(resnet50_3.children())[true_overlap:-2])
        del resnet50_3
        
        self.bn3 = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn3.weight.data, 1.0, 0.02)
        init.constant_(self.bn3.bias.data, 0.0)
        self.identity_classifier2 = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=config.MODEL.Class_2)
        
    def feature_generator(self,x):
        if self.split_common:
            for i,module in enumerate(self.common):
                x = module(x)
                if i == self.split_index - 1:
                    f3 = self.branch3(x)
            f1 = self.branch1(x)
            f2 = self.branch2(x)
        elif self.split_branch:
            x = self.common(x)
            f1 = self.branch1(x)
            for i,module in enumerate(self.branch2):
                x = module(x)
                if i == self.split_index - 1:
                    f3 = self.branch3(x)
            f2 = x
        else:
            x = self.common(x)
            f1 = self.branch1(x)
            f2 = self.branch2(x)
            f3 = self.branch3(x)

        return f1, f2, f3
    
    def pooling(self,f1, f2, f3, B):
        f1 = self.globalpooling(f1)
        f1 = f1.view(B, -1)
        f1 = self.bn1(f1)

        f2 = self.globalpooling(f2)
        f2 = f2.view(B, -1)
        f2 = self.bn2(f2)

        f3 = self.globalpooling(f3)
        f3 = f3.view(B, -1)
        f3 = self.bn3(f3)
        return f1, f2, f3

    def forward(self, x):
        f1, f2, f3 = self.feature_generator(x)
        B = x.size(0)
        f1, f2, f3 = self.pooling(f1, f2, f3, B)
        
        o2 = self.identity_classifier(f2)
        o3 = self.identity_classifier2(f3)

        if self.training:
            return f1, f2, f3, o2, o3
        else:
            return torch.cat([f1, f2, f3],-1)

# B1 & B2 & B3 (joint - 3) + Classifier [3rd Branch != Eval ]           
class ResNet50_JOINT3_3(ResNet50_JOINT3):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
     
    def forward(self, x, **kwargs):
        f1, f2, f3 = self.feature_generator(x)
        B = x.size(0)
        # print((f1 - f2).mean(), (f2 -f3).mean(), (f3 - f1).mean())
        f1, f2, f3 = self.pooling(f1, f2, f3, B)
        
        if self.teacher_mode:
            return f1, f2
        
        o3 = self.identity_classifier2(f3)
        o2 = self.identity_classifier(f2)
        if self.student_mode:
            return f1, f2, o2
        elif self.training:
            return f1, f2, f3, o2, o3
        else:
            return torch.cat([f1, f2],-1)



