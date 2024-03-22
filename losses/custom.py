import torch 
import torch.nn as nn 
from torch.nn import functional as F
from mmagic.models.losses.gan_loss import GANLoss
from losses.gather import GatherLayer
from tools.utils import rearrange
from losses.triplet_loss import TripletLoss
import torch.distributed as dist 

class MSE():
    def __init__(self, mode="l2", use_square=True, mean=True):
        self.use_square = use_square
        self.mode = mode
        if mode == None:
            self.mode = "l2"
        self.mean = mean

    def __call__(self, x, y):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        if self.mode == "l2":
            diff = x - y 
            diff = (diff ** 2).sum(-1)
            if self.mean:
                if self.use_square:
                    return diff.mean()
                else:
                    return (diff ** 0.5).mean()    
            else:
                if self.use_square:
                    return diff
                else:
                    return (diff ** 0.5)
        elif self.mode == "cosine":
            cosine = x * y 
            cosine = (cosine).sum(-1)
            diff = 1 - cosine
            if self.mean:
                return diff.mean()    
            else:
                return diff

def cosine_dist(x,y, w_i=1, **misc):
    cosine = nn.CosineSimilarity(-1)(x,y)
    # x = F.normalize(x, p=2, dim=-1)
    # y = F.normalize(y, p=2, dim=-1)
    # cosine = x * y
    # cosine = cosine.sum(-1)
    diff = 1 - cosine
    return (diff * w_i).mean()

def pairwise_l2_dist(x):
    x = F.normalize(x, p=2, dim=-1)
    x = x.unsqueeze(1) - x.unsqueeze(0)
    x = (x ** 2).sum(-1)
    return x


class KL_Loss():
    def __init__(self,tau=0.8, lambda1=0.64, ):
        self.tau = tau
        self.lambda1 = lambda1
        
    def __call__(self, teacher, student, **misc):
        # F.kl_div(F.log_softmax(student / self.tau, dim=1), F.log_softmax(teacher / self.tau, dim=1), reduction='mean', log_target=True )
        # (- (teacher / self.tau).softmax(-1) * ( (student / self.tau).softmax(-1) / (teacher / self.tau).softmax(-1) ).log()).mean() 
        # (- (teacher / self.tau).softmax(-1) * (student / self.tau).softmax(-1).log()).mean() + ((teacher / self.tau).softmax(-1) * (teacher / self.tau).softmax(-1).log()).mean()
        distillation_loss = F.kl_div(
            F.log_softmax(student / self.tau, dim=-1),
            F.log_softmax(teacher / self.tau, dim=-1),
            reduction='sum', log_target=True ) * (self.tau * self.tau) / student.numel()
        return self.lambda1 * distillation_loss

class Soft_CE(KL_Loss):
    
    def __call__(self, teacher, student, **misc):
        # distillation_loss = F.kl_div(
        #     F.log_softmax(student / self.tau, dim=-1),
        #     F.log_softmax(teacher / self.tau, dim=-1),
        #     reduction='sum', log_target=True )
        
        # distillation_loss = (- (teacher / self.tau).softmax(-1) * (student / self.tau).softmax(-1).log()).sum() + ((teacher / self.tau).softmax(-1) * (teacher / self.tau).softmax(-1).log()).sum()

        distillation_loss = (- (teacher / self.tau).softmax(-1) * (student / self.tau).softmax(-1).log()).sum()
        distillation_loss = distillation_loss * (self.tau * self.tau) / student.numel()
        return self.lambda1 * distillation_loss


# KL with output clothes + output id 
class KL_Loss_o_oid(KL_Loss):
    def __call__(self, outputs_id, outputs, **misc):
        distillation_loss = F.kl_div(F.log_softmax(outputs / self.tau, dim=-1), F.log_softmax(outputs_id / self.tau, dim=-1), reduction='sum', log_target=True ) * (self.tau * self.tau) / outputs.numel()
        return self.lambda1 * distillation_loss

class Center_Loss(nn.Module):
    def __init__(self, num_classes=-1, feat=-1, center_loss_weight=0.0005, metric='l2', **args) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.feat = feat
        self.center_loss_weight = center_loss_weight

        self.distance_fn = MSE(mode=metric, mean=False)
        self.centers = nn.Parameter(torch.randn(num_classes, self.feat))
        self.use_gpu = True 
        self.classes = torch.arange(self.num_classes).long()
        self.optimizer_center = torch.optim.SGD(self.parameters(), lr=0.5)
        
    def center_loss(self, x, labels):
        self.optimizer_center.zero_grad()
        distmat = self.distance_fn(x.unsqueeze(1), self.centers.unsqueeze(0))
        batch_size = x.size(0)
        classes = self.classes.to(labels.device) 
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return self.center_loss_weight * loss

    def forward(self, x, labels=-1):
        return self.center_loss(x, labels)

    def update_params(self, ):
        for param in self.parameters():
            if param.requires_grad:
                param.grad.data = param.grad.data.clip(min=-5, max=5)
                param.grad.data *= (1./self.center_loss_weight)
        # param.grad.data.mean(-1)
        # self.centers.sum(-1)
        # [(e.shape,e.grad.sum(-1)) for e in self.optimizer_center.param_groups[0]["params"] if e.requires_grad ]
        self.optimizer_center.step()
        self.optimizer_center.zero_grad()

# KL with output clothes + output id  + Maximize_pose_dist 
class KL_Pose(nn.Module):
    def __init__(self, pose_id=None, pose_cl=None, pose_both=None):
        super().__init__()
        self.custom_pose = False
        if pose_id or pose_cl or pose_both:
            self.custom_pose = True 
            self.pose_id = pose_id
            self.pose_cl = pose_cl
            self.pose_both = pose_both

        self.POSE = Maximize_pose_dist()
        self.KL = KL_Loss_o_oid()

    def __call__(self, outputs_id, outputs, features_id=None, features_pose=None, features_clothes=None, **misc):
        pose = None
        if self.custom_pose:
            if self.pose_id:
                pose = self.POSE(features_id, features_pose)
            elif self.pose_cl:
                pose = self.POSE(features_clothes, features_pose)
            elif self.pose_both:
                pose = self.POSE(features_clothes, features_pose)
                pose += self.POSE(features_id, features_pose)
        else:
            pose = self.POSE(features_id, features_pose)

        kl = self.KL(outputs_id, outputs)
        return kl + pose 

# KL with output clothes + output id  + Maximize_pose_dist3 
class KL_Pose3(KL_Pose):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.POSE = Maximize_pose_dist3()
        
    


# https://arxiv.org/pdf/2212.09498.pdf
class Maximize_pose_dist(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.cosine_product = nn.CosineSimilarity(-1)
        self.rectifier = nn.ReLU() 

    def forward(self, features_id=None, features_pose=None, **kwargs):
        
        loss = self.cosine_product(features_id, features_pose)
        loss = self.rectifier(loss).mean()
        return loss

# abs instead of relu for position 
class Maximize_pose_dist3(Maximize_pose_dist):
    def forward(self, features_id=None, features_pose=None, **kwargs):
        loss = self.cosine_product(features_id, features_pose)
        loss = loss.abs().mean()
        return loss

# L2 instead of relu 
class Maximize_pose_dist4(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.cosine_product = nn.MSELoss(reduction='none')
        self.feature_dim = dim 
    
    def forward(self, features_id=None, features_pose=None, **kwargs):
        loss = self.cosine_product(features_id, features_pose)
        loss = - (loss.sum(-1) / self.feature_dim).mean()
        return loss

class Pose_TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.m = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features_id, pids, pose=None, **kwargs):
        # l2-normlize
        features_id = F.normalize(features_id, p=2, dim=1)
        Batch_size = features_id.shape[0]
        # gather all samples from different GPUs as gallery to compute pairwise loss.
        gallery_inputs = torch.cat(GatherLayer.apply(features_id.contiguous()), dim=0)
        gallery_pids = torch.cat(GatherLayer.apply(pids.contiguous()), dim=0)
        gallery_pose = torch.cat(GatherLayer.apply(pose.contiguous()), dim=0)

        # compute distance
        dist = 1 - torch.matmul(features_id, gallery_inputs.t()) # values in [0, 2]

        # get positive and negative masks
        pids, gallery_pids = pids.view(-1,1), gallery_pids.view(-1,1)
        pose, gallery_pose = pose.view(-1,1), gallery_pose.view(-1,1)

        mask_pos = torch.eq(pids, gallery_pids.T).float().cuda()
        mask_neg = 1 - mask_pos

        mask_pose_pos = torch.eq(pose, gallery_pose.T).float().cuda()
        mask_pose_neg = 1 - mask_pose_pos
        
        # positive samples from the different pose 
        dist_ap_diff_pose, _ = torch.max((dist - (mask_neg + mask_pose_pos) * 99999999.), dim=1)
        
        # positive samples from the same pose 
        dist_ap_same_pose, _ = torch.max((dist - (mask_neg + mask_pose_neg) * 99999999.), dim=1)
        
        # negative samples from the same pose 
        dist_an, _ = torch.min((dist + (mask_pos + mask_pose_neg) * 99999999.), dim=1)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        
        # loss = self.ranking_loss(dist_an, dist_ap, y)
        loss_same = self.ranking_loss(dist_an, dist_ap_same_pose, y)
        loss_diff = self.ranking_loss(dist_an, dist_ap_diff_pose, y)

        return loss_same + loss_diff

class Maximize_pose_dist2(Maximize_pose_dist):
    def __init__(self, ):
        super().__init__()
        from models.utils import pooling
        self.globalpooling = pooling.MaxAvgPooling()

    def forward(self, features_id=None, pose=None, **kwargs):
        
        pose = rearrange(self.globalpooling(pose), "B C 1 1 -> B C")

        loss = self.cosine_product(features_id, pose)
        loss = self.rectifier(loss).mean()
        return loss


class Center_ID(Center_Loss):
    def __init__(self, **args) -> None:
        super().__init__(**args)
    def forward(self, features_id, pids=-1, **kwargs):
        return self.center_loss(features_id, pids)

class Center_CL(Center_Loss):
    def __init__(self, **args) -> None:
        super().__init__(**args)
    def forward(self, features_clothes, pids=-1, **kwargs):
        return self.center_loss(features_clothes, pids)

class Center_Conc(Center_Loss):
    def __init__(self, **args) -> None:
        super().__init__(**args)
    def forward(self, features_clothes, features_id, pids=-1, **kwargs):
        return self.center_loss(torch.cat([features_clothes, features_id],-1), pids)

class Center_Sep(nn.Module):
    def __init__(self, num_classes=-1, feat=-1, center_loss_weight=0.0005, metric='l2', **args) -> None:
        super().__init__()
        self.c1_id = Center_Loss(num_classes=num_classes, feat=feat, center_loss_weight=center_loss_weight, metric=metric, )
        self.c1_cl = Center_Loss(num_classes=num_classes, feat=feat, center_loss_weight=center_loss_weight, metric=metric, )

    def forward(self, features_clothes, features_id, pids=-1, **kwargs):
        c1 = self.c1_id(features_id, pids)
        c2 = self.c1_cl(features_clothes, pids)
        return c1 + c2

    def update_params(self, ):
        self.c1_id.update_params()
        self.c1_cl.update_params()

class Center_CL_KL_OOID(Center_Loss):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.KL_loss = KL_Loss_o_oid()
    def forward(self, features_clothes, pids=-1, outputs_id=None, outputs=None, **kwargs):
        cl = self.center_loss(features_clothes, pids)
        kl = self.KL_loss(outputs_id, outputs)
        return cl + kl 

class Center_Sep_KL_OOID(Center_Sep):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.KL_loss = KL_Loss_o_oid()
    
    def forward(self, features_clothes, pids=-1, outputs_id=None, outputs=None, features_id=None, **kwargs):
        c1 = self.c1_id(features_id, pids)
        c2 = self.c1_cl(features_clothes, pids)
        kl = self.KL_loss(outputs_id, outputs)
        return c1 + c2 + kl 

class Center_ID_KL_OOID(Center_ID):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.KL_loss = KL_Loss_o_oid()
    
    def forward(self, features_clothes, pids=-1, outputs_id=None, outputs=None, features_id=None, **kwargs):
        c1 = self.center_loss(features_id, pids)
        kl = self.KL_loss(outputs_id, outputs)
        return c1 + kl 



def features_avg_pid(x, pids, num_classes):
    B = x.shape[0]
    targets = torch.zeros([B, num_classes]).scatter_(1, pids.unsqueeze(1).data.cpu(), 1).cuda()
    
    # misc 
    label_count = targets.sum(0)
    class_present = label_count != 0 
    label_count = label_count + 1e-6

    # (102, B) x (B, D) == (102, D) # sum Features 
    targets = torch.mm(targets.t() , x)

    dist.all_reduce(targets, op=dist.ReduceOp.SUM)
    dist.all_reduce(label_count, op=dist.ReduceOp.SUM)
    # print(targets, label_count)

    # average of features
    targets = targets / label_count.unsqueeze(-1)
    # unqiue classes features 
    # targets = targets[class_present]

    target_labels = torch.arange(num_classes).cuda()
    # [class_present != 0 ]
    return targets, target_labels
