import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss
from tools import misc

class EdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, edge_factor=1.0):
        super(EdgeLoss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.edge_factor = edge_factor

    def get_boundary(self, x):
        laplacian_kernel_target = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).cuda(device=x.device)
        x = x.unsqueeze(1).float()
        x = F.conv2d(x, laplacian_kernel_target, padding=1)
        x = x.clamp(min=0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0

        return x

    def compute_edge_loss(self, logits, targets):
        bs = logits.size()[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        # print(boundary_targets.shape)
        logits = F.softmax(logits, dim=1).argmax(dim=1).squeeze(dim=1)
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        # print(boundary_pre)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        # print(boundary_pre)
        # dice_loss = 1 - ((2. * (boundary_pre * boundary_targets).sum(1) + 1.0) /
        #                  (boundary_pre.sum(1) + boundary_targets.sum(1) + 1.0))
        # dice_loss = dice_loss.mean()
        edge_loss = F.binary_cross_entropy_with_logits(boundary_pre, boundary_targets)

        return edge_loss

    def forward(self, logits, targets):
        loss = self.main_loss(logits, targets) + self.compute_edge_loss(logits, targets) * self.edge_factor
        return loss


class OHEM_CELoss(nn.Module):

    def __init__(self, thresh=0.7, ignore_index=255):
        super(OHEM_CELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_index].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class UnetFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels):
        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits
            loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux, labels)
        else:
            loss = self.main_loss(logits, labels)

        return loss

class OODConfidenceLoss(nn.modules.loss._Loss):
    def __init__(self,nb_classes=6,half_random=True,beta=0.3,lbda=0.1,lbda_control=True,ignore_index=255):
        self.nb_classes = nb_classes
        self.task = "segmentation"
        self.half_random = half_random
        self.beta = beta
        self.lbda = lbda
        self.lbda_control = lbda_control
        self.loss_nll, self.loss_confid = None, None
        self.ignore_index=ignore_index
        super().__init__()

    def forward(self, input, target):
        probs = F.softmax(input[0], dim=1)
        confidence = torch.sigmoid(input[1])

        # Make sure we don't have any numerical instability
        eps = 1e-12
        probs = torch.clamp(probs, 0.0 + eps, 1.0 - eps)
        confidence = torch.clamp(confidence, 0.0 + eps, 1.0 - eps)

        if self.half_random:
            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).to(input[0].device)
            conf = confidence * b + (1 - b)
        else:
            conf = confidence

        target[target==6] = 5
        labels_hot = misc.one_hot_embedding(target, self.nb_classes).to(input[0].device)
        # Segmentation special case
        if self.task == "segmentation":
            labels_hot = labels_hot.permute(0, 3, 1, 2)
        probs_interpol = torch.log(conf * probs + (1 - conf) * labels_hot)
        self.loss_nll = nn.NLLLoss(ignore_index=self.ignore_index)(probs_interpol, target)
        self.loss_confid = torch.mean(-(torch.log(confidence)))
        total_loss = self.loss_nll + self.lbda * self.loss_confid

        # Update lbda
        if self.lbda_control:
            if self.loss_confid >= self.beta:
                self.lbda /= 0.99
            else:
                self.lbda /= 1.01
        return total_loss

class OODConfidenceLossV2(nn.modules.loss._Loss):
    def __init__(self,nb_classes=6,half_random=False,beta=0.3,lbda=0.1,lbda_control=True,ignore_index=255):
        self.nb_classes = nb_classes
        self.task = "segmentation"
        self.half_random = half_random
        self.beta = beta
        self.lbda = lbda
        self.lbda_control = lbda_control
        self.loss_nll, self.loss_confid = None, None
        self.ignore_index=ignore_index
        super().__init__()

    def forward(self, input, target):
        probs = F.softmax(input[0], dim=1)
        confidence = torch.sigmoid(input[1])
        fullhead_confidence = torch.sigmoid(input[2])

        mean = torch.mean(fullhead_confidence)
        var = torch.var(fullhead_confidence)
        mask = torch.gt(fullhead_confidence,(mean+2*var)).int()
        

        # Make sure we don't have any numerical instability
        eps = 1e-12
        probs = torch.clamp(probs, 0.0 + eps, 1.0 - eps)
        confidence = torch.clamp(confidence, 0.0 + eps, 1.0 - eps)
        

        if self.half_random:
            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).to(input[0].device)
            conf = confidence * b + (1 - b)
        else:
            conf = confidence

        target[target==6] = 5
        labels_hot = misc.one_hot_embedding(target, self.nb_classes).to(input[0].device)
        # Segmentation special case
        if self.task == "segmentation":
            labels_hot = labels_hot.permute(0, 3, 1, 2)
        probs_interpol = torch.log(conf * probs + (1 - conf) * labels_hot)

        self.loss_nll = nn.NLLLoss(ignore_index=self.ignore_index,reduction='none')(probs_interpol, target)
        self.loss_nll = torch.mean((1-mask)*self.loss_nll)

        refine_confidence = confidence*(1-mask)+mask
        self.loss_confid = torch.mean(-(torch.log(refine_confidence)))
        total_loss = self.loss_nll + self.lbda * self.loss_confid

        # Update lbda
        if self.lbda_control:
            if self.loss_confid >= self.beta:
                self.lbda /= 0.99
            else:
                self.lbda /= 1.01
        return total_loss

class ConfidUnetFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
        self.refine_oodconfid_loss = OODConfidenceLoss(ignore_index=ignore_index)
        self.cem_oodconfid_loss = OODConfidenceLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        if self.training and len(logits) == 6:
            main_logits,refine_logits,refine_confidence, cem_logits, cem_confidence , logit_aux = logits
            main_loss1 = self.main_loss(refine_logits, labels)
            main_loss2 = self.main_loss(cem_logits, labels)
            #main_loss = self.main_loss(main_logits, labels)
            aux_loss = self.aux_loss(logit_aux, labels)
            refine_loss = self.refine_oodconfid_loss((refine_logits,refine_confidence),labels)
            cem_loss = self.cem_oodconfid_loss((cem_logits,cem_confidence),labels)

            loss = 0.5*main_loss1+0.5*main_loss2+0.5*cem_loss+0.5*refine_loss+0.4*aux_loss
        elif self.training and len(logits) == 3:
            refine_logits,refine_confidence, logit_aux = logits
            aux_loss = self.aux_loss(logit_aux, labels)
            refine_loss = self.refine_oodconfid_loss((refine_logits,refine_confidence),labels)

            loss = self.main_loss(refine_logits, labels)+0.5*refine_loss+0.4*aux_loss
            
        elif self.training and len(logits) == 2:
            logit_main, logit_aux = logits
            loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux, labels)
        else:
            loss = self.main_loss(logits, labels)

        return loss

class DualConfidLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
        self.oodconfid_loss_0 = OODConfidenceLoss(ignore_index=ignore_index)
        self.oodconfid_loss_1 = OODConfidenceLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        if self.training and len(logits) == 6:
            main_logits,score_0,confidence_0,score_1,confidence_1, logit_aux = logits
            main_loss1 = self.main_loss(score_0, labels)
            main_loss2 = self.main_loss(score_1, labels)

            aux_loss = self.aux_loss(logit_aux, labels)

            ood_0 = self.oodconfid_loss_0((score_0,confidence_0),labels)
            ood_1 = self.oodconfid_loss_1((score_1,confidence_1),labels)

            loss = 0.5*main_loss1+0.5*main_loss2+0.5*ood_0+0.5*ood_1+0.4*aux_loss

        elif self.training and len(logits) == 3:
            refine_logits,refine_confidence, logit_aux = logits
            aux_loss = self.aux_loss(logit_aux, labels)
            refine_loss = self.refine_oodconfid_loss((refine_logits,refine_confidence),labels)

            loss = self.main_loss(refine_logits, labels)+0.5*refine_loss+0.4*aux_loss
            
        elif self.training and len(logits) == 2:
            logit_main, logit_aux = logits
            loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux, labels)
        else:
            loss = self.main_loss(logits, labels)

        return loss

class VoteStageTrainingLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
        self.oodconfid_loss_0 = OODConfidenceLoss(ignore_index=ignore_index)
        self.oodconfid_loss_1 = OODConfidenceLossV2(ignore_index=ignore_index)

        self.UNKnow = 0
        self.TrainMainClassifier = 1
        self.TrainFullUncertaintyClassifier = 2
        self.TrainShadowUncertaintyClassifier  = 3
        self.mode = self.UNKnow
        
    def forward(self, logits, labels):
        loss=0
        if self.mode == self.TrainMainClassifier :
            if self.training and len(logits) == 2:
                logit_main, logit_aux = logits
                loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux, labels)
            else:
                loss = self.main_loss(logits, labels)
        
        if self.mode == self.TrainFullUncertaintyClassifier :
            if self.training and len(logits) == 2:
                score_0,confidence_0  = logits
                #main_loss0 = self.main_loss(score_0, labels)
                ood_0 = self.oodconfid_loss_0((score_0,confidence_0),labels)
                loss = ood_0
            else:
                loss = self.main_loss(logits, labels)

        if self.mode == self.TrainShadowUncertaintyClassifier:
            if self.training and len(logits) == 3:
                score_1,confidence_1,confidence_0  = logits
                #main_loss1 = self.main_loss(score_1, labels)
                ood_1 = self.oodconfid_loss_1((score_1,confidence_1,confidence_0),labels)
                loss = ood_1
            else:
                loss = self.main_loss(logits, labels)


        return loss

if __name__ == '__main__':
    targets = torch.randint(low=0, high=2, size=(2, 16, 16))
    logits = torch.randn((2, 2, 16, 16))
    # print(targets)
    model = EdgeLoss()
    loss = model.compute_edge_loss(logits, targets)

    print(loss)