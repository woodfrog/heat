import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
import scipy.ndimage.filters as filters
import numpy as np
from utils.geometry_utils import edge_acc

class CornerCriterion(nn.Module):

    def __init__(self, image_size):
        super().__init__()
        self.gamma = 1
        self.loss_rate = 9

    # def forward(self, outputs_s1, outputs_s2, s2_mask, s2_candidates, targets, gauss_targets, epoch=0):
    def forward(self, outputs_s1, targets, gauss_targets, epoch=0):
        # Compute the acc first, use the acc to guide the setup of loss weight
        preds_s1 = (outputs_s1 >= 0.5).float()
        pos_target_ids = torch.where(targets == 1)
        correct = (preds_s1[pos_target_ids] == targets[pos_target_ids]).float().sum()
        # acc = correct / (preds.shape[0] * preds.shape[1] * preds.shape[2])
        #num_pos_preds = (preds == 1).sum()
        recall_s1 = correct / len(pos_target_ids[0])
        #prec = correct / num_pos_preds if num_pos_preds > 0 else torch.tensor(0).to(correct.device)
        #f_score = 2.0 * prec * recall / (recall + prec + 1e-8)

        rate = self.loss_rate

        loss_weight = (gauss_targets > 0.5).float() * rate + 1
        loss_s1 = F.binary_cross_entropy(outputs_s1, gauss_targets, weight=loss_weight, reduction='none')
        loss_s1 = loss_s1.sum(-1).sum(-1).mean()

        # loss for stage-2
        # B, H, W = gauss_targets.shape
        # gauss_targets_1d = gauss_targets.view(B, H*W)
        # s2_ids = s2_candidates[:, :, 1] * H + s2_candidates[:, :, 0]
        # s2_labels = torch.gather(gauss_targets_1d, 1, s2_ids)
        # # try an aggressive labeling for s2
        # s2_th = 0.1
        # s2_labels = (s2_labels > s2_th).float()
        # loss_weight = (s2_labels > 0.5).float() * rate + 1
        # loss_s2 = F.binary_cross_entropy(outputs_s2, s2_labels, weight=loss_weight, reduction='none')
        # loss_s2[torch.where(s2_mask == True)] = 0
        # loss_s2 = loss_s2.sum(-1).sum(-1).mean()

        return loss_s1, recall_s1 
        #, loss_s2, recall_s1



class CornerCriterion4D(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, corner_labels, offset_labels):
        loss_dict = {'loss_jloc': 0.0, 'loss_joff': 0.0}
        for _, output in enumerate(outputs):
            loss_dict['loss_jloc'] += cross_entropy_loss_for_junction(output[:,:2], corner_labels)
            loss_dict['loss_joff'] += sigmoid_l1_loss(output[:,2:4], offset_labels, -0.5, corner_labels.float())
        return loss_dict


def cross_entropy_loss_for_junction(logits, positive):
    nlogp = -F.log_softmax(logits, dim=1)
    loss = (positive * nlogp[:, None, 1] + (1 - positive) * nlogp[:, None, 0])
    pos_rate = 4
    weights = (positive == 1) * pos_rate + 1
    loss = loss * weights

    loss = loss.mean()
    return loss


def sigmoid_l1_loss(logits, targets, offset = 0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp-targets)

    if mask is not None:
        w = mask.mean(3, True).mean(2,True)
        w[w==0] = 1
        loss = loss*(mask/w)

    loss = loss.mean()  # avg over batch dim
    return loss


class EdgeCriterion(nn.Module):

    def __init__(self):
        super().__init__()
        self.edge_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.33, 1.0]).cuda(), reduction='none')
        self.gamma = 1.0 # used by focal loss (if enabled)

    def forward(self, logits_s1, logits_s2_hybrid, logits_s2_rel, s2_ids, s2_edge_mask, edge_labels, edge_lengths, edge_mask, s2_gt_values):
        # loss for stage-1: edge filtering
        s1_losses = self.edge_loss(logits_s1, edge_labels)
        s1_losses[torch.where(edge_mask == True)] = 0
        s1_losses = s1_losses[torch.where(s1_losses > 0)].sum() / edge_mask.shape[0]
        gt_values = torch.ones_like(edge_mask).long() * 2
        s1_acc = edge_acc(logits_s1, edge_labels, edge_lengths, gt_values)

        # loss for stage-2
        s2_labels = torch.gather(edge_labels, 1, s2_ids)

        s2_losses_hybrid = self.edge_loss(logits_s2_hybrid, s2_labels)
        s2_losses_hybrid[torch.where((s2_edge_mask == True) | (s2_gt_values !=2))] = 0
        # aggregate the loss into the final scalar
        s2_losses_hybrid = s2_losses_hybrid[torch.where(s2_losses_hybrid > 0)].sum() / s2_edge_mask.shape[0]
        s2_edge_lengths = (s2_edge_mask == 0).sum(dim=-1)
        # compute edge-level f1-score
        s2_acc_hybrid = edge_acc(logits_s2_hybrid, s2_labels, s2_edge_lengths, s2_gt_values)

        s2_losses_rel = self.edge_loss(logits_s2_rel, s2_labels)
        s2_losses_rel[torch.where((s2_edge_mask == True) | (s2_gt_values != 2))] = 0
        # aggregate the loss into the final scalar
        s2_losses_rel = s2_losses_rel[torch.where(s2_losses_rel > 0)].sum() / s2_edge_mask.shape[0]
        s2_edge_lengths = (s2_edge_mask == 0).sum(dim=-1)
        # compute edge-level f1-score
        s2_acc_rel = edge_acc(logits_s2_rel, s2_labels, s2_edge_lengths, s2_gt_values)

        return s1_losses, s1_acc, s2_losses_hybrid, s2_acc_hybrid, s2_losses_rel, s2_acc_rel

