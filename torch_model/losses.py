import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogCosh(nn.Module):
    def __init__(self):
        super(LogCosh, self).__init__()

    def forward(self, preds, target):

        def _logcosh(x):
            return x + F.softplus(-2. * x) - np.log(2.)

        return torch.mean(_logcosh(target - preds))


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduce=False)

        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss


class LovaszLoss(nn.Module):
    def __init__(self, activation='relu'):
        assert activation in ['relu', 'elu1p'], f'Invalid activation: {activation}'
        super(LovaszLoss, self).__init__()

        if activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = self.elu1p

    @staticmethod
    def elu1p(inputs):
        return F.elu(inputs + 1)

    @staticmethod
    def lovasz_grad(gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()

        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)

        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

        return jaccard

    def forward(self, logits, labels):
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(torch.squeeze(self.activation(errors_sorted)),
                         torch.squeeze(grad))
        return loss
