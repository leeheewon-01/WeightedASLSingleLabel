import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedASLSingleLabel(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean', weights=None):
        super(WeightedASLSingleLabel, self).__init__()
        self.eps = eps
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        self.class_weights = weights

    def forward(self, inputs, target):
        num_classes = inputs.size()[-1]
        log_preds = F.log_softmax(inputs, dim=-1)
        targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        anti_targets = 1 - targets_classes
        xs_pos = torch.exp(log_preds) * targets_classes
        xs_neg = (1 - torch.exp(log_preds)) * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg, self.gamma_pos * targets_classes + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        # label smoothing
        if self.eps > 0:
            targets_classes = targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = F.nll_loss(log_preds, target, reduction=self.reduction, weight=self.class_weights)

        return loss
