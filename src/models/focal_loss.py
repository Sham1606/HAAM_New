import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma  : focusing parameter ≥ 0. Higher → more focus on hard examples.
        alpha  : class weights tensor of shape [n_classes] or None.
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__()
        self.gamma          = gamma
        self.alpha          = alpha          # tensor [C] or None
        self.reduction      = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs  : [B, C] raw logits
        targets : [B]    integer class indices
        """
        n_classes = inputs.size(1)

        # Label smoothing: blend one-hot with uniform
        if self.label_smoothing > 0.0:
            with torch.no_grad():
                smooth_val = self.label_smoothing / n_classes
                one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)
                smooth_targets = one_hot * (1.0 - self.label_smoothing) + smooth_val
            log_prob = F.log_softmax(inputs, dim=1)                     # [B, C]
            ce_loss  = -(smooth_targets * log_prob).sum(dim=1)          # [B]

            # For focal weighting we still use the hard-label p_t
            prob    = F.softmax(inputs.detach(), dim=1)                 # [B, C]
            p_t     = prob.gather(1, targets.unsqueeze(1)).squeeze(1)   # [B]
        else:
            log_prob = F.log_softmax(inputs, dim=1)
            ce_loss  = F.nll_loss(log_prob, targets, reduction='none')  # [B]
            prob     = torch.exp(log_prob)
            p_t      = prob.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]

        # Focal weight
        focal_weight = (1.0 - p_t) ** self.gamma                       # [B]

        # Alpha (class) weighting
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]             # [B]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss                                   # [B]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
