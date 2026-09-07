"""Teacher-value weighting, with update-wide normalization for accumulation."""

import torch
import torch.nn.functional as F


def weighted_value_losses(logits, result, value, minimum):
    q = value.detach().float()
    weights = minimum + (1 - minimum) * 4 * q * (1 - q)
    loss_result = F.binary_cross_entropy_with_logits(logits.float(), result.float(), reduction='none')
    loss_value = F.binary_cross_entropy_with_logits(logits.float(), q, reduction='none')
    return (loss_result * weights).mean(), (loss_value * weights).mean(), weights.mean()


def positive_denominator(weight):
    # All-zero weights contribute no value gradient, without changing policy loss.
    return torch.where(weight > 0, weight, torch.ones_like(weight))
