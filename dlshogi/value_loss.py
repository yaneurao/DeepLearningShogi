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


class ValueGradientAccumulator:
    def __init__(self, parameters):
        self.parameters = tuple(p for p in parameters if p.requires_grad)
        self.gradients = [None] * len(self.parameters)
        self.weight = None

    def backward(self, policy_loss, value_loss, weight, batches, scaler):
        # Keep only gradients, not previous micro-batches or their computation graphs.
        gradients = torch.autograd.grad(scaler.scale(value_loss / batches),
                                        self.parameters, retain_graph=True, allow_unused=True)
        for i, gradient in enumerate(gradients):
            if gradient is not None:
                if self.gradients[i] is None:
                    self.gradients[i] = gradient.detach()
                else:
                    self.gradients[i].add_(gradient.detach())
        weight = weight.detach()
        self.weight = weight if self.weight is None else self.weight + weight
        scaler.scale(policy_loss / batches).backward()

    def finish(self, batches):
        factor = batches / positive_denominator(self.weight)
        for parameter, gradient in zip(self.parameters, self.gradients):
            if gradient is not None:
                gradient.mul_(factor)
                if parameter.grad is None:
                    parameter.grad = gradient
                else:
                    parameter.grad.add_(gradient)
        self.gradients = []
        return factor.item()
