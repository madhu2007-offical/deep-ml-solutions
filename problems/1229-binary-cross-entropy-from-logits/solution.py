
import torch

def bce_with_logits(logits, targets):
    loss = torch.clamp(logits, min=0) - logits * targets + torch.log1p(torch.exp(-torch.abs(logits)))
    return round(loss.mean().item(), 4)
