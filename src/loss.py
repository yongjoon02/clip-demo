import torch
import torch.nn.functional as F

def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    b = logits_per_image.size(0)
    target = torch.arange(b, device=logits_per_image.device)
    return 0.5 * (F.cross_entropy(logits_per_image, target) +
                  F.cross_entropy(logits_per_text,  target))