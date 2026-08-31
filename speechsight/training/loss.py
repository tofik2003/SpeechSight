"""
Loss functions for Audio-Visual Speech Recognition (AVSR).
Implements CTC Loss and Contrastive Audio-Visual Synchrony Loss.
"""

from typing import Optional
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class AVSRLoss(nn.Module):
        def __init__(self, blank_idx: int = 0, av_sync_weight: float = 0.1):
            super().__init__()
            self.ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
            self.av_sync_weight = av_sync_weight

        def forward(
            self,
            log_probs: torch.Tensor,
            targets: torch.Tensor,
            input_lengths: torch.Tensor,
            target_lengths: torch.Tensor,
            v_embed: Optional[torch.Tensor] = None,
            a_embed: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            # log_probs: (T, B, C)
            ctc_l = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
            
            total_loss = ctc_l
            if v_embed is not None and a_embed is not None and self.av_sync_weight > 0:
                # Contrastive cosine similarity loss for AV sync
                sim = F.cosine_similarity(v_embed, a_embed, dim=-1)
                sync_loss = 1.0 - sim.mean()
                total_loss = total_loss + self.av_sync_weight * sync_loss

            return total_loss
else:
    class AVSRLoss:
        def __init__(self, *args, **kwargs):
            pass
