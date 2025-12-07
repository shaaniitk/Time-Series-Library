import torch
from .base import CurriculumLearning
from typing import Optional, Dict, Any

class HorizonCurriculum(CurriculumLearning):
    """
    Horizon-based Curriculum:
    Gradually increases the effective prediction length (horizon) over epochs.
    Starts with 'min_curriculum_ratio' of pred_len, expands to 100%.
    """
    def __init__(self, config, device):
        super().__init__(config, device)
        self.min_ratio = getattr(config, 'min_curriculum_ratio', 0.2)
        self.current_effective_len = config.pred_len # Default to full
        
    def update_epoch(self, epoch: int):
        super().update_epoch(epoch)
        # Calculate ratio
        progress = epoch / max(1, self.total_epochs - 1)
        curriculum_ratio = self.min_ratio + (1.0 - self.min_ratio) * progress
        curriculum_ratio = min(1.0, max(self.min_ratio, curriculum_ratio))
        
        # Calculate effective length
        full_pred_len = self.config.pred_len
        self.current_effective_len = int(full_pred_len * curriculum_ratio)
        self.current_effective_len = max(1, self.current_effective_len)
        self.current_ratio = curriculum_ratio

    def get_mask(self, pred_len: int) -> Optional[torch.Tensor]:
        mask = torch.zeros(pred_len, device=self.device)
        mask[:self.current_effective_len] = 1.0
        return mask

    def get_stats(self) -> Dict[str, Any]:
        return {
            "strategy": "HorizonCurriculum",
            "effective_len": self.current_effective_len,
            "ratio": f"{self.current_ratio:.2f}"
        }
