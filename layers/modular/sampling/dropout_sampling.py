
import torch
from .base import BaseSampling
from typing import Dict
from utils.logger import logger

class DropoutSampling(BaseSampling):
    """
    Performs forward pass with Monte Carlo dropout for uncertainty estimation.
    """
    def __init__(self, n_samples=50):
        super(DropoutSampling, self).__init__()
        self.n_samples = n_samples

    def forward(self, 
                model_forward_callable, 
                x_enc, x_mark_enc, 
                x_dec, x_mark_dec, 
                detailed=False) -> Dict:
        logger.debug(f"Computing MC Dropout uncertainty with {self.n_samples} samples")
        
        model = model_forward_callable.__self__
        model.train() # Enable dropout layers
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = model_forward_callable(x_enc, x_mark_enc, x_dec, x_mark_dec)
                predictions.append(pred)
        
        model.eval() # Disable dropout layers
        
        pred_stack = torch.stack(predictions)
        return self._compute_uncertainty_statistics(pred_stack, detailed)

    def _compute_uncertainty_statistics(self, pred_stack, detailed=False):
        mean_pred = torch.mean(pred_stack, dim=0)
        total_variance = torch.var(pred_stack, dim=0)
        total_std = torch.sqrt(total_variance)
        
        return {
            'prediction': mean_pred,
            'uncertainty': total_std,
            'variance': total_variance,
            'predictions_samples': pred_stack if detailed else None
        }
