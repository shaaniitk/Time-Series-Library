
import torch
from .base import BaseSampling
from typing import Dict
from utils.logger import logger

class BayesianSampling(BaseSampling):
    """
    Performs forward pass with Bayesian sampling for uncertainty estimation.
    """
    def __init__(self, n_samples=50, quantile_levels=None):
        super(BayesianSampling, self).__init__()
        print(f"--- [DEBUG] BayesianSampling __init__: received quantile_levels = {quantile_levels}")
        self.n_samples = n_samples
        self.quantile_levels = quantile_levels

    def forward(self, 
                model_forward_callable, 
                x_enc, x_mark_enc, 
                x_dec, x_mark_dec, 
                detailed=False) -> Dict:
        logger.debug(f"Computing Bayesian uncertainty with {self.n_samples} samples")
        
        # The model that will be used for sampling
        model = model_forward_callable.__self__
        
        # Temporarily set the quantile levels on the model if they are provided
        original_quantile_levels = None
        if self.quantile_levels and hasattr(model, 'set_quantile_levels'):
            original_quantile_levels = model.get_quantile_levels()
            model.set_quantile_levels(self.quantile_levels)

        predictions = []
        
        grad_context = torch.enable_grad if model.training else torch.no_grad

        with grad_context():
            for _ in range(self.n_samples):
                pred = model_forward_callable(x_enc, x_mark_enc, x_dec, x_mark_dec)
                predictions.append(pred)

        pred_stack = torch.stack(predictions)
        
        if not model.training:
            pred_stack = pred_stack.detach()

        # Restore original quantile levels if they were changed
        if original_quantile_levels is not None and hasattr(model, 'set_quantile_levels'):
            model.set_quantile_levels(original_quantile_levels)

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
