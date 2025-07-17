
from .base import BaseSampling
from typing import Dict

class DeterministicSampling(BaseSampling):
    """
    Standard deterministic forward pass.
    """
    def forward(self, 
                model_forward_callable, 
                x_enc, x_mark_enc, 
                x_dec, x_mark_dec, 
                detailed=False) -> Dict:
        """
        Performs a single, deterministic forward pass.
        """
        prediction = model_forward_callable(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return {'prediction': prediction}
