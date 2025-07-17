
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict

class BaseSampling(nn.Module, ABC):
    """
    Abstract base class for all sampling components.
    """
    def __init__(self):
        super(BaseSampling, self).__init__()

    @abstractmethod
    def forward(self, 
                model_forward_callable, 
                x_enc, x_mark_enc, 
                x_dec, x_mark_dec, 
                detailed=False) -> Dict:
        """
        The forward pass for the sampling mechanism.

        Args:
            model_forward_callable (callable): A callable that executes the model's single forward pass.
            x_enc, x_mark_enc, x_dec, x_mark_dec: The model inputs.
            detailed (bool, optional): Whether to return detailed outputs (e.g., all samples). Defaults to False.

        Returns:
            dict: A dictionary containing at least the 'prediction'.
        """
        pass
