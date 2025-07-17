
import torch.nn as nn

class StandardLossWrapper(nn.Module):
    """
    A wrapper for standard PyTorch loss functions like MSELoss.
    """
    def __init__(self, loss_class):
        super(StandardLossWrapper, self).__init__()
        self.loss = loss_class()
        # This component declares that it does not change the output dimension.
        self.output_dim_multiplier = 1

    def forward(self, preds, target):
        return self.loss(preds, target)
