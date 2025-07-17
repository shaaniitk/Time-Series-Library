
import torch
import torch.nn as nn

class PinballLoss(nn.Module):
    """
    Computes the pinball loss for quantile regression.
    """
    def __init__(self, quantiles):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles
        # This component declares that it requires the model's output dimension
        # to be multiplied by the number of quantiles.
        self.output_dim_multiplier = len(quantiles)

    def forward(self, preds, target):
        """
        preds: [Batch, Seq_Len, N_Targets * N_Quantiles]
        target: [Batch, Seq_Len, N_Targets]
        """
        # Reshape predictions to separate quantiles
        preds = preds.view(*target.shape[:-1], -1, len(self.quantiles))
        # Expand targets to match quantile structure
        target = target.unsqueeze(-1).expand_as(preds)

        error = target - preds
        loss = torch.max((torch.tensor(self.quantiles, device=preds.device) - 1) * error, torch.tensor(self.quantiles, device=preds.device) * error)
        return loss.mean()
