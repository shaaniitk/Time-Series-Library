
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
        Compute mean pinball (quantile) loss.

        Parameters
        ----------
        preds : torch.Tensor
            Either shape [B, L, T * Q] (flattened quantiles) or [B, L, T, Q].
        target : torch.Tensor
            Shape [B, L, T].
        """
        if preds.ndim == 3:
            # Flattened layout: [B, L, T * Q]
            B, L, _ = preds.shape
            T = target.shape[-1]
            Q = len(self.quantiles)
            if _ != T * Q:
                raise ValueError(f"Flattened predictions last dim {_} not equal to targets*T*Q={T*Q}")
            preds = preds.view(B, L, T, Q)
        elif preds.ndim == 4:
            # Already separated quantiles: [B, L, T, Q]
            B, L, T, Q = preds.shape
            if Q != len(self.quantiles):
                raise ValueError(f"Provided Q={Q} does not match configured quantiles {len(self.quantiles)}")
        else:
            raise ValueError(f"Unsupported prediction ndim={preds.ndim}; expected 3 or 4.")

        if target.shape != preds.shape[:3]:
            raise ValueError(f"Target shape {target.shape} not broadcastable to preds base {preds.shape[:3]}")

        # Expand targets to quantile dimension
        target_exp = target.unsqueeze(-1).expand_as(preds)
        error = target_exp - preds  # positive if target > pred

        # quantiles tensor shape [Q]
        q = torch.tensor(self.quantiles, device=preds.device).view(1, 1, 1, -1)
        # Pinball loss per element
        loss = torch.maximum((q - 1) * error, q * error)
        return loss.mean()
