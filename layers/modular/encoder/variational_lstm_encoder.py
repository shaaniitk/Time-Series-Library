import torch
import torch.nn as nn
from typing import Optional, Tuple

from .base_encoder import BaseEncoder
from layers.VariationalLSTM import VariationalLSTM


class VariationalLSTMEncoder(BaseEncoder):
    """
    Encoder wrapper around layers.VariationalLSTM following the modular encoder API.

    Args:
        d_model (int): Input (and default hidden) dimension.
        hidden_size (int, optional): Hidden size of the LSTM. Defaults to d_model.
        num_layers (int, optional): Number of LSTM layers. Defaults to 2.
        dropout (float, optional): Dropout between LSTM layers. Defaults to 0.1.
        prior_std (float, optional): Prior std for variational parameters. Defaults to 1.0.
        variational_dropout (bool, optional): Use variational dropout masks. Defaults to True.
    """

    def __init__(
        self,
        d_model: int,
        hidden_size: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        prior_std: float = 1.0,
        variational_dropout: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size or d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.prior_std = prior_std
        self.variational_dropout = variational_dropout

        self.vlstm = VariationalLSTM(
            input_size=d_model,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            prior_std=prior_std,
            variational_dropout=variational_dropout,
        )
        self.last_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.last_kl: Optional[torch.Tensor] = None

        # If hidden size != d_model, project back to d_model to keep downstream dims consistent
        self.proj: Optional[nn.Module]
        if self.hidden_size != d_model:
            self.proj = nn.Linear(self.hidden_size, d_model)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, L, D]
            attention_mask: Optional mask [B, L] (ignored by LSTM)
        Returns:
            Tensor of shape [B, L, D] (or projected to D if hidden_size != D)
        """
        out, hidden, kl = self.vlstm(x)
        self.last_hidden = hidden
        self.last_kl = kl
        if self.proj is not None:
            out = self.proj(out)
        return out

    # Optional convenience accessor
    def get_last_kl(self) -> Optional[torch.Tensor]:
        return self.last_kl