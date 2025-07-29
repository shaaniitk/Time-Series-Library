import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding
from utils.logger import logger

# Safe Mamba import with fallback
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    logger.warning("mamba_ssm not found in Mamba.py. This is normal on non-GPU systems (like macOS). "
                   "Using a standard LSTM as a fallback. Performance will differ.")

    class Mamba(nn.Module):
        """A fallback implementation using LSTM when mamba_ssm is not available."""
        def __init__(self, d_model, d_state, d_conv, expand):
            super().__init__()
            self.lstm = nn.LSTM(d_model, d_model, batch_first=True)

        def forward(self, x):
            output, _ = self.lstm(x)
            return output

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.d_inner = configs.d_model * configs.expand
        self.dt_rank = math.ceil(configs.d_model / 16) # TODO implement "auto"
        
        if not MAMBA_AVAILABLE:
            logger.warning("Mamba.py: Using LSTM fallback. Mamba-specific parameters "
                           "(d_state, d_conv, expand) will be ignored by the LSTM layer.")
        else:
            logger.info("Mamba.py: Using official mamba_ssm implementation.")

        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.mamba = Mamba(
            d_model = configs.d_model,
            d_state = configs.d_ff,
            d_conv = configs.d_conv,
            expand = configs.expand,
        )

        self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x)
        x_out = self.out_layer(x)

        x_out = x_out * std_enc + mean_enc
        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]

        # other tasks not implemented