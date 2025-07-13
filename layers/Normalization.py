import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm * self.eps) * self.weight

def get_norm_layer(norm_type, d_model):
    if norm_type == 'layernorm':
        return nn.LayerNorm(d_model)
    elif norm_type == 'rmsnorm':
        return RMSNorm(d_model)
    else:
        raise ValueError(f'Unknown norm_type: {norm_type}')
