import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_layers import BaseEncoderLayer, BaseDecoderLayer
from .common import FeedForward

class EnhancedEncoderLayer(BaseEncoderLayer):
    """
    Enhanced Autoformer encoder layer with improved decomposition and attention.
    """
    def __init__(self, attention_component, decomposition_component, d_model, n_heads, d_ff, dropout=0.1, activation="relu"):
        super(EnhancedEncoderLayer, self).__init__()
        self.attention = attention_component
        self.decomp1 = decomposition_component
        self.decomp2 = decomposition_component
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.gate_linear = nn.Linear(d_model, d_model)
        self.gate_act = nn.Sigmoid()
        self.attention_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.ffn_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(self.attention_scale * new_x)
        
        # Support decomposition modules that return either (residual, trend) or only trend
        decomp_out = self.decomp1(x)
        if isinstance(decomp_out, tuple):
            x, _ = decomp_out
        else:
            trend = decomp_out
            x = x - trend if trend.shape == x.shape else x
        
        residual = x
        # Lightweight gating on feature dimension to avoid layout pitfalls
        gate_values = self.gate_act(self.gate_linear(x))
        y = x * gate_values
        
        y = self.feed_forward(y)
        
        x = residual + self.ffn_scale * y
        
        res, _ = self.decomp2(x)
        
        return res, attn

class EnhancedDecoderLayer(BaseDecoderLayer):
    """
    Enhanced Autoformer decoder layer with advanced decomposition.
    """
    def __init__(self, self_attention_comp, cross_attention_comp, decomposition_comp, d_model, c_out, d_ff=None, dropout=0.1, activation="relu"):
        super(EnhancedDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention_comp
        self.cross_attention = cross_attention_comp
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        
        self.decomp1 = decomposition_comp
        self.decomp2 = decomposition_comp
        self.decomp3 = decomposition_comp
        
        self.dropout = nn.Dropout(dropout)
        
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=False)
        )
        
        self.activation = getattr(F, activation, F.relu)
        
        self.self_attn_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.cross_attn_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        residual = x
        new_x = self.self_attention(x, x, x, attn_mask=x_mask)[0]
        x = residual + self.dropout(self.self_attn_scale * new_x)
        # decomp1 may return trend only or (residual, trend)
        out1 = self.decomp1(x)
        if isinstance(out1, tuple):
            x, trend1 = out1
        else:
            trend1 = out1
            x = x - trend1 if trend1.shape == x.shape else x
        
        residual = x
        cross_tensor = self._extract_cross_memory(cross)
        # Normalize cross tensor shape to [B, L, D]
        import torch as _torch
        if _torch.is_tensor(cross_tensor):
            if cross_tensor.dim() == 2:
                cross_tensor = cross_tensor.unsqueeze(0)
            if cross_tensor.size(0) != x.size(0):
                if cross_tensor.size(0) == 1:
                    cross_tensor = cross_tensor.expand(x.size(0), -1, -1)
                else:
                    cross_tensor = cross_tensor[:x.size(0)]
        new_x = self.cross_attention(x, cross_tensor, cross_tensor, attn_mask=cross_mask)[0]
        x = residual + self.dropout(self.cross_attn_scale * new_x)
        
        out2 = self.decomp2(x)
        if isinstance(out2, tuple):
            x, trend2 = out2
        else:
            trend2 = out2
            x = x - trend2 if trend2.shape == x.shape else x
        
        residual = x
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = residual + y
        
        out3 = self.decomp3(x)
        if isinstance(out3, tuple):
            x, trend3 = out3
        else:
            trend3 = out3
            x = x - trend3 if trend3.shape == x.shape else x

        residual_trend = trend1 + trend2 + trend3 if (isinstance(trend1, type(trend2)) and isinstance(trend2, type(trend3))) else 0
        return x, residual_trend

    @staticmethod
    def _extract_cross_memory(cross):
        """Extract tensor memory from various encoder output forms.

        Supported forms:
        - Tensor
        - (tensor, attn_weights, ...)
        - Objects with `.seasonal` attribute (use that attribute)
        - List/tuple containing any of the above; first tensor-like element chosen
        Raises TypeError if no tensor-like content found.
        """
        import torch as _torch
        if _torch.is_tensor(cross):
            return cross
        # Structured decoder/encoder output objects
        if hasattr(cross, 'seasonal') and _torch.is_tensor(getattr(cross, 'seasonal')):  # DecoderOutput-like
            return getattr(cross, 'seasonal')
        # Tuple/list: search for first viable tensor or object with seasonal
        if isinstance(cross, (tuple, list)):
            for elem in cross:
                if _torch.is_tensor(elem):
                    return elem
                if hasattr(elem, 'seasonal') and _torch.is_tensor(getattr(elem, 'seasonal')):
                    return getattr(elem, 'seasonal')
        raise TypeError(f"Unsupported cross memory type: {type(cross)}")