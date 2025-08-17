import torch.nn as nn
import torch.nn.functional as F
from .abstract_layers import BaseEncoderLayer, BaseDecoderLayer
from .common import FeedForward

class StandardEncoderLayer(BaseEncoderLayer):
    """
    The standard Autoformer encoder layer.
    """
    def __init__(self, attention_component, decomposition_component, d_model, n_heads, d_ff, dropout=0.1, activation="relu"):
        super(StandardEncoderLayer, self).__init__()
        self.attention = attention_component
        self.decomp1 = decomposition_component
        self.decomp2 = decomposition_component
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        x = self.feed_forward(x)
        res, _ = self.decomp2(x)
        return res, attn

class StandardDecoderLayer(BaseDecoderLayer):
    """
    The standard Autoformer decoder layer.
    """
    def __init__(self, self_attention_comp, cross_attention_comp, decomposition_comp, d_model, c_out, d_ff=None, dropout=0.1, activation="relu"):
        super(StandardDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention_comp
        self.cross_attention = cross_attention_comp
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = decomposition_component
        self.decomp2 = decomposition_component
        self.decomp3 = decomposition_comp
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)
        residual_trend = trend1 + trend2 + trend3
        return x, residual_trend