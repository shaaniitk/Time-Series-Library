"""
Variational LSTM with Bayesian uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class VariationalLSTMCell(nn.Module):
    """LSTM cell with variational weights for Bayesian uncertainty"""
    
    def __init__(self, input_size: int, hidden_size: int, prior_std: float = 1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prior_std = prior_std
        self.register_buffer('prior_std_tensor', torch.tensor(prior_std))
        
        # Variational parameters for weights
        self.weight_mu = nn.Parameter(torch.randn(4 * hidden_size, input_size + hidden_size) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(4 * hidden_size, input_size + hidden_size) * 0.1)
        
        # Variational parameters for bias
        self.bias_mu = nn.Parameter(torch.zeros(4 * hidden_size))
        self.bias_logvar = nn.Parameter(torch.randn(4 * hidden_size) * 0.1)
        
    def forward(self, input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input: [B, input_size]
            hidden: (h_prev, c_prev) each [B, hidden_size]
        Returns:
            h_new, c_new, kl_loss
        """
        h_prev, c_prev = hidden
        
        # Sample weights from variational distribution
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight_eps = torch.randn_like(weight_std)
        weight = self.weight_mu + weight_eps * weight_std
        
        # Sample bias from variational distribution
        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias_eps = torch.randn_like(bias_std)
        bias = self.bias_mu + bias_eps * bias_std
        
        # LSTM computation with sampled parameters
        combined = torch.cat([input, h_prev], dim=1)
        gates = F.linear(combined, weight, bias)
        
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        
        c_new = f_gate * c_prev + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)
        
        # Compute KL divergence
        kl_loss = self._compute_kl_divergence()
        
        return h_new, c_new, kl_loss
        
    def _compute_kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between variational and prior distributions"""
        # KL for weights
        weight_var = torch.exp(self.weight_logvar)
        kl_weight = 0.5 * torch.sum(
            self.weight_mu**2 / self.prior_std_tensor**2 + 
            weight_var / self.prior_std_tensor**2 - 
            self.weight_logvar - 
            torch.log(self.prior_std_tensor**2)
        )
        
        # KL for bias
        bias_var = torch.exp(self.bias_logvar)
        kl_bias = 0.5 * torch.sum(
            self.bias_mu**2 / self.prior_std_tensor**2 + 
            bias_var / self.prior_std_tensor**2 - 
            self.bias_logvar - 
            torch.log(self.prior_std_tensor**2)
        )
        
        return kl_weight + kl_bias


class VariationalLSTM(nn.Module):
    """Multi-layer Variational LSTM with variational dropout"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, 
                 dropout: float = 0.1, prior_std: float = 1.0, variational_dropout: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.variational_dropout = variational_dropout
        self.dropout_rate = dropout
        
        self.cells = nn.ModuleList([
            VariationalLSTMCell(input_size if i == 0 else hidden_size, hidden_size, prior_std)
            for i in range(num_layers)
        ])
        
        if not variational_dropout:
            self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Args:
            x: [B, L, input_size]
            hidden: Initial hidden state
        Returns:
            output: [B, L, hidden_size]
            final_hidden: (h_final, c_final)
            total_kl_loss: KL divergence loss
        """
        B, L, _ = x.shape
        
        if hidden is None:
            h = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            c = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        else:
            h, c = hidden
            h = [h[i] for i in range(self.num_layers)]
            c = [c[i] for i in range(self.num_layers)]
        
        outputs = []
        total_kl_loss = 0
        
        # Generate dropout masks for variational dropout (same mask across time)
        if self.variational_dropout and self.training:
            dropout_masks = []
            for layer_idx in range(self.num_layers - 1):
                mask = torch.bernoulli(torch.full((B, self.hidden_size), 1 - self.dropout_rate, device=x.device))
                dropout_masks.append(mask / (1 - self.dropout_rate))  # Scale for training
        
        for t in range(L):
            layer_input = x[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx], kl_loss = cell(layer_input, (h[layer_idx], c[layer_idx]))
                total_kl_loss += kl_loss
                
                if layer_idx < self.num_layers - 1:
                    if self.variational_dropout and self.training:
                        layer_input = h[layer_idx] * dropout_masks[layer_idx]
                    else:
                        layer_input = self.dropout(h[layer_idx]) if hasattr(self, 'dropout') else h[layer_idx]
                else:
                    layer_input = h[layer_idx]
            
            outputs.append(layer_input)
        
        output = torch.stack(outputs, dim=1)
        final_hidden = (torch.stack(h), torch.stack(c))
        
        return output, final_hidden, total_kl_loss / L  # Average KL loss over sequence