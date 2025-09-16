import torch
import torch.nn as nn
import math

class SinusoidalTemporalEncoding(nn.Module):
    """
    Enhanced sinusoidal temporal positional encoding for time series data.
    Provides rich temporal position information before attention mechanisms.
    """
    
    def __init__(self, d_model, max_len=None, max_seq_len=5000, temperature=10000.0):
        super().__init__()
        self.d_model = d_model
        # Support both max_len (preferred) and max_seq_len (backward compat)
        self.max_len = max_len if max_len is not None else max_seq_len
        self.temperature = temperature
        
        # Pre-compute positional encodings
        pe = torch.zeros(self.max_len, d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        
        # Create sinusoidal patterns with different frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(temperature) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
        
    def forward(self, x):
        """
        Add sinusoidal positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        # Ensure we don't exceed pre-computed length
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_len}")
            
        # Add positional encoding
        pos_encoding = self.pe[:, :seq_len, :d_model]
        return x + pos_encoding

class AdaptiveTemporalEncoding(nn.Module):
    """
    Adaptive temporal encoding that learns to adjust positional information
    based on the temporal patterns in the data.
    """
    
    def __init__(self, d_model, max_len=None, max_seq_len=5000, num_scales=4):
        super().__init__()
        self.d_model = d_model
        # Support both max_len (preferred) and max_seq_len (backward compat)
        self.max_len = max_len if max_len is not None else max_seq_len
        self.num_scales = num_scales
        
        # Learnable scale parameters for different temporal frequencies
        self.scale_params = nn.Parameter(torch.ones(num_scales))
        
        # Base sinusoidal encoding
        self.base_encoding = SinusoidalTemporalEncoding(d_model, max_len=self.max_len)
        
        # Adaptive projection layers
        self.temporal_projection = nn.Linear(d_model, d_model)
        self.scale_attention = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        
    def forward(self, x):
        """
        Apply adaptive temporal encoding.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with adaptive temporal encoding
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get base sinusoidal encoding
        base_encoded = self.base_encoding(x)
        
        # Create multi-scale temporal features
        multi_scale_features = []
        for i, scale in enumerate(self.scale_params):
            # Scale the positional information
            scaled_pos = torch.arange(seq_len, device=x.device).float() * scale
            
            # Create scaled sinusoidal patterns
            div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() * 
                               (-math.log(10000.0) / d_model))
            
            scaled_encoding = torch.zeros(seq_len, d_model, device=x.device)
            scaled_encoding[:, 0::2] = torch.sin(scaled_pos.unsqueeze(1) * div_term)
            scaled_encoding[:, 1::2] = torch.cos(scaled_pos.unsqueeze(1) * div_term)
            
            multi_scale_features.append(scaled_encoding.unsqueeze(0).expand(batch_size, -1, -1))
        
        # Combine multi-scale features using attention
        stacked_features = torch.stack(multi_scale_features, dim=1)  # [batch, num_scales, seq_len, d_model]
        stacked_features = stacked_features.view(batch_size * self.num_scales, seq_len, d_model)
        
        # Apply self-attention to learn optimal scale combination
        attended_features, _ = self.scale_attention(stacked_features, stacked_features, stacked_features)
        attended_features = attended_features.view(batch_size, self.num_scales, seq_len, d_model)
        
        # Aggregate across scales
        adaptive_encoding = attended_features.mean(dim=1)  # [batch, seq_len, d_model]
        
        # Project and combine with input
        projected_encoding = self.temporal_projection(adaptive_encoding)
        
        return x + projected_encoding

class EnhancedTemporalEncoding(nn.Module):
    """
    Combined enhanced temporal encoding that uses both sinusoidal and adaptive components.
    """
    
    def __init__(self, d_model, max_len=None, max_seq_len=5000, num_scales=4, use_adaptive=True):
        super().__init__()
        self.d_model = d_model
        self.use_adaptive = use_adaptive
        # Support both max_len (preferred) and max_seq_len (backward compat)
        self.max_len = max_len if max_len is not None else max_seq_len
        self.num_scales = num_scales
        
        # Base sinusoidal encoding
        self.sinusoidal_encoding = SinusoidalTemporalEncoding(d_model, max_len=self.max_len)
        
        # Optional adaptive encoding
        if use_adaptive:
            self.adaptive_encoding = AdaptiveTemporalEncoding(d_model, max_len=self.max_len, num_scales=num_scales)
        
        # Learnable combination weights
        self.combination_weights = nn.Parameter(torch.tensor([0.7, 0.3]) if use_adaptive else torch.tensor([1.0]))
        
    def forward(self, x):
        """
        Apply enhanced temporal encoding.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with enhanced temporal encoding
        """
        # Apply sinusoidal encoding
        sinusoidal_encoded = self.sinusoidal_encoding(x)
        
        if self.use_adaptive:
            # Apply adaptive encoding
            adaptive_encoded = self.adaptive_encoding(x)
            
            # Combine encodings with learnable weights
            weights = torch.softmax(self.combination_weights, dim=0)
            combined = weights[0] * sinusoidal_encoded + weights[1] * adaptive_encoded
            return combined
        else:
            return sinusoidal_encoded