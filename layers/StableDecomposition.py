class StableSeriesDecomp(nn.Module):
    """Numerically stable series decomposition"""
    
    def __init__(self, kernel_size, eps=1e-8):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        
        # Use learnable weights instead of simple averaging
        self.trend_weights = nn.Parameter(torch.ones(kernel_size) / kernel_size)
        
    def forward(self, x):
        # Normalize weights to prevent numerical issues
        weights = F.softmax(self.trend_weights, dim=0)
        
        # Apply padding
        padding = self.kernel_size // 2
        x_padded = F.pad(x, (0, 0, padding, padding), mode='replicate')
        
        # Compute trend using learnable weights
        trend = F.conv1d(
            x_padded.transpose(1, 2),
            weights.view(1, 1, -1).expand(x.size(-1), 1, -1),
            groups=x.size(-1),
            padding=0
        ).transpose(1, 2)
        
        # Ensure numerical stability
        seasonal = x - trend
        
        # Add small epsilon to prevent zero gradients
        trend = trend + self.eps * torch.randn_like(trend) * trend.std()
        
        return seasonal, trend