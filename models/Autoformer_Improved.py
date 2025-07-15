class ImprovedAutoformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # Add input validation
        self._validate_configs(configs)
        
        # Initialize with proper error handling
        try:
            self.task_name = configs.task_name
            self.seq_len = configs.seq_len
            self.pred_len = configs.pred_len
            
            # Ensure kernel size is valid
            kernel_size = max(3, getattr(configs, 'moving_avg', 25))
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            
            self.decomp = series_decomp(kernel_size)
            
        except AttributeError as e:
            raise ValueError(f"Missing required config parameter: {e}")
    
    def _validate_configs(self, configs):
        """Validate configuration parameters"""
        required_attrs = ['task_name', 'seq_len', 'pred_len', 'enc_in', 'd_model']
        for attr in required_attrs:
            if not hasattr(configs, attr):
                raise ValueError(f"Missing required config: {attr}")
        
        if configs.seq_len <= 0 or configs.pred_len <= 0:
            raise ValueError("Sequence lengths must be positive")
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Add input validation
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Input dimension mismatch: expected {self.enc_in}, got {x_enc.size(-1)}")
        
        # Rest of forward pass...