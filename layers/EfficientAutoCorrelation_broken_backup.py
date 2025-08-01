class EfficientAutoCorrelation(nn.Module):
    """Memory and compute efficient AutoCorrelation"""
    
    def __init__(self, mask_flag=True, factor=1, attention_dropout=0.1, 
                 max_seq_len=1024, use_checkpoint=True):
        super().__init__()
        self.factor = factor
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        
        # Use gradient checkpointing for large sequences
        if self.use_checkpoint and L > self.max_seq_len:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, queries, keys, values, attn_mask
            )
        else:
            return self._forward_impl(queries, keys, values, attn_mask)
    
    def _forward_impl(self, queries, keys, values, attn_mask):
        # Chunked processing for memory efficiency
        chunk_size = min(512, queries.size(1))
        outputs = []
        
        for i in range(0, queries.size(1), chunk_size):
            end_idx = min(i + chunk_size, queries.size(1))
            chunk_q = queries[:, i:end_idx]
            chunk_out = self._process_chunk(chunk_q, keys, values, attn_mask)
            outputs.append(chunk_out)
        
        return torch.cat(outputs, dim=1), None
    
    def _process_chunk(self, queries, keys, values, attn_mask):
        # Efficient FFT-based correlation computation
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        
        # Use half precision for intermediate computations if available
        if queries.device.type == 'cuda':
            q_fft = q_fft.half()
            k_fft = k_fft.half()
        
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1).float()
        
        # Simplified aggregation
        return self._efficient_time_delay_agg(values, corr)