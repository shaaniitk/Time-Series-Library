import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OptimizedAutoCorrelation(nn.Module):
    def __init__(self, mask_flag=True, factor=1, attention_dropout=0.1, output_attention=False, max_seq_len=1024):
        super().__init__()
        self.factor = factor
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.max_seq_len = max_seq_len

    def time_delay_agg_optimized(self, values, corr):
        """Memory-optimized aggregation"""
        B, H, D, L = values.shape
        top_k = min(int(self.factor * math.log(L)), L // 4, 32)  # Cap max k
        
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, indices = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)
        tmp_corr = F.softmax(weights, dim=-1)
        
        # Vectorized aggregation
        delays_agg = torch.zeros_like(values)
        for i in range(top_k):
            shift = int(indices[i])
            pattern = torch.roll(values, -shift, -1)
            weight = tmp_corr[:, i].view(B, 1, 1, 1)
            delays_agg.add_(pattern * weight)
        
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # Handle sequence length mismatch
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # Use chunked processing for large sequences
        if L > self.max_seq_len:
            return self._chunked_forward(queries, keys, values)

        # Efficient FFT computation
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        
        # Use mixed precision if available
        if queries.device.type == 'cuda' and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                res = q_fft * torch.conj(k_fft)
                corr = torch.fft.irfft(res, dim=-1)
        else:
            res = q_fft * torch.conj(k_fft)
            corr = torch.fft.irfft(res, dim=-1)

        # Optimized time delay aggregation
        V = self.time_delay_agg_optimized(values.permute(0, 2, 3, 1).contiguous(), corr)
        V = V.permute(0, 3, 1, 2)

        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        else:
            return V.contiguous(), None

    def _chunked_forward(self, queries, keys, values):
        """Process large sequences in chunks"""
        chunk_size = self.max_seq_len // 2
        outputs = []
        
        for i in range(0, queries.size(1), chunk_size):
            end_idx = min(i + chunk_size, queries.size(1))
            chunk_q = queries[:, i:end_idx]
            chunk_k = keys[:, i:end_idx]
            chunk_v = values[:, i:end_idx]
            
            chunk_out, _ = self.forward(chunk_q, chunk_k, chunk_v, None)
            outputs.append(chunk_out)
        
        return torch.cat(outputs, dim=1), None


class OptimizedAutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn