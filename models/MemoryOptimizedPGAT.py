"""Memory-optimized SOTA Temporal PGAT with efficient graph operations."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import torch.utils.checkpoint as checkpoint

class MemoryOptimizedPGAT(nn.Module):
    """Memory-efficient version of SOTA Temporal PGAT."""
    
    def __init__(self, config, mode='probabilistic'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.d_model = getattr(config, 'd_model', 512)
        self.n_heads = getattr(config, 'n_heads', 8)
        
        # Pre-initialize all components in __init__ (not forward!)
        self._initialize_components()
        
        # Cache for graph structures
        self._graph_cache: Dict[str, torch.Tensor] = {}
        self._adjacency_cache: Optional[torch.Tensor] = None
        
        # Memory optimization flags
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', True)
        self.use_sparse_attention = getattr(config, 'use_sparse_attention', True)
        self.chunk_size = getattr(config, 'chunk_size', 32)  # Process in chunks
        
        # Device compatibility
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _initialize_components(self):
        """Initialize all components upfront to avoid forward-pass allocation."""
        from layers.modular.graph.dynamic_graph import DynamicGraphConstructor, AdaptiveGraphStructure
        from layers.modular.encoder.spatiotemporal_encoding import AdaptiveSpatioTemporalEncoder
        from layers.modular.embedding.graph_positional_encoding import GraphAwarePositionalEncoding
        from layers.modular.attention.multihead_graph_attention import MultiHeadGraphAttention
        
        # Initialize with reasonable defaults
        num_waves = getattr(self.config, 'num_waves', 7)
        num_targets = getattr(self.config, 'num_targets', 3)
        num_transitions = getattr(self.config, 'num_transitions', min(num_waves, num_targets))
        max_seq_len = getattr(self.config, 'seq_len', 96) + getattr(self.config, 'pred_len', 96)
        max_nodes = num_waves + num_targets + num_transitions
        
        # Pre-initialize all dynamic components
        self.dynamic_graph = DynamicGraphConstructor(
            d_model=self.d_model,
            num_waves=num_waves,
            num_targets=num_targets,
            num_transitions=num_transitions
        )
        
        self.adaptive_graph = AdaptiveGraphStructure(
            d_model=self.d_model,
            num_waves=num_waves,
            num_targets=num_targets,
            num_transitions=num_transitions
        )
        
        self.spatiotemporal_encoder = AdaptiveSpatioTemporalEncoder(
            d_model=self.d_model,
            max_seq_len=max_seq_len,
            max_nodes=max_nodes,
            num_layers=2,
            num_heads=self.n_heads
        )
        
        self.graph_pos_encoding = GraphAwarePositionalEncoding(
            d_model=self.d_model,
            max_nodes=max_nodes,
            max_seq_len=max_seq_len,
            encoding_types=['distance', 'centrality']  # Reduced for memory
        )
        
        self.graph_attention = MultiHeadGraphAttention(
            d_model=self.d_model,
            num_heads=self.n_heads
        )
        
        # Simple embedding layer
        self.embedding = nn.Linear(getattr(self.config, 'enc_in', 7), self.d_model)
        
        # Output decoder
        self.decoder = nn.Linear(self.d_model, getattr(self.config, 'c_out', 1))
        
    def _get_cached_adjacency(self, batch_size: int, seq_len: int, num_nodes: int, device: torch.device) -> torch.Tensor:
        """Get cached adjacency matrix or create sparse version."""
        cache_key = f"{batch_size}_{seq_len}_{num_nodes}_{device}"
        
        if cache_key not in self._graph_cache:
            if self.use_sparse_attention:
                # Create sparse adjacency matrix (block diagonal + some connections)
                adj = self._create_sparse_adjacency(seq_len, num_nodes, device)
            else:
                # Dense adjacency matrix (memory intensive)
                adj = torch.ones(num_nodes, num_nodes, device=device, dtype=torch.float32)
            
            self._graph_cache[cache_key] = adj
            
        return self._graph_cache[cache_key]
    
    def _create_sparse_adjacency(self, seq_len: int, num_nodes: int, device: torch.device) -> torch.Tensor:
        """Create memory-efficient sparse adjacency matrix."""
        # Block diagonal structure with limited connections
        adj = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.float32)
        
        # Local connections (more memory efficient than full connectivity)
        for i in range(num_nodes):
            # Connect to neighbors within window
            window = min(5, num_nodes)  # Limit connection window
            start = max(0, i - window // 2)
            end = min(num_nodes, i + window // 2 + 1)
            adj[i, start:end] = 1.0
            
        return adj
    
    def _process_in_chunks(self, tensor: torch.Tensor, processing_fn, chunk_size: Optional[int] = None) -> torch.Tensor:
        """Process large tensors in chunks to reduce memory usage."""
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        batch_size, seq_len = tensor.shape[:2]
        
        if seq_len <= chunk_size:
            return processing_fn(tensor)
        
        # Process in chunks
        chunks = []
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk = tensor[:, i:end_idx]
            
            if self.use_gradient_checkpointing:
                chunk_result = checkpoint.checkpoint(processing_fn, chunk)
            else:
                chunk_result = processing_fn(chunk)
                
            chunks.append(chunk_result)
        
        return torch.cat(chunks, dim=1)
    
    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark) -> torch.Tensor:
        """Memory-efficient forward pass."""
        batch_size = batch_x.shape[0]
        device = batch_x.device
        
        # Combine inputs efficiently
        combined_input = torch.cat([batch_x, dec_inp], dim=1)
        seq_len = combined_input.shape[1]
        
        # Embedding with memory efficiency
        embedded = self.embedding(combined_input)
        
        # Determine graph dimensions
        num_nodes = getattr(self.config, 'num_nodes', seq_len)
        
        # Get cached adjacency matrix
        adj_matrix = self._get_cached_adjacency(batch_size, seq_len, num_nodes, device)
        
        # Memory-efficient spatial-temporal processing
        def spatial_temporal_process(x):
            # Expand for graph processing (memory intensive operation)
            x_expanded = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            
            # Apply spatial-temporal encoding
            encoded = self.spatiotemporal_encoder(x_expanded, adj_matrix)
            
            # Reduce back to sequence dimension
            return encoded.mean(dim=2)  # Average over nodes
        
        # Process in chunks if sequence is long
        if seq_len > self.chunk_size:
            spatiotemporal_output = self._process_in_chunks(embedded, spatial_temporal_process)
        else:
            if self.use_gradient_checkpointing:
                spatiotemporal_output = checkpoint.checkpoint(spatial_temporal_process, embedded)
            else:
                spatiotemporal_output = spatial_temporal_process(embedded)
        
        # Final decoding
        output = self.decoder(spatiotemporal_output)
        
        return output
    
    def clear_cache(self):
        """Clear cached graph structures to free memory."""
        self._graph_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            'cached_graphs': len(self._graph_cache),
            'model_parameters': sum(p.numel() for p in self.parameters()),
            'device_type': self.device_type,
        }
        
        if torch.cuda.is_available() and self.device_type == "cuda":
            try:
                stats['cuda_allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
                stats['cuda_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
            except:
                stats['cuda_error'] = "Could not get CUDA memory stats"
            
        return stats