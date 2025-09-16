import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

class GraphAwarePositionalEncoding(nn.Module):
    """
    Graph-aware positional encoding that incorporates topology information
    including shortest path distances, node centrality, and structural features
    """
    
    def __init__(self, d_model: int, max_nodes: int, max_seq_len: int, 
                 encoding_types: List[str] = ['distance', 'centrality', 'spectral', 'random_walk']):
        super().__init__()
        self.d_model = d_model
        self.max_nodes = max_nodes
        self.max_seq_len = max_seq_len
        self.encoding_types = encoding_types
        
        # Dimension allocation for different encoding types
        self.encoding_dims = self._allocate_dimensions()
        
        # Distance-based encoding
        if 'distance' in encoding_types:
            self.distance_embedding = nn.Embedding(max_nodes, self.encoding_dims['distance'])
            
        # Centrality-based encoding
        if 'centrality' in encoding_types:
            self.centrality_projection = nn.Linear(4, self.encoding_dims['centrality'])  # 4 centrality measures
            
        # Spectral encoding
        if 'spectral' in encoding_types:
            # Spectral features are padded/truncated to 32 dims, so projection must accept 32
            self.spectral_projection = nn.Linear(32, self.encoding_dims['spectral'])
        
        # Random walk encoding
        if 'random_walk' in encoding_types:
            self.random_walk_projection = nn.Linear(16, self.encoding_dims['random_walk'])  # 16-dim RW features
        
        # Temporal positional encoding (standard sinusoidal)
        self.temporal_pos_encoding = self._create_temporal_encoding(max_seq_len, d_model // 4)
        
        # Learnable combination weights
        self.combination_weights = nn.Parameter(torch.ones(len(encoding_types) + 1))  # +1 for temporal
        
        # Final projection to ensure correct dimensionality
        total_spatial_dim = sum(self.encoding_dims.values())
        self.spatial_projection = nn.Linear(total_spatial_dim, d_model * 3 // 4)  # 3/4 for spatial
        self.temporal_projection = nn.Linear(d_model // 4, d_model // 4)  # 1/4 for temporal
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def _allocate_dimensions(self) -> Dict[str, int]:
        """
        Allocate dimensions for different encoding types
        """
        total_spatial_dim = self.d_model * 3 // 4  # Reserve 1/4 for temporal
        num_types = len(self.encoding_types)
        
        base_dim = total_spatial_dim // num_types
        remainder = total_spatial_dim % num_types
        
        dims = {}
        for i, enc_type in enumerate(self.encoding_types):
            dims[enc_type] = base_dim + (1 if i < remainder else 0)
            
        return dims
    
    def _create_temporal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Create standard sinusoidal temporal positional encoding
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, batch_size: int, seq_len: int, num_nodes: int, 
                adjacency_matrix: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Generate graph-aware positional encodings
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_nodes: Number of nodes
            adjacency_matrix: Graph adjacency matrix [num_nodes, num_nodes]
            device: Device to create tensors on
            
        Returns:
            Positional encodings [batch_size, seq_len, num_nodes, d_model]
        """
        spatial_encodings = []
        
        # Generate different types of spatial encodings
        if 'distance' in self.encoding_types:
            distance_enc = self._distance_encoding(adjacency_matrix, num_nodes, device)
            spatial_encodings.append(distance_enc)
            
        if 'centrality' in self.encoding_types:
            centrality_enc = self._centrality_encoding(adjacency_matrix, num_nodes, device)
            spatial_encodings.append(centrality_enc)
            
        if 'spectral' in self.encoding_types:
            spectral_enc = self._spectral_encoding(adjacency_matrix, num_nodes, device)
            spatial_encodings.append(spectral_enc)
            
        if 'random_walk' in self.encoding_types:
            rw_enc = self._random_walk_encoding(adjacency_matrix, num_nodes, device)
            spatial_encodings.append(rw_enc)
        
        # Combine spatial encodings
        if spatial_encodings:
            combined_spatial = torch.cat(spatial_encodings, dim=-1)  # [num_nodes, total_spatial_dim]
            combined_spatial = self.spatial_projection(combined_spatial)  # [num_nodes, d_model*3/4]
        else:
            combined_spatial = torch.zeros(num_nodes, self.d_model * 3 // 4, device=device)
        
        # Generate temporal encoding
        temporal_enc = self.temporal_pos_encoding[:seq_len].to(device)  # [seq_len, d_model/4]
        temporal_enc = self.temporal_projection(temporal_enc)  # [seq_len, d_model/4]
        
        # Combine spatial and temporal encodings
        # Spatial: [num_nodes, d_model*3/4] -> [1, 1, num_nodes, d_model*3/4]
        # Temporal: [seq_len, d_model/4] -> [1, seq_len, 1, d_model/4]
        spatial_expanded = combined_spatial.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        temporal_expanded = temporal_enc.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, num_nodes, -1)
        
        # Concatenate spatial and temporal
        full_encoding = torch.cat([spatial_expanded, temporal_expanded], dim=-1)  # [batch, seq, nodes, d_model]
        
        return self.layer_norm(full_encoding)
    
    def _distance_encoding(self, adjacency_matrix: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        Create distance-based positional encoding using shortest path distances
        """
        # Convert to numpy for shortest path computation
        adj_np = adjacency_matrix.cpu().numpy()
        
        # Compute shortest path distances
        distances = shortest_path(csr_matrix(adj_np), directed=False, unweighted=True)
        distances[np.isinf(distances)] = num_nodes  # Replace inf with max distance
        distances = torch.tensor(distances, dtype=torch.long, device=device)
        
        # Use average distance to other nodes as position
        avg_distances = distances.float().mean(dim=1).long()
        avg_distances = torch.clamp(avg_distances, 0, self.max_nodes - 1)
        
        # Embed distances
        distance_embeddings = self.distance_embedding(avg_distances)  # [num_nodes, distance_dim]
        
        return distance_embeddings
    
    def _centrality_encoding(self, adjacency_matrix: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        Create centrality-based positional encoding
        """
        adj = adjacency_matrix.float()
        
        # Degree centrality
        degree_centrality = adj.sum(dim=1) / (num_nodes - 1)
        
        # Eigenvector centrality (approximated)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(adj)
            principal_eigenvector = eigenvectors[:, -1].abs()
            eigenvector_centrality = principal_eigenvector / principal_eigenvector.sum()
        except:
            eigenvector_centrality = torch.ones(num_nodes, device=device) / num_nodes
        
        # Closeness centrality (approximated using degree)
        closeness_centrality = degree_centrality / (degree_centrality.sum() + 1e-8)
        
        # Betweenness centrality (approximated)
        betweenness_centrality = degree_centrality * (1 - degree_centrality)
        
        # Combine centrality measures
        centrality_features = torch.stack([
            degree_centrality, eigenvector_centrality, 
            closeness_centrality, betweenness_centrality
        ], dim=1)  # [num_nodes, 4]
        
        centrality_encoding = self.centrality_projection(centrality_features)
        
        return centrality_encoding
    
    def _spectral_encoding(self, adjacency_matrix: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        Create spectral positional encoding using graph Laplacian eigenvectors
        """
        adj = adjacency_matrix.float()
        
        # Compute graph Laplacian
        degree = adj.sum(dim=1)
        degree_matrix = torch.diag(degree)
        laplacian = degree_matrix - adj
        
        # Normalized Laplacian
        degree_sqrt_inv = torch.diag(1.0 / torch.sqrt(degree + 1e-8))
        normalized_laplacian = degree_sqrt_inv @ laplacian @ degree_sqrt_inv
        
        try:
            # Compute eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(normalized_laplacian)
            
            # Use smallest eigenvalues' eigenvectors (excluding the first trivial one)
            num_eigenvectors = min(num_nodes - 1, 32)
            spectral_features = eigenvectors[:, 1:num_eigenvectors + 1]  # [num_nodes, num_eigenvectors]
            
            # Pad if necessary to 32 dims (matches spectral_projection input)
            if spectral_features.size(1) < 32:
                padding = torch.zeros(num_nodes, 32 - spectral_features.size(1), device=device, dtype=spectral_features.dtype)
                spectral_features = torch.cat([spectral_features, padding], dim=1)
            elif spectral_features.size(1) > 32:
                spectral_features = spectral_features[:, :32]
                
        except Exception:
            # Fallback to random features with 32 dims
            spectral_features = torch.randn(num_nodes, 32, device=device)
        
        spectral_encoding = self.spectral_projection(spectral_features)
        
        return spectral_encoding
    
    def _random_walk_encoding(self, adjacency_matrix: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        Create random walk-based positional encoding
        """
        adj = adjacency_matrix.float()
        
        # Normalize adjacency matrix for random walk
        degree = adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # Avoid division by zero
        transition_matrix = adj / degree
        
        # Compute random walk features
        rw_features = []
        current_prob = torch.eye(num_nodes, device=device)
        
        for step in range(16):  # 16 steps
            if step > 0:
                current_prob = current_prob @ transition_matrix
            
            # Use diagonal (return probability) as feature
            rw_features.append(current_prob.diag().unsqueeze(1))
        
        rw_features = torch.cat(rw_features, dim=1)  # [num_nodes, 16]
        rw_encoding = self.random_walk_projection(rw_features)
        
        return rw_encoding


class HierarchicalGraphPositionalEncoding(nn.Module):
    """
    Hierarchical graph positional encoding for multi-scale graph structures
    """
    
    def __init__(self, d_model: int, max_nodes: int, max_seq_len: int, num_scales: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        
        # Multi-scale graph encoders
        self.scale_encoders = nn.ModuleList([
            GraphAwarePositionalEncoding(
                d_model // num_scales, max_nodes, max_seq_len,
                encoding_types=['distance', 'centrality'] if i == 0 else 
                              ['spectral', 'random_walk'] if i == 1 else ['distance']
            ) for i in range(num_scales)
        ])
        
        # Scale combination
        self.scale_combination = nn.Linear(d_model, d_model)
        
    def forward(self, batch_size: int, seq_len: int, num_nodes: int,
                adjacency_matrices: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        """
        Generate hierarchical graph-aware positional encodings
        
        Args:
            adjacency_matrices: List of adjacency matrices at different scales
        """
        scale_encodings = []
        
        for i, (encoder, adj_matrix) in enumerate(zip(self.scale_encoders, adjacency_matrices)):
            scale_enc = encoder(batch_size, seq_len, num_nodes, adj_matrix, device)
            scale_encodings.append(scale_enc)
        
        # Combine multi-scale encodings
        combined_encoding = torch.cat(scale_encodings, dim=-1)
        final_encoding = self.scale_combination(combined_encoding)
        
        return final_encoding