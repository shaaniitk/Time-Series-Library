import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.data import HeteroData
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

class LaplacianEigenmaps(nn.Module):
    """
    Computes Laplacian Eigenmaps for structural positional encoding.
    
    This provides each node with an embedding based on its global position
    and role in the graph's structure, independent of node features.
    """
    
    def __init__(self, num_eigenvectors: int = 16, normalization: str = 'sym', max_eigenvectors: Optional[int] = None):
        """
        Args:
            num_eigenvectors: Number of eigenvectors to compute
            normalization: Type of Laplacian normalization ('sym', 'rw', or None)
            max_eigenvectors: Deprecated alias; if provided, overrides num_eigenvectors
        """
        super().__init__()
        self.num_eigenvectors = max_eigenvectors if max_eigenvectors is not None else num_eigenvectors
        self.normalization = normalization
        self.register_buffer('cached_eigenvalues', torch.empty(0))
        self.register_buffer('cached_eigenvectors', torch.empty(0))
        self.cached_edge_index = None
        self.register_buffer('cached_adj', torch.empty(0))
        
    def _laplacian_from_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Compute (normalized) graph Laplacian from a dense adjacency matrix."""
        A = adj
        N = A.size(0)
        deg = A.sum(dim=1)
        if self.normalization == 'sym':
            # L_sym = I - D^{-1/2} A D^{-1/2}
            d_inv_sqrt = torch.pow(deg.clamp_min(1e-6), -0.5)
            D_inv_sqrt = torch.diag(d_inv_sqrt)
            L = torch.eye(N, device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
        elif self.normalization == 'rw':
            # L_rw = I - D^{-1} A
            d_inv = torch.pow(deg.clamp_min(1e-6), -1.0)
            D_inv = torch.diag(d_inv)
            L = torch.eye(N, device=A.device) - D_inv @ A
        else:
            # Unnormalized Laplacian L = D - A
            D = torch.diag(deg)
            L = D - A
        return L
        
    def _append_constant_eigenvector(self, eigvecs: torch.Tensor, N: int, device: torch.device) -> torch.Tensor:
        """Append the constant eigenvector (associated with zero eigenvalue) if needed."""
        const = torch.ones(N, 1, device=device) / (N ** 0.5)
        if eigvecs.numel() == 0:
            return const
        # Ensure it's not already present (basic check via correlation)
        # Always prepend for consistency
        return torch.cat([const, eigvecs], dim=1)
        
    def compute_laplacian_eigenmaps(self, edge_index: torch.Tensor, 
                                   num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Laplacian eigenmaps for the given graph (edge_index input).
        """
        # Check if we can use cached results
        if (self.cached_edge_index is not None and 
            torch.equal(edge_index, self.cached_edge_index)):
            return self.cached_eigenvalues, self.cached_eigenvectors
        
        # Get Laplacian matrix
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, normalization=self.normalization, num_nodes=num_nodes
        )
        
        # Convert to dense Laplacian matrix for eigendecomposition
        L = to_dense_adj(edge_index_lap, edge_attr=edge_weight_lap, 
                        max_num_nodes=num_nodes).squeeze(0)
        
        # Compute eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            idx = torch.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            # Include zero-mode and take the first k eigenvectors
            k = min(self.num_eigenvectors, eigenvectors.size(1))
            selected_eigenvalues = eigenvalues[:k]
            selected_eigenvectors = eigenvectors[:, :k]
            # If fewer than requested and we can, append constant vector
            if k < self.num_eigenvectors:
                selected_eigenvectors = self._append_constant_eigenvector(selected_eigenvectors, num_nodes, L.device)
                selected_eigenvectors = selected_eigenvectors[:, :self.num_eigenvectors]
        except Exception:
            L_np = L.detach().cpu().numpy()
            try:
                # eigsh requires k < N; compute k' and later prepend constant vector if needed
                k_req = min(self.num_eigenvectors, num_nodes - 1) if num_nodes > 1 else 0
                if k_req > 0:
                    eigenvalues_np, eigenvectors_np = eigsh(
                        L_np, k=k_req, which='SM', return_eigenvectors=True
                    )
                    selected_eigenvalues = torch.from_numpy(eigenvalues_np).float().to(L.device)
                    selected_eigenvectors = torch.from_numpy(eigenvectors_np).float().to(L.device)
                else:
                    selected_eigenvalues = torch.zeros(1, device=L.device)
                    selected_eigenvectors = torch.zeros(num_nodes, 1, device=L.device)
                # Prepend constant eigenvector if we still need more columns
                if selected_eigenvectors.size(1) < self.num_eigenvectors:
                    selected_eigenvectors = self._append_constant_eigenvector(selected_eigenvectors, num_nodes, L.device)
                    if selected_eigenvectors.size(1) > self.num_eigenvectors:
                        selected_eigenvectors = selected_eigenvectors[:, :self.num_eigenvectors]
            except Exception:
                print(f"Warning: Eigendecomposition failed, using random embeddings")
                selected_eigenvalues = torch.randn(self.num_eigenvectors).to(L.device)
                selected_eigenvectors = torch.randn(num_nodes, self.num_eigenvectors).to(L.device)
        
        # Cache results
        self.cached_eigenvalues = selected_eigenvalues
        self.cached_eigenvectors = selected_eigenvectors
        self.cached_edge_index = edge_index.clone()
        self.cached_adj = torch.empty(0, device=edge_index.device)
        
        return selected_eigenvalues, selected_eigenvectors
    
    def compute_from_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Compute eigenvectors directly from a dense adjacency matrix."""
        # Check cache
        if self.cached_adj.numel() > 0 and torch.equal(adj, self.cached_adj):
            return self.cached_eigenvectors
        N = adj.size(0)
        L = self._laplacian_from_adj(adj)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            idx = torch.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            k = min(self.num_eigenvectors, eigenvectors.size(1))
            selected_eigenvectors = eigenvectors[:, :k]
            if k < self.num_eigenvectors:
                selected_eigenvectors = self._append_constant_eigenvector(selected_eigenvectors, N, L.device)
                selected_eigenvectors = selected_eigenvectors[:, :self.num_eigenvectors]
        except Exception:
            L_np = L.detach().cpu().numpy()
            try:
                k_req = min(self.num_eigenvectors, N - 1) if N > 1 else 0
                if k_req > 0:
                    _, eigenvectors_np = eigsh(
                        L_np, k=k_req, which='SM', return_eigenvectors=True
                    )
                    selected_eigenvectors = torch.from_numpy(eigenvectors_np).float().to(L.device)
                else:
                    selected_eigenvectors = torch.zeros(N, 1, device=L.device)
                if selected_eigenvectors.size(1) < self.num_eigenvectors:
                    selected_eigenvectors = self._append_constant_eigenvector(selected_eigenvectors, N, L.device)
                    if selected_eigenvectors.size(1) > self.num_eigenvectors:
                        selected_eigenvectors = selected_eigenvectors[:, :self.num_eigenvectors]
            except Exception:
                print(f"Warning: Eigendecomposition failed, using random embeddings")
                selected_eigenvectors = torch.randn(N, self.num_eigenvectors).to(L.device)
        # Cache
        self.cached_eigenvectors = selected_eigenvectors
        self.cached_adj = adj.clone()
        self.cached_edge_index = None
        return selected_eigenvectors
    
    def forward(self, adj_or_edge_index: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass to compute structural positional encodings.
        
        Supports either:
        - Dense adjacency matrix: forward(adj_matrix)
        - Edge index with node count: forward(edge_index, num_nodes)
        
        Returns:
            Structural positional encodings [num_nodes, num_eigenvectors]
        """
        if adj_or_edge_index.dim() == 2 and adj_or_edge_index.size(0) == adj_or_edge_index.size(1):
            # Dense adjacency matrix path
            return self.compute_from_adj(adj_or_edge_index)
        else:
            assert num_nodes is not None, "num_nodes must be provided when using edge_index input"
            _, eigvecs = self.compute_laplacian_eigenmaps(adj_or_edge_index, num_nodes)
            return eigvecs

class StructuralPositionalEncoding(nn.Module):
    """
    Structural Positional Encoding module for homogeneous graphs (test path)
    and heterogeneous graphs (legacy path).
    
    Adds structural awareness to node embeddings using Laplacian Eigenmaps,
    allowing the model to understand node positions and roles in the graph structure.
    """
    
    def __init__(self, d_model: int, num_eigenvectors: int = 16, 
                 dropout: float = 0.1, learnable_projection: bool = True, max_nodes: Optional[int] = None):
        """
        Args:
            d_model: Model dimension
            num_eigenvectors: Number of eigenvectors to use
            dropout: Dropout rate
            learnable_projection: Whether to use learnable projection for eigenvectors
            max_nodes: Unused placeholder for API compatibility with tests
        """
        super().__init__()
        self.d_model = d_model
        self.num_eigenvectors = num_eigenvectors
        self.learnable_projection = learnable_projection
        
        # Laplacian eigenmap computation
        self.laplacian_eigenmaps = LaplacianEigenmaps(num_eigenvectors=num_eigenvectors)
        
        if learnable_projection:
            # Learnable projection from eigenvector space to model space
            self.projection = nn.Linear(num_eigenvectors, d_model)
        else:
            if num_eigenvectors < d_model:
                self.register_buffer('padding', torch.zeros(d_model - num_eigenvectors))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x_or_x_dict, adj_or_edge_index_dict):
        """
        Add structural positional encodings to node embeddings.
        
        Two supported modes:
        - Homogeneous (tests): x: Tensor [N, d_model], adj: Tensor [N, N]
        - Legacy heterogeneous: x_dict: Dict[str, Tensor], edge_index_dict: Dict[Tuple[str,str,str], Tensor]
        """
        # Homogeneous simple path for tests
        if isinstance(x_or_x_dict, torch.Tensor) and isinstance(adj_or_edge_index_dict, torch.Tensor):
            x: torch.Tensor = x_or_x_dict
            adj: torch.Tensor = adj_or_edge_index_dict
            # Compute structural encodings
            structural_encoding = self.laplacian_eigenmaps(adj)  # [N, k']
            # Pad/truncate to num_eigenvectors
            if structural_encoding.size(1) < self.num_eigenvectors:
                padding = torch.zeros(
                    structural_encoding.size(0), 
                    self.num_eigenvectors - structural_encoding.size(1),
                    device=structural_encoding.device,
                    dtype=structural_encoding.dtype
                )
                structural_encoding = torch.cat([structural_encoding, padding], dim=1)
            elif structural_encoding.size(1) > self.num_eigenvectors:
                structural_encoding = structural_encoding[:, :self.num_eigenvectors]
            # Project or pad to d_model
            if self.learnable_projection:
                structural_encoding = self.projection(structural_encoding)
            else:
                if structural_encoding.size(1) < self.d_model:
                    padding = self.padding.unsqueeze(0).expand(structural_encoding.size(0), -1)
                    structural_encoding = torch.cat([structural_encoding, padding], dim=1)
                elif structural_encoding.size(1) > self.d_model:
                    structural_encoding = structural_encoding[:, :self.d_model]
            enhanced_x = x + self.dropout(structural_encoding)
            return self.layer_norm(enhanced_x)
        
        # Legacy heterogeneous path (retain for compatibility)
        x_dict: Dict[str, torch.Tensor] = x_or_x_dict
        edge_index_dict = adj_or_edge_index_dict
        enhanced_x_dict = {}
        
        for node_type, x in x_dict.items():
            # Find relevant edges for this node type
            relevant_edges = []
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                if src_type == node_type or dst_type == node_type:
                    relevant_edges.append(edge_index)
            
            if relevant_edges:
                combined_edge_index = torch.cat(relevant_edges, dim=1)
                combined_edge_index = torch.unique(combined_edge_index, dim=1)
                num_nodes = x.size(0)
                structural_encoding = self.laplacian_eigenmaps(combined_edge_index, num_nodes)
                # Project to model dimension
                if self.learnable_projection:
                    if structural_encoding.size(1) < self.num_eigenvectors:
                        padding = torch.zeros(
                            structural_encoding.size(0), 
                            self.num_eigenvectors - structural_encoding.size(1),
                            device=structural_encoding.device,
                            dtype=structural_encoding.dtype
                        )
                        structural_encoding = torch.cat([structural_encoding, padding], dim=1)
                    elif structural_encoding.size(1) > self.num_eigenvectors:
                        structural_encoding = structural_encoding[:, :self.num_eigenvectors]
                    structural_encoding = self.projection(structural_encoding)
                else:
                    if structural_encoding.size(1) < self.d_model:
                        padding = self.padding.unsqueeze(0).expand(
                            structural_encoding.size(0), -1
                        )
                        structural_encoding = torch.cat([structural_encoding, padding], dim=1)
                    elif structural_encoding.size(1) > self.d_model:
                        structural_encoding = structural_encoding[:, :self.d_model]
                enhanced_x = x + self.dropout(structural_encoding)
                enhanced_x_dict[node_type] = self.layer_norm(enhanced_x)
            else:
                enhanced_x_dict[node_type] = x
        return enhanced_x_dict

class EnhancedInitialEmbedding(nn.Module):
    """
    Enhanced Initial Embedding with Structural Positional Encoding.
    
    Simplified API for tests: maps input features to d_model and adds structural encoding
    computed from a provided adjacency matrix.
    """
    
    def __init__(self, c_in: Optional[int] = None, d_model: Optional[int] = None,
                 use_structural_encoding: bool = True, num_eigenvectors: int = 8,
                 config: Optional[object] = None):
        super().__init__()
        # Two modes: simple (tests) or legacy (config)
        self.use_structural_encoding = use_structural_encoding
        if config is not None:
            # Legacy path (not used in current tests but kept for compatibility)
            self.d_model = config.d_model
            self.input_proj = nn.Linear(getattr(config, 'wave_features', c_in or 1), config.d_model)
            self.structural_encoding = StructuralPositionalEncoding(
                d_model=config.d_model,
                num_eigenvectors=getattr(config, 'max_eigenvectors', num_eigenvectors),
                dropout=getattr(config, 'dropout', 0.1)
            ) if use_structural_encoding else None
            self.dropout = nn.Dropout(getattr(config, 'dropout', 0.1))
        else:
            assert c_in is not None and d_model is not None, "c_in and d_model must be provided when config is None"
            self.d_model = d_model
            self.input_proj = nn.Linear(c_in, d_model)
            self.structural_encoding = StructuralPositionalEncoding(
                d_model=d_model,
                num_eigenvectors=num_eigenvectors,
                dropout=0.1
            ) if use_structural_encoding else None
            self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(getattr(self, 'd_model', d_model))
    
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor, graph: Optional[HeteroData] = None) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, seq_len, c_in]
            adj_matrix: Dense adjacency [seq_len, seq_len]
            graph: Optional legacy graph (unused in test path)
        Returns:
            Embedded features [batch_size, seq_len, d_model]
        """
        # Project inputs
        base = self.input_proj(x)  # [B, T, d_model]
        if not self.use_structural_encoding:
            return self.layer_norm(self.dropout(base))
        # Compute structural encodings per position from adjacency
        evecs = self.structural_encoding.laplacian_eigenmaps(adj_matrix)  # [T, k']
        # Pad/truncate to expected num_eigenvectors used by the structural module
        k = self.structural_encoding.num_eigenvectors
        if evecs.size(1) < k:
            padding = torch.zeros(evecs.size(0), k - evecs.size(1), device=evecs.device, dtype=evecs.dtype)
            evecs = torch.cat([evecs, padding], dim=1)
        elif evecs.size(1) > k:
            evecs = evecs[:, :k]
        # Project to model dimension
        if self.structural_encoding.learnable_projection:
            struct_proj = self.structural_encoding.projection(evecs)  # [T, d_model]
        else:
            # If not learnable, pad/truncate directly to d_model
            if evecs.size(1) < self.structural_encoding.d_model:
                pad = self.structural_encoding.padding.unsqueeze(0).expand(evecs.size(0), -1)
                struct_proj = torch.cat([evecs, pad], dim=1)
            else:
                struct_proj = evecs[:, :self.structural_encoding.d_model]
        # Broadcast and add to all batch elements
        struct_proj_b = struct_proj.unsqueeze(0).expand(base.size(0), -1, -1)
        out = base + self.structural_encoding.dropout(struct_proj_b)
        return self.layer_norm(out)