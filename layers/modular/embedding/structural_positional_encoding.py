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
    
    def __init__(self, max_eigenvectors: int = 16, normalization: str = 'sym'):
        """
        Args:
            max_eigenvectors: Maximum number of eigenvectors to compute
            normalization: Type of Laplacian normalization ('sym', 'rw', or None)
        """
        super().__init__()
        self.max_eigenvectors = max_eigenvectors
        self.normalization = normalization
        self.register_buffer('cached_eigenvalues', torch.empty(0))
        self.register_buffer('cached_eigenvectors', torch.empty(0))
        self.cached_edge_index = None
        
    def compute_laplacian_eigenmaps(self, edge_index: torch.Tensor, 
                                   num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Laplacian eigenmaps for the given graph.
        
        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes in the graph
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        # Check if we can use cached results
        if (self.cached_edge_index is not None and 
            torch.equal(edge_index, self.cached_edge_index)):
            return self.cached_eigenvalues, self.cached_eigenvectors
        
        # Get Laplacian matrix
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, normalization=self.normalization, num_nodes=num_nodes
        )
        
        # Convert to dense adjacency matrix for eigendecomposition
        L = to_dense_adj(edge_index_lap, edge_attr=edge_weight_lap, 
                        max_num_nodes=num_nodes).squeeze(0)
        
        # Compute eigendecomposition
        try:
            # Use PyTorch's eigendecomposition for differentiability
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            
            # Sort by eigenvalues (ascending)
            idx = torch.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Take the smallest non-zero eigenvalues and their eigenvectors
            # Skip the first eigenvalue (which should be 0 for connected graphs)
            start_idx = 1 if eigenvalues[0].abs() < 1e-6 else 0
            end_idx = min(start_idx + self.max_eigenvectors, len(eigenvalues))
            
            selected_eigenvalues = eigenvalues[start_idx:end_idx]
            selected_eigenvectors = eigenvectors[:, start_idx:end_idx]
            
        except Exception as e:
            # Fallback to scipy for numerical stability
            L_np = L.detach().cpu().numpy()
            try:
                eigenvalues_np, eigenvectors_np = eigsh(
                    L_np, k=min(self.max_eigenvectors + 1, num_nodes - 1), 
                    which='SM', return_eigenvectors=True
                )
                
                # Convert back to torch tensors
                selected_eigenvalues = torch.from_numpy(eigenvalues_np[1:]).float().to(L.device)
                selected_eigenvectors = torch.from_numpy(eigenvectors_np[:, 1:]).float().to(L.device)
                
            except Exception:
                # Ultimate fallback: random embeddings
                print(f"Warning: Eigendecomposition failed, using random embeddings")
                selected_eigenvalues = torch.randn(self.max_eigenvectors).to(L.device)
                selected_eigenvectors = torch.randn(num_nodes, self.max_eigenvectors).to(L.device)
        
        # Cache results
        self.cached_eigenvalues = selected_eigenvalues
        self.cached_eigenvectors = selected_eigenvectors
        self.cached_edge_index = edge_index.clone()
        
        return selected_eigenvalues, selected_eigenvectors
    
    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Forward pass to compute structural positional encodings.
        
        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            
        Returns:
            Structural positional encodings [num_nodes, max_eigenvectors]
        """
        eigenvalues, eigenvectors = self.compute_laplacian_eigenmaps(edge_index, num_nodes)
        return eigenvectors

class StructuralPositionalEncoding(nn.Module):
    """
    Structural Positional Encoding module for heterogeneous graphs.
    
    Adds structural awareness to node embeddings using Laplacian Eigenmaps,
    allowing the model to understand node positions and roles in the graph structure.
    """
    
    def __init__(self, d_model: int, max_eigenvectors: int = 16, 
                 dropout: float = 0.1, learnable_projection: bool = True):
        """
        Args:
            d_model: Model dimension
            max_eigenvectors: Maximum number of eigenvectors to use
            dropout: Dropout rate
            learnable_projection: Whether to use learnable projection for eigenvectors
        """
        super().__init__()
        self.d_model = d_model
        self.max_eigenvectors = max_eigenvectors
        self.learnable_projection = learnable_projection
        
        # Laplacian eigenmap computation
        self.laplacian_eigenmaps = LaplacianEigenmaps(max_eigenvectors)
        
        if learnable_projection:
            # Learnable projection from eigenvector space to model space
            self.projection = nn.Linear(max_eigenvectors, d_model)
        else:
            # Simple padding/truncation to match d_model
            if max_eigenvectors < d_model:
                self.register_buffer('padding', torch.zeros(d_model - max_eigenvectors))
            
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add structural positional encodings to node embeddings.
        
        Args:
            x_dict: Dictionary of node embeddings {node_type: [num_nodes, d_model]}
            edge_index_dict: Dictionary of edge indices {edge_type: [2, num_edges]}
            
        Returns:
            Dictionary of enhanced node embeddings with structural encodings
        """
        enhanced_x_dict = {}
        
        for node_type, x in x_dict.items():
            # Find relevant edges for this node type
            relevant_edges = []
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                if src_type == node_type or dst_type == node_type:
                    relevant_edges.append(edge_index)
            
            if relevant_edges:
                # Combine all relevant edges
                combined_edge_index = torch.cat(relevant_edges, dim=1)
                
                # Remove duplicates and self-loops
                combined_edge_index = torch.unique(combined_edge_index, dim=1)
                
                # Compute structural positional encoding
                num_nodes = x.size(0)
                structural_encoding = self.laplacian_eigenmaps(combined_edge_index, num_nodes)
                
                # Project to model dimension
                if self.learnable_projection:
                    # Pad if necessary
                    if structural_encoding.size(1) < self.max_eigenvectors:
                        padding = torch.zeros(
                            structural_encoding.size(0), 
                            self.max_eigenvectors - structural_encoding.size(1)
                        ).to(structural_encoding.device)
                        structural_encoding = torch.cat([structural_encoding, padding], dim=1)
                    elif structural_encoding.size(1) > self.max_eigenvectors:
                        structural_encoding = structural_encoding[:, :self.max_eigenvectors]
                    
                    structural_encoding = self.projection(structural_encoding)
                else:
                    # Simple padding/truncation
                    if structural_encoding.size(1) < self.d_model:
                        padding = self.padding.unsqueeze(0).expand(
                            structural_encoding.size(0), -1
                        )
                        structural_encoding = torch.cat([structural_encoding, padding], dim=1)
                    elif structural_encoding.size(1) > self.d_model:
                        structural_encoding = structural_encoding[:, :self.d_model]
                
                # Add structural encoding to node embeddings
                enhanced_x = x + self.dropout(structural_encoding)
                enhanced_x_dict[node_type] = self.layer_norm(enhanced_x)
            else:
                # No edges found, return original embeddings
                enhanced_x_dict[node_type] = x
                
        return enhanced_x_dict
    
    def get_structural_similarity(self, x_dict: Dict[str, torch.Tensor],
                                 edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                                 node_type: str) -> torch.Tensor:
        """
        Compute structural similarity matrix for nodes of a given type.
        
        Args:
            x_dict: Dictionary of node embeddings
            edge_index_dict: Dictionary of edge indices
            node_type: Type of nodes to compute similarity for
            
        Returns:
            Structural similarity matrix [num_nodes, num_nodes]
        """
        if node_type not in x_dict:
            raise ValueError(f"Node type {node_type} not found in x_dict")
            
        # Find relevant edges
        relevant_edges = []
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if src_type == node_type or dst_type == node_type:
                relevant_edges.append(edge_index)
        
        if not relevant_edges:
            # Return identity matrix if no edges
            num_nodes = x_dict[node_type].size(0)
            return torch.eye(num_nodes).to(x_dict[node_type].device)
        
        # Combine edges and compute structural encodings
        combined_edge_index = torch.cat(relevant_edges, dim=1)
        combined_edge_index = torch.unique(combined_edge_index, dim=1)
        
        num_nodes = x_dict[node_type].size(0)
        structural_encoding = self.laplacian_eigenmaps(combined_edge_index, num_nodes)
        
        # Compute cosine similarity between structural encodings
        structural_encoding_norm = F.normalize(structural_encoding, p=2, dim=1)
        similarity_matrix = torch.mm(structural_encoding_norm, structural_encoding_norm.t())
        
        return similarity_matrix

class EnhancedInitialEmbedding(nn.Module):
    """
    Enhanced Initial Embedding with Structural Positional Encoding.
    
    Extends the original InitialEmbedding to include structural awareness
    through Laplacian Eigenmaps.
    """
    
    def __init__(self, config, use_structural_encoding: bool = True):
        """
        Args:
            config: Model configuration
            use_structural_encoding: Whether to use structural positional encoding
        """
        super().__init__()
        self.d_model = config.d_model
        self.use_structural_encoding = use_structural_encoding
        
        # Original embedding layers
        self.wave_embedding = nn.Linear(config.wave_features, config.d_model)
        self.target_embedding = nn.Linear(config.target_features, config.d_model)
        
        # Structural positional encoding
        if use_structural_encoding:
            self.structural_encoding = StructuralPositionalEncoding(
                d_model=config.d_model,
                max_eigenvectors=getattr(config, 'max_eigenvectors', 16),
                dropout=getattr(config, 'dropout', 0.1)
            )
        
        self.dropout = nn.Dropout(getattr(config, 'dropout', 0.1))
        
    def forward(self, wave_x: torch.Tensor, target_x: torch.Tensor, 
                graph: HeteroData) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass with structural positional encoding.
        
        Args:
            wave_x: Wave features [batch_size, num_waves, wave_features]
            target_x: Target features [batch_size, num_targets, target_features]
            graph: Heterogeneous graph data
            
        Returns:
            Tuple of (x_dict, t_dict) with structural encodings applied
        """
        batch_size = wave_x.size(0)
        
        # Original embeddings
        wave_emb = self.wave_embedding(wave_x)
        target_emb = self.target_embedding(target_x)
        
        # Create dictionaries
        x_dict = {
            'wave': wave_emb.view(-1, self.d_model),
            'target': target_emb.view(-1, self.d_model)
        }
        
        # Transition embeddings (learnable parameters)
        num_transitions = graph['transition'].x.size(0) if hasattr(graph['transition'], 'x') else len(graph.node_types) - 2
        transition_emb = nn.Parameter(torch.randn(num_transitions, self.d_model)).to(wave_x.device)
        
        t_dict = {
            'transition': transition_emb.expand(batch_size * num_transitions, -1)
        }
        
        # Apply structural positional encoding
        if self.use_structural_encoding:
            x_dict = self.structural_encoding(x_dict, graph.edge_index_dict)
        
        # Apply dropout
        x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        t_dict = {k: self.dropout(v) for k, v in t_dict.items()}
        
        return x_dict, t_dict