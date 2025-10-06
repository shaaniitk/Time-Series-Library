import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from torch_geometric.data import HeteroData
from layers.modular.graph.registry import GraphComponentRegistry

@GraphComponentRegistry.register("dynamic_graph_constructor")
class DynamicGraphConstructor(nn.Module):
    """
    Dynamic Graph Constructor that learns optimal graph structure from data
    instead of using fixed connectivity patterns.
    """
    
    def __init__(self, d_model: int, num_waves: int, num_targets: int, num_transitions: int):
        super().__init__()
        self.d_model = d_model
        self.num_waves = num_waves
        self.num_targets = num_targets
        self.num_transitions = num_transitions
        
        # Learnable adjacency matrices
        self.wave_to_transition_adj = nn.Parameter(
            torch.randn(num_waves, num_transitions) * 0.1
        )
        self.transition_to_target_adj = nn.Parameter(
            torch.randn(num_transitions, num_targets) * 0.1
        )
        
        # Temperature parameter for Gumbel softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Edge weight predictors
        self.wave_transition_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.transition_target_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_dict: Dict[str, torch.Tensor], training: bool = True) -> Tuple[HeteroData, Dict[str, torch.Tensor]]:
        """
        Construct dynamic graph structure based on input features
        
        Args:
            x_dict: Dictionary containing node features for each node type
            training: Whether in training mode
            
        Returns:
            graph_data: HeteroData object with dynamic edges
            edge_weights: Dictionary containing edge weights
        """
        device = x_dict['wave'].device
        batch_size = x_dict['wave'].shape[0] if len(x_dict['wave'].shape) > 2 else 1
        
        # Create HeteroData object
        graph_data = HeteroData()
        
        # Set node counts
        graph_data['wave'].num_nodes = self.num_waves
        graph_data['transition'].num_nodes = self.num_transitions
        graph_data['target'].num_nodes = self.num_targets
        
        # Compute dynamic adjacency matrices
        if training:
            # Use Gumbel softmax for differentiable sampling
            wave_to_trans_probs = F.gumbel_softmax(
                self.wave_to_transition_adj, tau=self.temperature, hard=False
            )
            trans_to_target_probs = F.gumbel_softmax(
                self.transition_to_target_adj, tau=self.temperature, hard=False
            )
        else:
            # Use hard sampling during inference
            wave_to_trans_probs = F.softmax(self.wave_to_transition_adj, dim=-1)
            trans_to_target_probs = F.softmax(self.transition_to_target_adj, dim=-1)
        
        # Generate edge indices based on learned adjacency
        wave_trans_edges, wave_trans_weights = self._generate_edges(
            wave_to_trans_probs, x_dict['wave'], x_dict['transition'], 
            self.wave_transition_predictor
        )
        
        trans_target_edges, trans_target_weights = self._generate_edges(
            trans_to_target_probs, x_dict['transition'], x_dict['target'],
            self.transition_target_predictor
        )
        
        # Set edge indices
        graph_data['wave', 'interacts_with', 'transition'].edge_index = wave_trans_edges
        graph_data['transition', 'influences', 'target'].edge_index = trans_target_edges
        
        # Store edge weights
        edge_weights = {
            ('wave', 'interacts_with', 'transition'): wave_trans_weights,
            ('transition', 'influences', 'target'): trans_target_weights
        }
        
        return graph_data, edge_weights
    
    def _generate_edges(self, adj_matrix: torch.Tensor, source_features: torch.Tensor, 
                       target_features: torch.Tensor, weight_predictor: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate edge indices and weights from adjacency matrix and node features
        """
        # Handle batch dimension properly
        batch_size = source_features.shape[0]
        num_source_nodes = min(adj_matrix.shape[0], source_features.shape[1])
        num_target_nodes = min(adj_matrix.shape[1], target_features.shape[1])
        
        # Adjust adjacency matrix to match actual feature dimensions
        adj_matrix_adjusted = adj_matrix[:num_source_nodes, :num_target_nodes]
        
        # Get top-k connections for sparsity
        k = min(3, adj_matrix_adjusted.shape[1])  # Connect each node to top-3 neighbors
        top_k_values, top_k_indices = torch.topk(adj_matrix_adjusted, k, dim=1)
        
        # Create edge indices
        source_indices = torch.arange(num_source_nodes, device=adj_matrix.device).unsqueeze(1).expand(-1, k)
        edge_index = torch.stack([source_indices.flatten(), top_k_indices.flatten()])
        
        # Get the features for the source and target nodes of each edge.
        # We use the features from the first batch, assuming homogeneity across the batch for graph construction.
        source_node_features = source_features[0]  # [num_source_nodes, d_model]
        target_node_features = target_features[0]  # [num_target_nodes, d_model]

        # Select the features for the endpoints of each edge
        edge_source_features = source_node_features[source_indices.flatten()] # [num_edges, d_model]
        edge_target_features = target_node_features[top_k_indices.flatten()] # [num_edges, d_model]

        # Concatenate the features to predict edge weights
        edge_features = torch.cat([edge_source_features, edge_target_features], dim=-1) # [num_edges, 2 * d_model]
        edge_weights = weight_predictor(edge_features).squeeze(-1)
        
        # Apply adjacency probabilities as additional weights
        adj_weights = top_k_values.flatten()
        final_weights = edge_weights * adj_weights
        
        return edge_index, final_weights
    
    def get_adjacency_matrices(self) -> Dict[str, torch.Tensor]:
        """
        Get current learned adjacency matrices
        """
        return {
            'wave_to_transition': F.softmax(self.wave_to_transition_adj, dim=-1),
            'transition_to_target': F.softmax(self.transition_to_target_adj, dim=-1)
        }
    
    def regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss to encourage sparsity in adjacency matrices
        """
        l1_loss = torch.abs(self.wave_to_transition_adj).mean() + torch.abs(self.transition_to_target_adj).mean()
        entropy_loss = -torch.sum(F.softmax(self.wave_to_transition_adj, dim=-1) * F.log_softmax(self.wave_to_transition_adj, dim=-1))
        entropy_loss += -torch.sum(F.softmax(self.transition_to_target_adj, dim=-1) * F.log_softmax(self.transition_to_target_adj, dim=-1))
        
        return 0.01 * l1_loss + 0.001 * entropy_loss


@GraphComponentRegistry.register("adaptive_graph_structure")
class AdaptiveGraphStructure(nn.Module):
    """
    Adaptive Graph Structure that evolves during training
    """
    
    def __init__(self, d_model: int, num_waves: int, num_targets: int, num_transitions: int):
        super().__init__()
        self.dynamic_constructor = DynamicGraphConstructor(d_model, num_waves, num_targets, num_transitions)
        
        # Graph structure evolution parameters
        self.structure_memory = nn.Parameter(torch.zeros(num_waves + num_transitions + num_targets, d_model))
        self.memory_update_rate = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x_dict: Dict[str, torch.Tensor], training: bool = True) -> Tuple[HeteroData, Dict[str, torch.Tensor]]:
        """
        Forward pass with adaptive graph structure
        """
        # Update structure memory
        if training:
            # x_dict tensors have shape [batch_size, num_nodes, seq_len]
            # We need to average over batch and sequence dimensions to get [num_nodes, d_model]
            wave_avg = x_dict['wave'].mean(dim=(0, 2))  # [num_nodes]
            transition_avg = x_dict['transition'].mean(dim=(0, 2))  # [num_nodes]
            target_avg = x_dict['target'].mean(dim=(0, 2))  # [num_nodes]
            
            # Expand to d_model dimensions
            d_model = self.structure_memory.shape[1]
            wave_expanded = wave_avg.unsqueeze(-1).expand(-1, d_model)
            transition_expanded = transition_avg.unsqueeze(-1).expand(-1, d_model)
            target_expanded = target_avg.unsqueeze(-1).expand(-1, d_model)
            
            all_features = torch.cat([wave_expanded, transition_expanded, target_expanded], dim=0)
            
            # Ensure dimensions match
            if all_features.shape[0] == self.structure_memory.shape[0]:
                self.structure_memory.data = (1 - self.memory_update_rate) * self.structure_memory.data + \
                                           self.memory_update_rate * all_features.detach()
        
        # Generate dynamic graph
        graph_data, edge_weights = self.dynamic_constructor(x_dict, training)
        
        return graph_data, edge_weights
    
    def get_structure_evolution(self) -> torch.Tensor:
        """
        Get the evolution of graph structure over time
        """
        return self.structure_memory