import torch
import torch.nn as nn
import inspect
from typing import Any, Dict, Optional, Tuple, cast

from layers.modular.attention.registry import AttentionRegistry, get_attention_component
from layers.modular.decoder.registry import DecoderRegistry, get_decoder_component
from layers.modular.graph.registry import GraphComponentRegistry
from layers.modular.embedding.registry import EmbeddingRegistry
from layers.modular.graph.dynamic_graph import DynamicGraphConstructor, AdaptiveGraphStructure
from layers.modular.attention.multihead_graph_attention import MultiHeadGraphAttention
from layers.modular.encoder.spatiotemporal_encoding import AdaptiveSpatioTemporalEncoder
from layers.modular.embedding.graph_positional_encoding import GraphAwarePositionalEncoding
# New enhanced components
from layers.modular.decoder.mixture_density_decoder import MixtureDensityDecoder, MixtureNLLLoss
from layers.modular.attention.autocorr_temporal_attention import AutoCorrTemporalAttention
from layers.modular.embedding.structural_positional_encoding import StructuralPositionalEncoding
from layers.modular.embedding.enhanced_temporal_encoding import EnhancedTemporalEncoding
from layers.modular.graph.enhanced_pgat_layer import EnhancedPGAT_CrossAttn_Layer
from utils.graph_aware_dimension_manager import GraphAwareDimensionManager, create_graph_aware_dimension_manager
from layers.modular.graph.gated_graph_combiner import GatedGraphCombiner
from layers.modular.encoder.registry import get_encoder_component
from layers.modular.graph.registry import get_graph_component


class _LazyLinearEmbedding(nn.Module):
    """Minimal lazy linear projection used by the fallback embedding path."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.LazyLinear(output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project raw inputs to the model dimension with normalization."""
        projected = self.projection(inputs)
        return self.layer_norm(projected)

class SOTA_Temporal_PGAT(nn.Module):
    """
    State-of-the-Art Temporal Probabilistic Graph Attention Transformer
    Refactored to use modular components from registries
    """
    
    def __init__(self, config, mode='probabilistic'):
        super().__init__()
        self.config = config
        self.mode = mode

        self._validate_config()
        
        # Track probabilistic configuration early for loss/decoder wiring
        self._use_mdn_outputs = bool(getattr(config, 'use_mixture_density', True) and mode != 'standard')
        self.mixture_loss = MixtureNLLLoss() if self._use_mdn_outputs else None
        decoder_output_dim = getattr(config, 'c_out', None)
        if not isinstance(decoder_output_dim, int) or decoder_output_dim <= 0:
            fallback_dim = getattr(config, 'c_out_evaluation', 1)
            decoder_output_dim = fallback_dim if isinstance(fallback_dim, int) and fallback_dim > 0 else 1
        self.decoder_output_dim = decoder_output_dim
        
        # Note: Registries are used via get_attention_component and get_decoder_component functions
        # No need to store registry instances as they're accessed statically
        
        # Initialize enhanced graph-aware dimension manager
        self.dim_manager = create_graph_aware_dimension_manager(config)
        
        # Validate graph compatibility upfront
        graph_components = {
            'enhanced_pgat': {'num_nodes': getattr(config, 'enc_in', 7), 'd_model': getattr(config, 'd_model', 512)},
            'spatial_encoder': {'num_nodes': getattr(config, 'enc_in', 7), 'd_model': getattr(config, 'd_model', 512)},
            'temporal_attention': {'d_model': getattr(config, 'd_model', 512)}
        }
        
        is_compatible, issues = self.dim_manager.validate_graph_compatibility(graph_components)
        if not is_compatible:
            print(f"Warning: Graph compatibility issues detected: {issues}")
            print(f"Dimension manager info: {self.dim_manager}")
        
        # Get components from registries
        self._embedding_source = 'unknown'
        self._fallback_input_dim: Optional[int] = None
        self.embedding = self._initialize_embedding(config)
        
        # Enhanced graph components initialization
        self.d_model = getattr(config, 'd_model', 512)
        self.n_heads = getattr(config, 'n_heads', 8)
        
        # MEMORY FIX 1: Initialize all components in __init__ instead of forward pass
        # This prevents memory fragmentation and repeated allocations
        
        # Pre-initialize with reasonable defaults, will be updated in forward if needed
        default_wave_nodes = getattr(config, 'enc_in', 7)
        default_target_nodes = getattr(config, 'c_out', 3) 
        default_transition_nodes = min(default_wave_nodes, default_target_nodes)
        default_total_nodes = default_wave_nodes + default_target_nodes + default_transition_nodes
        default_seq_len = getattr(config, 'seq_len', 96) + getattr(config, 'pred_len', 24)
        
        # Initialize graph components from registry
        self.dynamic_graph = get_graph_component(
            'dynamic_graph_constructor',
            d_model=self.d_model,
            num_waves=default_wave_nodes,
            num_targets=default_target_nodes,
            num_transitions=default_transition_nodes
        )
        
        self.adaptive_graph = get_graph_component(
            'adaptive_graph_structure',
            d_model=self.d_model,
            num_waves=default_wave_nodes,
            num_targets=default_target_nodes,
            num_transitions=default_transition_nodes
        )
        
        if getattr(config, 'use_gated_graph_combiner', False):
            self.graph_combiner = get_graph_component(
                'gated_graph_combiner',
                num_nodes=default_total_nodes,
                d_model=self.d_model,
                num_graphs=2  # Base + adaptive graphs
            )
        else:
            self.graph_combiner = None
            
        self.spatiotemporal_encoder = get_graph_component(
            'adaptive_spatiotemporal_encoder',
            d_model=self.d_model,
            max_seq_len=default_seq_len,
            max_nodes=default_total_nodes,
            num_layers=2,
            num_heads=self.n_heads
        )
        
        self.graph_pos_encoding = get_graph_component(
            'graph_aware_positional_encoding',
            d_model=self.d_model,
            max_nodes=default_total_nodes,
            max_seq_len=default_seq_len,
            encoding_types=['distance', 'centrality', 'spectral']
        )
        
        self.graph_attention = get_graph_component(
            'multihead_graph_attention',
            d_model=self.d_model,
            num_heads=self.n_heads
        )
        
        # Feature projection for fallback cases - only create when needed
        # (Removed duplicate - will be created in exception handler if needed)
        
        # MEMORY FIX 2: Add memory management utilities
        self._memory_cache = {}
        self._enable_memory_optimization = getattr(config, 'enable_memory_optimization', True)
        self._chunk_size = getattr(config, 'memory_chunk_size', 32)
        self._use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        
        # CRITICAL FIX: Initialize temporal-to-spatial conversion modules properly
        # These MUST be in __init__ to be registered with PyTorch's module system
        self.wave_temporal_to_spatial = None  # Will be initialized on first forward pass
        self.target_temporal_to_spatial = None  # Will be initialized on first forward pass
        
        # Enhanced spatial encoder with dynamic edge weights
        # Note: Previously forced disabled due to index errors; now gated via config with safe fallback
        try:
            spatial_encoder_name = getattr(config, 'spatial_encoder', 'enhanced_pgat_cross_attn')
            self.spatial_encoder = get_encoder_component(
                spatial_encoder_name,
                d_model=config.d_model,
                num_heads=getattr(config, 'n_heads', 8),
                use_dynamic_weights=getattr(config, 'use_dynamic_edge_weights', True)
            )
            self.enhanced_pgat_enabled = isinstance(self.spatial_encoder, EnhancedPGAT_CrossAttn_Layer)
        except Exception as e:
            print(f"Info: Spatial encoder '{getattr(config, 'spatial_encoder', 'enhanced_pgat_cross_attn')}' unavailable ({e}); falling back to Linear spatial encoder")
            self.enhanced_pgat_enabled = False
            # Use simple linear layer for spatial encoding
            self.spatial_encoder = nn.Linear(config.d_model, config.d_model)
            # Feature projection for concatenated features when enhanced PGAT is disabled
            total_features = config.d_model * 3  # wave + transition + target
            self.feature_projection = nn.Linear(total_features, config.d_model)
        
        # Enhanced temporal encoder - respect use_autocorr_attention setting
        if getattr(config, 'use_autocorr_attention', False):
            try:
                temporal_encoder_name = getattr(config, 'temporal_encoder', 'autocorr_temporal')
                self.temporal_encoder = get_encoder_component(
                    temporal_encoder_name,
                    d_model=config.d_model,
                    num_heads=getattr(config, 'n_heads', 8),
                    dropout=getattr(config, 'dropout', 0.1),
                    factor=getattr(config, 'autocorr_factor', 1)
                )
            except Exception as e:
                print(f"Info: Temporal encoder '{getattr(config, 'temporal_encoder', 'autocorr_temporal')}' unavailable ({e}); falling back to default temporal attention.")
                # Use the attention factory for robust instantiation
                self.temporal_encoder = get_attention_component(
                    'temporal_attention',
                    d_model=config.d_model,
                    n_heads=getattr(config, 'n_heads', 8),
                    dropout=getattr(config, 'dropout', 0.1)
                )
        else:
            # Use simple temporal attention when autocorr is disabled
            self.temporal_encoder = get_attention_component(
                'temporal_attention',
                d_model=config.d_model,
                n_heads=getattr(config, 'n_heads', 8),
                dropout=getattr(config, 'dropout', 0.1)
            )
        
        # Graph information storage flag
        self.store_graph_info = False
        self.last_adjacency_matrix = None
        self.last_edge_weights = None
        
        # Enhanced decoder selection with mixture density network
        if self.mode == 'standard':
            self.decoder = get_decoder_component(
                'custom_standard',
                d_model=config.d_model,
                output_dim=self.decoder_output_dim,
            )
        elif self._use_mdn_outputs:
            self.decoder = MixtureDensityDecoder(
                d_model=config.d_model,
                pred_len=getattr(config, 'pred_len', 96),
                num_components=getattr(config, 'mdn_components', 3)
            )
        else:
            self.decoder = get_decoder_component(
                'probabilistic',
                d_model=config.d_model,
                output_dim=self.decoder_output_dim,
            )
        
        # Add structural positional encoding
        self.structural_pos_encoding = StructuralPositionalEncoding(
            d_model=config.d_model,
            num_eigenvectors=getattr(config, 'max_eigenvectors', 16),
            dropout=getattr(config, 'dropout', 0.1),
            learnable_projection=True
        )
        
        # Add enhanced temporal positional encoding
        self.temporal_pos_encoding = EnhancedTemporalEncoding(
            d_model=config.d_model,
            max_seq_len=getattr(config, 'seq_len', 96) + getattr(config, 'pred_len', 96),
            use_adaptive=getattr(config, 'use_adaptive_temporal', True)

        )
    
    def forward(self, wave_window, target_window, graph=None):
        """
        Forward pass through the SOTA Temporal PGAT model
        
        Args:
            wave_window: Wave input data
            target_window: Target input data
            graph: Graph adjacency matrix (currently unused - reserved for future use)
            
        Returns:
            Model output (single tensor for standard mode, tuple for probabilistic mode)
        """
        self._validate_forward_inputs(wave_window, target_window)

        # Initial embedding - concatenate wave and target windows
        # wave_window: [batch, seq_len, features], target_window: [batch, pred_len, features]
        combined_input = torch.cat([wave_window, target_window], dim=1)  # [batch, seq_len+pred_len, features]
        batch_size, seq_len, features = combined_input.shape
        
        # Check if input is already embedded (features == d_model) or needs embedding
        if features == getattr(self, 'd_model', 512):
            embedded = combined_input  # Already embedded
        else:
            embedded = self.embedding(combined_input.view(-1, features)).view(batch_size, seq_len, -1)
        
        self._validate_embedding_output(embedded, batch_size, seq_len)

        # Apply enhanced temporal positional encoding
        embedded = self.temporal_pos_encoding(embedded)
        
        # Spatial encoding with graph attention
        # Split embedded tensor back into wave and target parts
        wave_len = wave_window.shape[1]
        target_len = target_window.shape[1]
        wave_embedded = embedded[:, :wave_len, :]
        target_embedded = embedded[:, wave_len:wave_len+target_len, :]

        # CRITICAL FIX: Use feature dimensions as node counts, NOT sequence lengths
        # Nodes represent variables/features, not time steps
        wave_nodes = getattr(self.config, 'enc_in', 7)  # Number of input features
        target_nodes = getattr(self.config, 'c_out', 3)  # Number of output features  
        transition_nodes = max(1, min(wave_nodes, target_nodes))  # Transition nodes between input and output

        # CRITICAL FIX: Convert temporal tensors to spatial (node-based) tensors
        # wave_embedded: [batch, seq_len=96, d_model] -> [batch, wave_nodes=7, d_model]
        # target_embedded: [batch, pred_len=24, d_model] -> [batch, target_nodes=3, d_model]
        
        # Enhancement 2: Conditionally use Attention-based temporal-to-spatial conversion
        if getattr(self.config, 'use_attention_temp_to_spatial', False):
            from layers.modular.embedding.attention_temporal_to_spatial import AttentionTemporalToSpatial
            if self.wave_temporal_to_spatial is None:
                self.wave_temporal_to_spatial = AttentionTemporalToSpatial(self.d_model, wave_nodes, n_heads=self.n_heads)
                self.add_module('wave_temporal_to_spatial', self.wave_temporal_to_spatial)
            if self.target_temporal_to_spatial is None:
                self.target_temporal_to_spatial = AttentionTemporalToSpatial(self.d_model, target_nodes, n_heads=self.n_heads)
                self.add_module('target_temporal_to_spatial', self.target_temporal_to_spatial)
            
            # Ensure modules are on correct device
            self.wave_temporal_to_spatial = self.wave_temporal_to_spatial.to(wave_embedded.device)
            self.target_temporal_to_spatial = self.target_temporal_to_spatial.to(target_embedded.device)

            wave_spatial = self.wave_temporal_to_spatial(wave_embedded)
            target_spatial = self.target_temporal_to_spatial(target_embedded)

        else:
            # Original Linear conversion logic
            if self.wave_temporal_to_spatial is None:
                self.wave_temporal_to_spatial = nn.Linear(wave_len, wave_nodes)
                self.add_module('wave_temporal_to_spatial', self.wave_temporal_to_spatial)
            if self.target_temporal_to_spatial is None:
                self.target_temporal_to_spatial = nn.Linear(target_len, target_nodes)
                self.add_module('target_temporal_to_spatial', self.target_temporal_to_spatial)
            
            # Ensure modules are on correct device
            self.wave_temporal_to_spatial = self.wave_temporal_to_spatial.to(wave_embedded.device)
            self.target_temporal_to_spatial = self.target_temporal_to_spatial.to(target_embedded.device)
            
            # Convert temporal to spatial: [batch, seq_len, d_model] -> [batch, d_model, seq_len] -> [batch, d_model, nodes] -> [batch, nodes, d_model]
            wave_spatial = self.wave_temporal_to_spatial(wave_embedded.transpose(1, 2)).transpose(1, 2)  # [batch, wave_nodes, d_model]
            target_spatial = self.target_temporal_to_spatial(target_embedded.transpose(1, 2)).transpose(1, 2)  # [batch, target_nodes, d_model]
        
        # Validate the converted tensors
        self._validate_node_tensor(wave_spatial, "wave_spatial", batch_size, wave_nodes)
        self._validate_node_tensor(target_spatial, "target_spatial", batch_size, target_nodes)
        # Calculate total graph nodes
        total_graph_nodes = wave_nodes + target_nodes + transition_nodes
        
        self.config.num_waves = wave_nodes
        self.config.num_targets = target_nodes
        self.config.num_transitions = transition_nodes
        self.config.num_nodes = total_graph_nodes
        self.dim_manager.num_nodes = total_graph_nodes
        if hasattr(self.dim_manager, 'node_counts'):
            self.dim_manager.node_counts['wave'] = wave_nodes
            self.dim_manager.node_counts['target'] = target_nodes
            self.dim_manager.node_counts['transition'] = transition_nodes
            self.dim_manager.node_counts['spatial'] = total_graph_nodes
        
        # Create proper graph structure for spatial encoding
        from utils.graph_utils import get_pyg_graph
        
        # Update config with correct node counts (feature-based, not sequence-based)
        self.config.num_waves = wave_nodes
        self.config.num_targets = target_nodes  
        self.config.num_transitions = transition_nodes
        
        # Get graph structure (import already done above)
        graph_data = get_pyg_graph(self.config, wave_embedded.device)
        
        # Prepare node features using the converted spatial tensors
        # Aggregate across batch dimension: [batch, nodes, d_model] -> [nodes, d_model]
        x_dict = {
            'wave': wave_spatial.mean(dim=0),      # [wave_nodes, d_model]
            'target': target_spatial.mean(dim=0)   # [target_nodes, d_model]
        }
        
        # Add transition features (learnable parameters)
        transition_dim = getattr(self.config, 'd_model', 512)
        if not hasattr(self, 'transition_features'):
            transition_init = torch.randn(
                transition_nodes,
                transition_dim,
                device=embedded.device,
                dtype=embedded.dtype,
            )
            self.register_parameter('transition_features', nn.Parameter(transition_init))
        elif self.transition_features.size(0) != transition_nodes:
            transition_init = torch.randn(
                transition_nodes,
                transition_dim,
                device=embedded.device,
                dtype=embedded.dtype,
            )
            self.transition_features = nn.Parameter(transition_init)
        else:
            if self.transition_features.device != embedded.device or self.transition_features.dtype != embedded.dtype:
                self.transition_features.data = self.transition_features.data.to(embedded.device, dtype=embedded.dtype)
 
        transition_features_param = self.transition_features
        transition_broadcast = transition_features_param.unsqueeze(0).expand(batch_size, -1, -1)
        self._validate_node_tensor(transition_broadcast, "transition_broadcast", batch_size, transition_nodes)
        x_dict['transition'] = transition_broadcast.mean(dim=0)
        
        # Prepare topology features with correct dimensions matching x_dict
        t_dict = {
            'wave': torch.zeros_like(x_dict['wave']),        # [wave_nodes, d_model] 
            'target': torch.zeros_like(x_dict['target']),    # [target_nodes, d_model]
            'transition': torch.zeros_like(x_dict['transition'])  # [transition_nodes, d_model]
        }
        
        # Prepare edge indices
        edge_index_dict = {
            ('wave', 'interacts_with', 'transition'): graph_data['wave', 'interacts_with', 'transition'].edge_index,
            ('transition', 'influences', 'target'): graph_data['transition', 'influences', 'target'].edge_index
        }
        
        # Enhanced spatial encoding with dynamic graph components
        num_nodes = total_graph_nodes

        # MEMORY FIX 3 + 9: Efficient batch device placement
        device = combined_input.device
        
        # Batch device placement - more efficient than individual checks
        self._ensure_device_placement(device)
            
        # Update component dimensions if they've changed significantly
        # Only recreate if absolutely necessary to avoid memory fragmentation
        if (hasattr(self.dynamic_graph, 'num_waves') and 
            abs(self.dynamic_graph.num_waves - wave_nodes) > 2):
            print(f"Info: Updating dynamic graph dimensions: {self.dynamic_graph.num_waves} -> {wave_nodes}")
            # Only update the internal parameters, don't recreate the whole module
            self.dynamic_graph.num_waves = wave_nodes
            self.dynamic_graph.num_targets = target_nodes
            self.dynamic_graph.num_transitions = transition_nodes

        # Create dynamic adjacency matrix using converted spatial tensors
        node_features_dict = {
            'wave': wave_spatial,      # Use converted spatial tensor [batch, wave_nodes, d_model]
            'transition': transition_broadcast,
            'target': target_spatial   # Use converted spatial tensor [batch, target_nodes, d_model]
        }

        self._validate_node_feature_dict(node_features_dict, batch_size)
        
        # FIXED: Dynamic graph construction with proper combination of both components
        if getattr(self.config, 'enable_dynamic_graph', True):
            # First, get dynamic graph structure
            dyn_result = self.dynamic_graph(node_features_dict)
            if isinstance(dyn_result, (tuple, list)):
                base_adjacency, base_edge_weights = dyn_result[0], dyn_result[1]
            else:
                base_adjacency, base_edge_weights = dyn_result, None
            
            # Then, adaptively refine the structure (don't overwrite, combine)
            adapt_result = self.adaptive_graph(node_features_dict)
            if isinstance(adapt_result, (tuple, list)):
                adaptive_adjacency, adaptive_edge_weights = adapt_result[0], adapt_result[1]
            else:
                # Handle case where adaptive_graph returns a single tensor
                adaptive_adjacency, adaptive_edge_weights = adapt_result, None

            # Combine graphs using the new gated combiner if available
            if self.graph_combiner is not None:
                adjacency_matrix, edge_weights = self.graph_combiner(
                    base_adjacency, adaptive_adjacency, base_edge_weights, adaptive_edge_weights
                )
            else:
                # Fallback to original logic
                if base_adjacency is not None and adaptive_adjacency is not None:
                    if isinstance(base_adjacency, torch.Tensor) and isinstance(adaptive_adjacency, torch.Tensor):
                        base_weight = getattr(self.config, 'base_adjacency_weight', 0.7)
                        adaptive_weight = getattr(self.config, 'adaptive_adjacency_weight', 0.3)
                        adjacency_matrix = base_weight * base_adjacency + adaptive_weight * adaptive_adjacency
                    else:
                        adjacency_matrix = adaptive_adjacency
                    edge_weights = adaptive_edge_weights if adaptive_edge_weights is not None else base_edge_weights
                else:
                    adjacency_matrix = adaptive_adjacency if adaptive_adjacency is not None else base_adjacency
                    edge_weights = adaptive_edge_weights if adaptive_edge_weights is not None else base_edge_weights
        else:
            # Use simple adjacency matrix for now
            adjacency_matrix = self._create_adjacency_matrix(seq_len, num_nodes, combined_input.device)
            edge_weights = None

        self._validate_dynamic_graph_outputs(adjacency_matrix, edge_weights, num_nodes)
        
        # MEMORY FIX 4: Replace massive tensor expansion with memory-efficient processing
        # Original: spatiotemporal_input = embedded.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        # This creates a tensor of size [batch, seq, nodes, d_model] which can be huge
        
        if self._enable_memory_optimization:
            # Memory-efficient approach: process in chunks and use broadcasting
            # Instead of expanding to full size, we'll process node-wise
            spatiotemporal_input = self._create_memory_efficient_spatiotemporal_input(
                embedded, num_nodes, batch_size, seq_len
            )
        else:
            # Original approach for compatibility
            spatiotemporal_input = embedded.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        # MEMORY FIX 5: Structural positional encoding (already initialized in __init__)
        if getattr(self.config, 'enable_structural_pos_encoding', True):
            try:
                # Ensure structural_pos_encoding is on correct device
                try:
                    # Get device from module parameters
                    module_device = next(self.structural_pos_encoding.parameters()).device
                    if module_device != combined_input.device:
                        self.structural_pos_encoding = self.structural_pos_encoding.to(combined_input.device)
                except StopIteration:
                    # Module has no parameters, move it anyway
                    self.structural_pos_encoding = self.structural_pos_encoding.to(combined_input.device)

                # Use cached adjacency matrix if available
                cache_key = f"adj_matrix_{seq_len}_{num_nodes}"
                if cache_key in self._memory_cache:
                    adj_matrix_tensor = self._memory_cache[cache_key]
                else:
                    adj_matrix_tensor = self._create_adjacency_matrix(seq_len, num_nodes, combined_input.device)
                    if adj_matrix_tensor.numel() < 1e6:  # Cache if not too large
                        self._memory_cache[cache_key] = adj_matrix_tensor

                # Memory-efficient structural encoding
                d_model = spatiotemporal_input.size(-1)
                
                # Use cached base tensor if available
                base_cache_key = f"base_x_{num_nodes}_{d_model}"
                if base_cache_key in self._memory_cache:
                    base_x = self._memory_cache[base_cache_key]
                    if base_x.device != combined_input.device:
                        base_x = base_x.to(combined_input.device)
                else:
                    base_x = torch.zeros(num_nodes, d_model, device=combined_input.device, dtype=spatiotemporal_input.dtype)
                    if base_x.numel() < 1e5:  # Cache small tensors
                        self._memory_cache[base_cache_key] = base_x.clone()
                
                node_struct_encoding = self.structural_pos_encoding(base_x, adj_matrix_tensor)  # [num_nodes, d_model]
                
                # MEMORY FIX 8: Vectorized broadcasting (much more efficient than loops)
                if self._enable_memory_optimization:
                    # Vectorized approach - much faster and more memory efficient than nested loops
                    node_struct_encoding_expanded = node_struct_encoding.unsqueeze(0).unsqueeze(0)  # [1, 1, nodes, d_model]
                    spatiotemporal_input += node_struct_encoding_expanded  # Broadcasting handles the rest
                else:
                    # Original broadcasting approach
                    node_struct_encoding = node_struct_encoding.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
                    spatiotemporal_input = spatiotemporal_input + node_struct_encoding
                    
            except Exception as e:
                print(f"Info: Structural positional encoding skipped: {e}")
        # MEMORY FIX 6: Graph-aware positional encoding (reuse cached adjacency matrix)
        if getattr(self.config, 'enable_graph_positional_encoding', True):
            try:
                # Reuse cached adjacency matrix from structural encoding
                cache_key = f"adj_matrix_{seq_len}_{num_nodes}"
                if cache_key in self._memory_cache:
                    adj_matrix_tensor = self._memory_cache[cache_key]
                else:
                    adj_matrix_tensor = self._create_adjacency_matrix(seq_len, num_nodes, combined_input.device)
                    if adj_matrix_tensor.numel() < 1e6:
                        self._memory_cache[cache_key] = adj_matrix_tensor
                
                # Ensure graph_pos_encoding is on correct device
                try:
                    # Get device from module parameters
                    module_device = next(self.graph_pos_encoding.parameters()).device
                    if module_device != combined_input.device:
                        self.graph_pos_encoding = self.graph_pos_encoding.to(combined_input.device)
                except StopIteration:
                    # Module has no parameters, move it anyway
                    self.graph_pos_encoding = self.graph_pos_encoding.to(combined_input.device)
                
                pos_encoding = self.graph_pos_encoding(
                    batch_size, seq_len, num_nodes, adj_matrix_tensor, combined_input.device
                )
                spatiotemporal_input = spatiotemporal_input + pos_encoding
            except Exception as e:
                print(f"Info: Graph positional encoding skipped: {e}")
        # MEMORY FIX 12: Reuse cached adjacency matrix for spatiotemporal encoder
        cache_key = f"adj_matrix_{seq_len}_{num_nodes}"
        if cache_key in self._memory_cache:
            adj_matrix_tensor = self._memory_cache[cache_key]
        else:
            adj_matrix_tensor = self._create_adjacency_matrix(seq_len, num_nodes, combined_input.device)
            if adj_matrix_tensor.numel() < 1e6:
                self._memory_cache[cache_key] = adj_matrix_tensor
        
        # MEMORY FIX 11: Apply joint spatial-temporal encoding with optional gradient checkpointing
        if self._use_gradient_checkpointing and self.training:
            # Use gradient checkpointing for memory-compute tradeoff
            import torch.utils.checkpoint as checkpoint
            spatiotemporal_encoded = checkpoint.checkpoint(
                self.spatiotemporal_encoder, spatiotemporal_input, adj_matrix_tensor
            )
        else:
            spatiotemporal_encoded = self.spatiotemporal_encoder(
                spatiotemporal_input, adj_matrix_tensor
            )

        self._validate_spatiotemporal_encoding(spatiotemporal_encoded, batch_size, seq_len, num_nodes)
        
        # CRITICAL BUG FIX #1: Proper node feature extraction from spatiotemporal encoding
        # spatiotemporal_encoded shape: [batch, seq, nodes, d_model]
        # Node types should be differentiated by SPATIAL dimension (nodes), not TEMPORAL dimension (seq)
        
        # CRITICAL FIX: Create enhanced_x_dict from our converted spatial tensors
        # Use the properly converted spatial tensors instead of trying to extract from spatiotemporal_encoded
        enhanced_x_dict = {
            # Use converted spatial tensors, aggregated across batch dimension
            'wave': wave_spatial.mean(dim=0),        # [wave_nodes, d_model]
            'transition': self.transition_features,   # [transition_nodes, d_model] - learnable parameters
            'target': target_spatial.mean(dim=0)     # [target_nodes, d_model]
        }

        # Use the more robust get_pyg_graph() method with topology features
        from utils.graph_utils import get_pyg_graph
        
        # Update config with correct node counts for graph construction
        self.config.num_waves = wave_nodes
        self.config.num_targets = target_nodes
        self.config.num_transitions = transition_nodes
        
        graph_data = get_pyg_graph(self.config, combined_input.device)
        
        # IMPORTANT: Edge index convention fix
        # - get_pyg_graph() follows PyTorch Geometric standard: edge_index[0] = source, edge_index[1] = target
        # - Our graph attention layer expects: edge_index[0] = target, edge_index[1] = source
        # - Solution: Use .flip(0) to swap the convention
        # - This allows us to use the more robust get_pyg_graph() method with topology features
        enhanced_edge_index_dict = {
            ('wave', 'interacts_with', 'transition'): graph_data['wave', 'interacts_with', 'transition'].edge_index.flip(0),
            ('transition', 'influences', 'target'): graph_data['transition', 'influences', 'target'].edge_index.flip(0)
        }

        # Runtime validations for enhanced_x_dict and enhanced_edge_index_dict
        for nt in ['wave', 'transition', 'target']:
            if nt not in enhanced_x_dict:
                raise ValueError(f"Missing node type '{nt}' in enhanced_x_dict")
            x_nt = enhanced_x_dict[nt]
            if not isinstance(x_nt, torch.Tensor):
                raise ValueError(f"enhanced_x_dict['{nt}'] must be a Tensor, got {type(x_nt)}")
            if x_nt.dim() != 2:
                raise ValueError(f"enhanced_x_dict['{nt}'] must be 2D [num_nodes, d_model], got shape {tuple(x_nt.shape)}")
        d_w = enhanced_x_dict['wave'].size(1)
        d_t = enhanced_x_dict['transition'].size(1)
        d_g = enhanced_x_dict['target'].size(1)
        if not (d_w == d_t == d_g):
            raise ValueError(f"Mismatched feature dims across nodes: wave={d_w}, transition={d_t}, target={d_g}")

        def _validate_edges_pg(key, src_key, tgt_key):
            if key in enhanced_edge_index_dict:
                ei = enhanced_edge_index_dict[key]
                if not isinstance(ei, torch.Tensor):
                    raise ValueError(f"edge_index for {key} must be a Tensor, got {type(ei)}")
                if ei.dtype not in (torch.long, torch.int64):
                    raise ValueError(f"edge_index for {key} must be of dtype torch.long, got {ei.dtype}")
                if ei.dim() != 2 or ei.size(0) != 2:
                    raise ValueError(f"edge_index for {key} must have shape [2, E], got {tuple(ei.shape)}")
                if ei.numel() > 0:
                    # CRITICAL FIX: Correct convention - ei[0] = target, ei[1] = source
                    max_tgt = int(ei[0].max().item())  # ei[0] contains target indices
                    max_src = int(ei[1].max().item())  # ei[1] contains source indices
                    min_tgt = int(ei[0].min().item())  # ei[0] contains target indices
                    min_src = int(ei[1].min().item())  # ei[1] contains source indices
                    if min_src < 0 or min_tgt < 0:
                        raise ValueError(f"edge_index for {key} contains negative indices")
                    if max_src >= enhanced_x_dict[src_key].size(0):
                        raise ValueError(f"edge_index source index out of bounds for {key}: max {max_src} >= {enhanced_x_dict[src_key].size(0)}")
                    if max_tgt >= enhanced_x_dict[tgt_key].size(0):
                        raise ValueError(f"edge_index target index out of bounds for {key}: max {max_tgt} >= {enhanced_x_dict[tgt_key].size(0)}")

        _validate_edges_pg(('wave','interacts_with','transition'), 'wave', 'transition')
        _validate_edges_pg(('transition','influences','target'), 'transition', 'target')
        
        # Apply multi-head graph attention (config-gated with safe fallback)
        if getattr(self.config, 'enable_graph_attention', True):
            try:
                attended_features = self.graph_attention(enhanced_x_dict, enhanced_edge_index_dict)
            except Exception as e:
                print(f"Info: Graph attention skipped due to error: {e}")
                attended_features = enhanced_x_dict
        else:
            # Use enhanced features directly without graph attention
            attended_features = enhanced_x_dict

        # Apply spatial encoding based on encoder type
        use_dynamic_weights = getattr(self.config, 'use_dynamic_edge_weights', True) and isinstance(self.spatial_encoder, EnhancedPGAT_CrossAttn_Layer)
        if use_dynamic_weights and hasattr(self.spatial_encoder, 'forward'):
            # Enhanced spatial encoder with graph structure
            spatial_x_dict, spatial_t_dict = self.spatial_encoder(x_dict, t_dict, enhanced_edge_index_dict)
            spatial_encoded = {
                'wave': spatial_x_dict['wave'].unsqueeze(0).expand(batch_size, -1, -1) + attended_features['wave'],
                'transition': spatial_x_dict['transition'].unsqueeze(0).expand(batch_size, -1, -1) + attended_features['transition'],
                'target': spatial_x_dict['target'].unsqueeze(0).expand(batch_size, -1, -1) + attended_features['target']
            }
        else:
            # MEMORY FIX 10: Memory-efficient spatial encoding without large tensor expansions
            if self._enable_memory_optimization:
                # Process each node type and expand more efficiently
                spatial_encoded = {}
                for node_type in ['wave', 'transition', 'target']:
                    encoded_features = self.spatial_encoder(attended_features[node_type])  # [nodes, d_model]
                    # Use broadcasting instead of explicit expansion
                    spatial_encoded[node_type] = encoded_features.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # Original approach
                wave_features = self.spatial_encoder(attended_features['wave']).unsqueeze(0).expand(batch_size, -1, -1)
                transition_features = self.spatial_encoder(attended_features['transition']).unsqueeze(0).expand(batch_size, -1, -1)
                target_features = self.spatial_encoder(attended_features['target']).unsqueeze(0).expand(batch_size, -1, -1)
                spatial_encoded = {
                    'wave': wave_features,
                    'transition': transition_features,
                    'target': target_features
                }

        self._validate_node_tensor(spatial_encoded['wave'], 'spatial_wave', batch_size, wave_nodes)
        self._validate_node_tensor(spatial_encoded['transition'], 'spatial_transition', batch_size, transition_nodes)
        self._validate_node_tensor(spatial_encoded['target'], 'spatial_target', batch_size, target_nodes)
        
        # Store graph information if enabled
        if self.store_graph_info:
            adj_to_store = adjacency_matrix
            ew_to_store = edge_weights
            if isinstance(adj_to_store, (tuple, list)):
                if ew_to_store is None and len(adj_to_store) >= 2:
                    adj_candidate, ew_candidate = adj_to_store[0], adj_to_store[1]
                    adj_to_store, ew_to_store = adj_candidate, ew_candidate
                elif len(adj_to_store) >= 1:
                    adj_to_store = adj_to_store[0]
                else:
                    adj_to_store = None
            self.last_adjacency_matrix = adj_to_store
            self.last_edge_weights = ew_to_store
        
        # Temporal encoding with temporal attention
        # Extract target embeddings from spatial encoded output
        target_spatial = spatial_encoded['target']
        
        # Call temporal encoder robustly: prefer (query, keys, values), then fallbacks
        out = None
        try:
            # Try keyword arguments first (most explicit)
            out = self.temporal_encoder(query=target_spatial, key=target_spatial, value=target_spatial)
        except TypeError:
            try:
                # Fallback to positional Q, K, V
                out = self.temporal_encoder(target_spatial, target_spatial, target_spatial)
            except TypeError:
                try:
                    # Some temporal encoders may accept only a single tensor
                    out = self.temporal_encoder(target_spatial)
                except TypeError:
                    # Last resort: two-tensor signature
                    out = self.temporal_encoder(target_spatial, target_spatial)
        
        # Unpack output if needed
        temporal_encoded = out[0] if isinstance(out, tuple) else out
        
        # Combine spatial and temporal features
        # Use target spatial features for combination
        if temporal_encoded.shape == target_spatial.shape:
            final_embedding = temporal_encoded + target_spatial
        else:
            # Handle dimension mismatch by using only temporal encoded features
            final_embedding = temporal_encoded
        
        # Project features if dimension mismatch
        if final_embedding.size(-1) != self.d_model:
            if not hasattr(self, 'final_projection') or self.final_projection is None:
                self.final_projection = nn.Linear(final_embedding.size(-1), self.d_model).to(final_embedding.device)
            final_embedding = self.final_projection(final_embedding)
        
        self._validate_decoder_ready(final_embedding)

        # Decode to final output
        output = self.decoder(final_embedding)
        
        # MEMORY FIX 7: Optional memory cleanup after forward pass
        if self._enable_memory_optimization and len(self._memory_cache) > 10:
            # Clean up old cache entries to prevent memory buildup
            # Keep only the most recent entries
            cache_keys = list(self._memory_cache.keys())
            if len(cache_keys) > 10:
                for key in cache_keys[:-5]:  # Keep last 5 entries
                    del self._memory_cache[key]
        
        return output
    
    def configure_optimizer_loss(self, base_criterion: nn.Module, verbose: bool = False) -> nn.Module:
        """Select MixtureNLLLoss when MDN outputs are active.

        Args:
            base_criterion: Loss instantiated by experiment runner prior to model hook.
            verbose: When True, emit informational message about swapping the criterion.

        Returns:
            nn.Module: MixtureNLLLoss for probabilistic mode with MDN outputs, otherwise the original criterion.
        """
        if self._use_mdn_outputs and self.mixture_loss is not None:
            if verbose:
                print("PGAT using MixtureNLLLoss for MDN outputs")
            return self.mixture_loss
        return base_criterion
    
    def _validate_config(self) -> None:
        """Validate critical configuration attributes before component construction."""
        d_model = getattr(self.config, 'd_model', None)
        seq_len = getattr(self.config, 'seq_len', None)
        pred_len = getattr(self.config, 'pred_len', None)
        n_heads = getattr(self.config, 'n_heads', None)
        dropout = getattr(self.config, 'dropout', 0.0)

        for name, value in {
            'd_model': d_model,
            'seq_len': seq_len,
            'pred_len': pred_len,
        }.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"SOTA_Temporal_PGAT requires config.{name} as positive int, received {value!r}.")

        if not isinstance(n_heads, int) or n_heads <= 0:
            raise ValueError(f"SOTA_Temporal_PGAT requires config.n_heads as positive int, received {n_heads!r}.")

        d_model_int = cast(int, d_model)
        n_heads_int = cast(int, n_heads)
        if d_model_int % n_heads_int != 0:
            raise ValueError(
                f"SOTA_Temporal_PGAT expects d_model divisible by n_heads; received d_model={d_model_int}, n_heads={n_heads_int}."
            )

        if not isinstance(dropout, (float, int)) or not 0.0 <= float(dropout) <= 1.0:
            raise ValueError(f"SOTA_Temporal_PGAT requires dropout within [0, 1]; received {dropout!r}.")

    def _validate_forward_inputs(self, wave_window: torch.Tensor, target_window: torch.Tensor) -> None:
        """Ensure forward inputs share batch/device/dtype and expected dimensionality."""
        if not isinstance(wave_window, torch.Tensor) or not isinstance(target_window, torch.Tensor):
            raise TypeError("wave_window and target_window must be torch.Tensor instances.")

        if wave_window.ndim != 3 or target_window.ndim != 3:
            raise ValueError("wave_window and target_window must be 3D tensors shaped [batch, seq, features].")

        if wave_window.size(0) != target_window.size(0):
            raise ValueError(
                f"Batch size mismatch between wave ({wave_window.size(0)}) and target ({target_window.size(0)}) windows."
            )

        if wave_window.device != target_window.device:
            raise ValueError("wave_window and target_window must reside on the same device.")

        if wave_window.dtype != target_window.dtype:
            raise ValueError("wave_window and target_window must share the same dtype.")

        if wave_window.size(1) <= 0 or target_window.size(1) <= 0:
            raise ValueError("wave_window and target_window must contain at least one timestep.")

        if wave_window.size(2) <= 0 or target_window.size(2) <= 0:
            raise ValueError("wave_window and target_window must contain non-empty feature dimensions.")

    def _validate_embedding_output(self, embedded: torch.Tensor, batch_size: int, seq_len: int) -> None:
        """Validate embedded tensor matches expected batch/sequence/model dimensions."""
        if not isinstance(embedded, torch.Tensor):
            raise TypeError("Embedded representation must be a torch.Tensor.")

        if embedded.ndim != 3:
            raise ValueError("Embedded representation must be 3D shaped [batch, sequence, d_model].")

        if embedded.size(0) != batch_size or embedded.size(1) != seq_len:
            raise ValueError(
                f"Embedded tensor shape mismatch: expected batch={batch_size}, seq={seq_len}; got {tuple(embedded.shape)}."
            )

        if embedded.size(2) != self.d_model:
            raise ValueError(
                f"Embedded tensor last dimension must equal d_model={self.d_model}; received {embedded.size(2)}."
            )

        if not self.dim_manager.validate_tensor_shape(embedded, 'embedded', batch_size):
            raise ValueError("Embedded tensor does not align with dimension manager expectations.")

    def _validate_node_tensor(self, tensor: torch.Tensor, name: str, batch_size: int, expected_nodes: int) -> None:
        """Validate node-specific tensor adheres to [batch, nodes, d_model] layout."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")

        if tensor.ndim != 3:
            raise ValueError(f"{name} must be 3D shaped [batch, nodes, d_model]; got shape {tuple(tensor.shape)}.")

        if tensor.size(0) != batch_size or tensor.size(1) != expected_nodes:
            raise ValueError(
                f"{name} expected batch={batch_size}, nodes={expected_nodes}; received {tuple(tensor.shape)}."
            )

        if tensor.size(2) != self.d_model:
            raise ValueError(f"{name} last dimension must equal d_model={self.d_model}; received {tensor.size(2)}.")

    def _validate_node_feature_dict(self, node_features: Dict[str, torch.Tensor], batch_size: int) -> None:
        """Validate node feature dictionary prior to dynamic graph construction."""
        required_nodes = ('wave', 'transition', 'target')
        for node_type in required_nodes:
            if node_type not in node_features:
                raise ValueError(f"node_features missing required key '{node_type}'.")

        node_counts = getattr(self.dim_manager, 'node_counts', {})
        for node_type in required_nodes:
            expected_nodes = node_counts.get(node_type)
            if expected_nodes is None:
                continue
            self._validate_node_tensor(node_features[node_type], f"{node_type}_features", batch_size, expected_nodes)

    def _validate_dynamic_graph_outputs(
        self,
        adjacency_matrix: Any,
        edge_weights: Optional[Any],
        expected_nodes: int,
    ) -> None:
        """Ensure dynamic/adaptive graph outputs expose expected topology data."""
        if adjacency_matrix is None:
            return

        if isinstance(adjacency_matrix, torch.Tensor):
            if adjacency_matrix.ndim != 2:
                raise ValueError("Adjacency matrix tensor must be 2D.")
            if adjacency_matrix.size(0) != expected_nodes or adjacency_matrix.size(1) != expected_nodes:
                raise ValueError(
                    f"Adjacency matrix tensor must be square with dimension {expected_nodes}; got {tuple(adjacency_matrix.shape)}."
                )
        elif hasattr(adjacency_matrix, 'edge_types'):
            required_edges = [
                ('wave', 'interacts_with', 'transition'),
                ('transition', 'influences', 'target'),
            ]
            missing_edges = [edge for edge in required_edges if edge not in getattr(adjacency_matrix, 'edge_types', [])]
            if missing_edges:
                raise ValueError(f"Dynamic graph output missing edge types: {missing_edges}.")
        else:
            raise TypeError(
                f"Unsupported adjacency_matrix type {type(adjacency_matrix)}; expected torch.Tensor or HeteroData-like object."
            )

        if isinstance(edge_weights, dict):
            for edge_type, weights in edge_weights.items():
                if not isinstance(weights, torch.Tensor):
                    raise TypeError(f"Edge weights for {edge_type} must be torch.Tensor, received {type(weights)}.")
                if weights.ndim <= 0:
                    raise ValueError(f"Edge weights for {edge_type} must contain at least one dimension.")
                if not torch.isfinite(weights).all():
                    raise ValueError(f"Edge weights for {edge_type} contain non-finite values.")

    def _validate_spatiotemporal_encoding(
        self,
        encoded: torch.Tensor,
        batch_size: int,
        seq_len: int,
        num_nodes: int,
    ) -> None:
        """Validate joint spatiotemporal encoding output dimensions."""
        if not isinstance(encoded, torch.Tensor):
            raise TypeError("Spatiotemporal encoding output must be a torch.Tensor.")

        if encoded.ndim != 4:
            raise ValueError("Spatiotemporal encoding output must be 4D shaped [batch, seq, nodes, d_model].")

        if encoded.size(0) != batch_size or encoded.size(1) != seq_len or encoded.size(2) != num_nodes:
            raise ValueError(
                f"Spatiotemporal encoding output shape mismatch; expected ({batch_size}, {seq_len}, {num_nodes}, *)."
            )

        if encoded.size(3) != self.d_model:
            raise ValueError(
                f"Spatiotemporal encoding output last dimension must equal d_model={self.d_model}; got {encoded.size(3)}."
            )

    def _validate_decoder_ready(self, embedding: torch.Tensor) -> None:
        """Validate final embedding prior to decoder invocation."""
        if not isinstance(embedding, torch.Tensor):
            raise TypeError("Decoder input must be a torch.Tensor.")

        if embedding.ndim != 3:
            raise ValueError("Decoder input must be 3D shaped [batch, nodes, d_model].")

        if embedding.size(2) != self.d_model:
            raise ValueError(
                f"Decoder input last dimension must equal d_model={self.d_model}; received {embedding.size(2)}."
            )

    def _initialize_embedding(self, config: Any) -> nn.Module:
        """Select an embedding module with robust fallbacks for tensor-only inputs.

        The production architecture registers a bespoke InitialEmbedding that
        expects typed node dictionaries and structural metadata. When the
        registry entry is missing (e.g., tests or stripped configs) we fall
        back to tensor-friendly projections while documenting which path was
        chosen for downstream debugging.
        """
        registry = EmbeddingRegistry()
        d_model = getattr(config, 'd_model', 512)

        try:
            embedding_cls = registry.get('initial_embedding')
        except ValueError as exc:
            print(f"Info: initial_embedding not registered; using fallback ({exc}).")
        else:
            try:
                candidate = embedding_cls(config)
                if self._module_accepts_single_tensor(candidate):
                    self._embedding_source = 'registry'
                    return candidate
                raise TypeError('initial_embedding requires non-tensor inputs')
            except Exception as exc:  # noqa: BLE001 - guardrail for registry experiments
                print(f"Info: Registry initial embedding unusable ({exc}); falling back.")

        inferred_dim = self._infer_input_feature_dim(config)
        if inferred_dim is not None:
            self._fallback_input_dim = inferred_dim
            self._embedding_source = 'linear_fallback'
            return nn.Sequential(
                nn.LayerNorm(inferred_dim),
                nn.Linear(inferred_dim, d_model)
            )

        self._embedding_source = 'lazy_linear_fallback'
        return _LazyLinearEmbedding(d_model)

    def _infer_input_feature_dim(self, config: Any) -> Optional[int]:
        """Infer the raw feature dimension used by the fallback embedding path."""
        candidate_attrs = (
            'enc_in',
            'wave_feature_dim',
            'input_feature_dim',
            'feature_dim',
        )
        for attr in candidate_attrs:
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return value

        manager_enc_in = getattr(self.dim_manager, 'enc_in', None)
        if isinstance(manager_enc_in, int) and manager_enc_in > 0:
            return manager_enc_in

        if hasattr(self.dim_manager, 'num_targets') and hasattr(self.dim_manager, 'num_covariates'):
            combined = self.dim_manager.num_targets + self.dim_manager.num_covariates
            if combined > 0:
                return combined

        return None

    @staticmethod
    def _module_accepts_single_tensor(module: nn.Module) -> bool:
        """Return True when module.forward can process a single positional tensor input."""
        signature = inspect.signature(module.forward)
        parameters = [param for name, param in signature.parameters.items() if name != 'self']
        if not parameters:
            return False

        first = parameters[0]
        if first.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            for param in parameters[1:]:
                if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) and param.default is inspect._empty:
                    return False
            return True

        if first.kind == inspect.Parameter.VAR_POSITIONAL:
            return True

        return False
    
    def _create_edge_index(self, num_source: int, num_target: int, device: torch.device) -> torch.Tensor:
        """
        Create edge indices for heterogeneous graph attention with dimension validation
        """
        try:
            # Use dimension manager's safe edge index creation
            edge_index = self.dim_manager.create_safe_edge_index(num_source, num_target, device)

            
            # Additional validation for this specific edge type
            if edge_index.shape[0] != 2:
                raise ValueError(f"Edge index should have 2 rows, got {edge_index.shape[0]}")
            
            if edge_index.shape[1] != num_source * num_target:
                raise ValueError(f"Edge index should have {num_source * num_target} edges, got {edge_index.shape[1]}")
            
            return edge_index
            
        except Exception as e:
            print(f"Warning: Safe edge index creation failed ({e}), falling back to basic method")
            print(f"Dimensions: source={num_source}, target={num_target}")

            
            # Fallback to original method with bounds checking
            if num_source <= 0 or num_target <= 0:
                print(f"Invalid node counts: source={num_source}, target={num_target}")
                return torch.empty((2, 0), device=device, dtype=torch.long)
            
            # Create fully connected bipartite graph (original method)
            source_nodes = torch.arange(num_source, device=device).repeat(num_target)
            target_nodes = torch.arange(num_target, device=device).repeat_interleave(num_source)
            # CRITICAL FIX: Swap order to match enhanced_pgat_layer convention: edge_index[0] = target, edge_index[1] = source
            edge_index = torch.stack([target_nodes, source_nodes], dim=0)
            return edge_index
    
    def _create_adjacency_matrix(self, seq_len, num_nodes, device):
        """Create adjacency matrix for temporal connections
        Returns a [num_nodes, num_nodes] adjacency matrix
        """
        # Create a simple fully connected graph for demonstration
        # In practice, this should be based on actual spatial relationships
        adj_matrix = torch.ones(num_nodes, num_nodes, device=device)
        # Remove self-loops for better graph structure
        adj_matrix.fill_diagonal_(0)
        # Add small values to diagonal to avoid numerical issues
        diagonal_value = getattr(self.config, 'adjacency_diagonal_value', 0.1)
        adj_matrix.fill_diagonal_(diagonal_value)
        return adj_matrix
    
    def get_graph_structure(self):
        """
        Return the current graph structure for analysis
        """
        if hasattr(self, 'last_adjacency_matrix') and self.last_adjacency_matrix is not None:
            return {
                'adjacency_matrix': self.last_adjacency_matrix,
                'edge_weights': self.last_edge_weights
            }
        return None
    
    def set_graph_info_storage(self, store: bool = True):
        """
        Enable/disable storing graph information
        """
        self.store_graph_info = store
    
    def get_attention_weights(self, wave_window, target_window, graph):
        """
        Get attention weights for visualization and analysis
        """
        if self._embedding_source != 'registry':
            raise RuntimeError("Attention weight inspection requires a registry-backed embedding module.")
        with torch.no_grad():
            embedded = self.embedding(wave_window, target_window, graph)

            spatial_encoded, spatial_attn = self.spatial_encoder(
                embedded['wave_embedded'],
                embedded['target_embedded']
            )

            return {
                'spatial_attention': spatial_attn,
                'embedded_features': embedded
            }
    
    # REMOVED: Duplicate methods - these are defined properly later in the file
    
    def _create_memory_efficient_spatiotemporal_input(self, embedded, num_nodes, batch_size, seq_len):
        """
        MEMORY FIX: Create spatiotemporal input without massive tensor expansion.
        
        Instead of creating [batch, seq, nodes, d_model] tensor directly,
        we create a more memory-efficient representation.
        """
        device = embedded.device
        d_model = embedded.size(-1)
        
        # Check if we can use cached version
        cache_key = f"spatiotemporal_{batch_size}_{seq_len}_{num_nodes}_{d_model}"
        if cache_key in self._memory_cache:
            cached_tensor = self._memory_cache[cache_key]
            if cached_tensor.device == device:
                # CRITICAL FIX: Clone cached tensor before modifying to prevent data corruption
                result_tensor = cached_tensor.clone()
                result_tensor[:, :, 0, :] = embedded  # Update first node with embedded features
                return result_tensor
        
        # Create memory-efficient version
        # Instead of expanding embedded to all nodes, we create a sparse representation
        spatiotemporal_input = torch.zeros(
            batch_size, seq_len, num_nodes, d_model, 
            device=device, dtype=embedded.dtype
        )
        
        # Only fill the first "node" with actual embedded features
        # Other nodes will be handled by the spatiotemporal encoder
        spatiotemporal_input[:, :, 0, :] = embedded
        
        # Cache for reuse if tensor is not too large
        if spatiotemporal_input.numel() < 1e7:  # Only cache if < 10M elements
            self._memory_cache[cache_key] = spatiotemporal_input.clone()
        
        return spatiotemporal_input
    
    def clear_memory_cache(self):
        """Clear memory cache to free up memory."""
        self._memory_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_stats(self):
        """Get current memory usage statistics."""
        stats = {
            'cached_tensors': len(self._memory_cache),
            'model_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'memory_optimization_enabled': self._enable_memory_optimization,
            'gradient_checkpointing_enabled': self._use_gradient_checkpointing,
            'chunk_size': self._chunk_size
        }
        
        if torch.cuda.is_available():
            stats['cuda_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
            stats['cuda_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
            stats['cuda_max_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024**2)
        
        return stats
    
    def _ensure_device_placement(self, device):
        """Efficiently ensure all components are on the correct device."""
        components = [
            'dynamic_graph', 'adaptive_graph', 'spatiotemporal_encoder',
            'graph_pos_encoding', 'graph_attention', 'structural_pos_encoding',
            'temporal_pos_encoding'
        ]
        
        for comp_name in components:
            if hasattr(self, comp_name):
                comp = getattr(self, comp_name)
                if comp is not None and hasattr(comp, 'device') and comp.device != device:
                    setattr(self, comp_name, comp.to(device))
    
    def enable_memory_optimization(self, enable=True, chunk_size=32, use_gradient_checkpointing=False):
        """Enable or disable memory optimization features."""
        self._enable_memory_optimization = enable
        self._chunk_size = chunk_size
        self._use_gradient_checkpointing = use_gradient_checkpointing
        if not enable:
            self.clear_memory_cache()
    
    def enable_mixed_precision(self, enable=True):
        """Enable mixed precision training for additional memory savings."""
        if enable:
            # Convert model to half precision for forward pass
            # Note: This should be used with torch.cuda.amp.autocast() during training
            for module in self.modules():
                # CRITICAL FIX: Actually call .half() on modules (logic was backwards before)
                if not isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                    # Keep normalization layers in FP32 for stability
                    module.half()
        else:
            # CRITICAL FIX: Add path to revert to FP32
            self.float()
        return self
    
    def configure_for_training(self):
        """Configure model for training mode with memory optimization."""
        self.train()
        self.enable_memory_optimization(True)
        return self
    
    def configure_for_inference(self):
        """Configure model for inference mode."""
        self.eval()
        # Keep memory optimization for inference too
        return self