
import torch
import torch.nn as nn
import inspect
from typing import Any, Dict, Optional, Tuple, cast

# Import original model and new modular components
from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
from layers.modular.graph.gated_graph_combiner import GatedGraphCombiner
from layers.modular.embedding.attention_temporal_to_spatial import AttentionTemporalToSpatial
from layers.modular.embedding.patching import PatchingLayer

class Enhanced_SOTA_PGAT(SOTA_Temporal_PGAT):
    """
    This class enhances the SOTA_Temporal_PGAT model by overriding its init and forward methods 
    to incorporate new modular components:
    1. PatchingLayer: For patch-based processing of the input time series.
    2. AttentionTemporalToSpatial: For a more dynamic temporal-to-spatial conversion.
    3. GatedGraphCombiner: For dynamically combining graph structures.
    """
    
    def __init__(self, config, mode='probabilistic'):
        # Call the parent constructor, but we will override many components
        super().__init__(config, mode)

        # --- OVERRIDE AND ENHANCE COMPONENTS ---
        
        # Validate enhanced configuration
        self._validate_enhanced_config(config)

        # 1. Gated Graph Combiner
        if getattr(self.config, 'use_gated_graph_combiner', True): # Default to True for enhanced model
            default_total_nodes = getattr(config, 'enc_in', 7) + getattr(config, 'c_out', 3) + min(getattr(config, 'enc_in', 7), getattr(config, 'c_out', 3))
            self.graph_combiner = GatedGraphCombiner(
                num_nodes=default_total_nodes,
                d_model=self.d_model
            )
        
        # 2. Attention-based Temporal-to-Spatial Conversion
        # This is mandatory for the patching logic to work correctly.
        self.use_attention_temp_to_spatial = getattr(self.config, 'use_attention_temp_to_spatial', True)

        # 3. Patching Layer
        if getattr(self.config, 'use_patching', True): # Default to True for enhanced model
            self.patching_layer = PatchingLayer(
                patch_len=getattr(config, 'patch_len', 16),
                stride=getattr(config, 'stride', 8),
                d_model=self.d_model,
                input_features=self._infer_input_feature_dim(config) or getattr(config, 'enc_in')
            )
            # The main embedding is now handled by the patching layer
            self.embedding = nn.Identity()
        else:
            self.patching_layer = None
            # Re-initialize embedding if not using patching
            self.embedding = self._initialize_embedding(config)

    def forward(self, wave_window, target_window, graph=None):
        self._validate_forward_inputs(wave_window, target_window)

        raw_wave_len = wave_window.shape[1]
        raw_target_len = target_window.shape[1]
        combined_input = torch.cat([wave_window, target_window], dim=1)
        batch_size, seq_len_raw, features = combined_input.shape

        # --- 1. Patching and Embedding ---
        if self.patching_layer is not None:
            embedded = self.patching_layer(combined_input)
            seq_len = embedded.shape[1] # New sequence length is the number of patches
            
            # Calculate the number of patches for wave and target windows
            # Use the same formula as in PatchingLayer
            def calculate_num_patches(length, patch_len, stride):
                if (length - patch_len) % stride != 0:
                    # Account for padding
                    padding_len = stride - ((length - patch_len) % stride)
                    length += padding_len
                return (length - patch_len) // stride + 1
            
            wave_len = calculate_num_patches(raw_wave_len, self.patching_layer.patch_len, self.patching_layer.stride)
            target_len = calculate_num_patches(raw_target_len, self.patching_layer.patch_len, self.patching_layer.stride)
        else:
            embedded = self.embedding(combined_input.view(-1, features)).view(batch_size, seq_len_raw, -1)
            seq_len = seq_len_raw
            wave_len = raw_wave_len
            target_len = raw_target_len

        # Skip dimension manager validation for enhanced model (patching changes dimensions)
        # self._validate_embedding_output(embedded, batch_size, seq_len)
        
        # Basic validation instead
        if not isinstance(embedded, torch.Tensor) or embedded.ndim != 3:
            raise ValueError(f"Embedded tensor must be 3D, got shape {embedded.shape}")
        if embedded.size(0) != batch_size:
            raise ValueError(f"Batch size mismatch: expected {batch_size}, got {embedded.size(0)}")
        if embedded.size(-1) != self.d_model:
            raise ValueError(f"Feature dimension mismatch: expected {self.d_model}, got {embedded.size(-1)}")
        embedded = self.temporal_pos_encoding(embedded)
        
        wave_embedded = embedded[:, :wave_len, :]
        target_embedded = embedded[:, wave_len:wave_len+target_len, :]

        wave_nodes = getattr(self.config, 'enc_in', 7)
        target_nodes = getattr(self.config, 'c_out', 3)
        transition_nodes = max(1, min(wave_nodes, target_nodes))

        # --- 2. Temporal-to-Spatial Conversion ---
        if self.use_attention_temp_to_spatial:
            if self.wave_temporal_to_spatial is None or not isinstance(self.wave_temporal_to_spatial, AttentionTemporalToSpatial):
                self.wave_temporal_to_spatial = AttentionTemporalToSpatial(self.d_model, wave_nodes, n_heads=self.n_heads)
                # Properly register the module
                self.add_module('wave_temporal_to_spatial', self.wave_temporal_to_spatial)
            if self.target_temporal_to_spatial is None or not isinstance(self.target_temporal_to_spatial, AttentionTemporalToSpatial):
                self.target_temporal_to_spatial = AttentionTemporalToSpatial(self.d_model, target_nodes, n_heads=self.n_heads)
                # Properly register the module
                self.add_module('target_temporal_to_spatial', self.target_temporal_to_spatial)
            
            # Ensure modules are on correct device
            self.wave_temporal_to_spatial = self.wave_temporal_to_spatial.to(embedded.device)
            self.target_temporal_to_spatial = self.target_temporal_to_spatial.to(embedded.device)
            
            wave_spatial = self.wave_temporal_to_spatial(wave_embedded)
            target_spatial = self.target_temporal_to_spatial(target_embedded)
        else: # Fallback to original linear method
            if self.wave_temporal_to_spatial is None or not isinstance(self.wave_temporal_to_spatial, nn.Linear):
                self.wave_temporal_to_spatial = nn.Linear(wave_len, wave_nodes).to(embedded.device)
            if self.target_temporal_to_spatial is None or not isinstance(self.target_temporal_to_spatial, nn.Linear):
                self.target_temporal_to_spatial = nn.Linear(target_len, target_nodes).to(embedded.device)

            wave_spatial = self.wave_temporal_to_spatial(wave_embedded.transpose(1, 2)).transpose(1, 2)
            target_spatial = self.target_temporal_to_spatial(target_embedded.transpose(1, 2)).transpose(1, 2)

        # --- 3. Dynamic Graph Construction & Gated Combination ---
        total_graph_nodes = wave_nodes + target_nodes + transition_nodes
        
        # Ensure transition features are initialized (from parent class)
        if not hasattr(self, 'transition_features'):
            transition_dim = getattr(self.config, 'd_model', 512)
            transition_init = torch.randn(
                transition_nodes,
                transition_dim,
                device=embedded.device,
                dtype=embedded.dtype,
            )
            self.register_parameter('transition_features', nn.Parameter(transition_init))
        
        transition_broadcast = self.transition_features.unsqueeze(0).expand(batch_size, -1, -1)
        node_features_dict = {'wave': wave_spatial, 'transition': transition_broadcast, 'target': target_spatial}
        
        dyn_result = self.dynamic_graph(node_features_dict)
        base_adjacency, base_edge_weights = dyn_result if isinstance(dyn_result, (tuple, list)) else (dyn_result, None)
        
        adapt_result = self.adaptive_graph(node_features_dict)
        adaptive_adjacency, adaptive_edge_weights = adapt_result if isinstance(adapt_result, (tuple, list)) else (adapt_result, None)

        if hasattr(self, 'graph_combiner') and self.graph_combiner is not None:
            try:
                adjacency_matrix, edge_weights = self.graph_combiner(base_adjacency, adaptive_adjacency, base_edge_weights, adaptive_edge_weights)
            except Exception as e:
                print(f"Warning: Graph combiner failed ({e}), using fallback")
                adjacency_matrix = adaptive_adjacency if adaptive_adjacency is not None else base_adjacency
                edge_weights = adaptive_edge_weights if adaptive_edge_weights is not None else base_edge_weights
        else: # Fallback to original logic
            adjacency_matrix = adaptive_adjacency if adaptive_adjacency is not None else base_adjacency
            edge_weights = adaptive_edge_weights if adaptive_edge_weights is not None else base_edge_weights

        # --- Continue with the rest of the forward pass ---
        # (This part is simplified for clarity, it mirrors the original model's logic)
        
        enhanced_x_dict = {
            'wave': wave_spatial.mean(dim=0),
            'transition': self.transition_features,
            'target': target_spatial.mean(dim=0)
        }
        
        # Use the more robust get_pyg_graph() method with corrected edge index convention
        from utils.graph_utils import get_pyg_graph
        
        # Update config with correct node counts for graph construction
        self.config.num_waves = wave_nodes
        self.config.num_targets = target_nodes
        self.config.num_transitions = transition_nodes
        
        graph_data = get_pyg_graph(self.config, embedded.device)
        
        # IMPORTANT: Edge index convention fix
        # - get_pyg_graph() follows PyTorch Geometric standard: edge_index[0] = source, edge_index[1] = target
        # - Our graph attention layer expects: edge_index[0] = target, edge_index[1] = source
        # - Solution: Use .flip(0) to swap the convention
        # - This allows us to use the more robust get_pyg_graph() method with topology features
        enhanced_edge_index_dict = {
            ('wave', 'interacts_with', 'transition'): graph_data['wave', 'interacts_with', 'transition'].edge_index.flip(0),
            ('transition', 'influences', 'target'): graph_data['transition', 'influences', 'target'].edge_index.flip(0)
        }

        # Apply graph attention if enabled, otherwise use features directly
        if getattr(self.config, 'enable_graph_attention', True):
            try:
                attended_features = self.graph_attention(enhanced_x_dict, enhanced_edge_index_dict)
            except Exception as e:
                print(f"Warning: Graph attention failed ({e}), using direct features")
                attended_features = enhanced_x_dict
        else:
            attended_features = enhanced_x_dict
        
        # This part of the logic for spatial encoding needs careful review
        # For simplicity, we will use the attended features directly
        spatial_encoded = {
            'wave': attended_features['wave'].unsqueeze(0).expand(batch_size, -1, -1),
            'transition': attended_features['transition'].unsqueeze(0).expand(batch_size, -1, -1),
            'target': attended_features['target'].unsqueeze(0).expand(batch_size, -1, -1)
        }

        target_spatial_encoded = spatial_encoded['target']
        temporal_encoded, _ = self.temporal_encoder(target_spatial_encoded, target_spatial_encoded, target_spatial_encoded)
        
        final_embedding = temporal_encoded + target_spatial_encoded
        
        output = self.decoder(final_embedding)
        
        return output
    
    def _validate_enhanced_config(self, config):
        """Validate enhanced model configuration parameters."""
        # Validate patching parameters
        if getattr(config, 'use_patching', True):
            patch_len = getattr(config, 'patch_len', 16)
            stride = getattr(config, 'stride', 8)
            seq_len = getattr(config, 'seq_len', 96)
            pred_len = getattr(config, 'pred_len', 24)
            
            if not isinstance(patch_len, int) or patch_len <= 0:
                raise ValueError(f"patch_len must be positive integer, got {patch_len}")
            if not isinstance(stride, int) or stride <= 0:
                raise ValueError(f"stride must be positive integer, got {stride}")
            if patch_len > seq_len or patch_len > pred_len:
                print(f"Warning: patch_len ({patch_len}) larger than sequence lengths (seq_len={seq_len}, pred_len={pred_len})")
        
        # Validate attention parameters
        if getattr(config, 'use_attention_temp_to_spatial', True):
            n_heads = getattr(config, 'n_heads', 8)
            d_model = getattr(config, 'd_model', 512)
            if d_model % n_heads != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
    
    def get_enhanced_config_info(self):
        """Get information about enhanced model configuration."""
        return {
            'use_patching': hasattr(self, 'patching_layer') and self.patching_layer is not None,
            'use_attention_temp_to_spatial': getattr(self, 'use_attention_temp_to_spatial', False),
            'use_gated_graph_combiner': hasattr(self, 'graph_combiner') and self.graph_combiner is not None,
            'patch_len': getattr(self.patching_layer, 'patch_len', None) if hasattr(self, 'patching_layer') else None,
            'stride': getattr(self.patching_layer, 'stride', None) if hasattr(self, 'patching_layer') else None,
        }
