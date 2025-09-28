import torch
import torch.nn as nn
import inspect
from typing import Optional

from layers.modular.attention.registry import AttentionRegistry
from layers.modular.decoder.registry import DecoderRegistry
from layers.modular.graph.registry import GraphComponentRegistry
from layers.modular.embedding.registry import EmbeddingRegistry
from layers.modular.graph.dynamic_graph import DynamicGraphConstructor, AdaptiveGraphStructure
from layers.modular.attention.multihead_graph_attention import MultiHeadGraphAttention, GraphTransformerLayer
from layers.modular.encoder.spatiotemporal_encoding import JointSpatioTemporalEncoding, AdaptiveSpatioTemporalEncoder
from layers.modular.embedding.graph_positional_encoding import GraphAwarePositionalEncoding, HierarchicalGraphPositionalEncoding
# New enhanced components
from layers.modular.decoder.mixture_density_decoder import MixtureDensityDecoder, MixtureNLLLoss
from layers.modular.attention.autocorr_temporal_attention import AutoCorrTemporalAttention
from layers.modular.embedding.structural_positional_encoding import StructuralPositionalEncoding
from layers.modular.embedding.enhanced_temporal_encoding import EnhancedTemporalEncoding
from layers.modular.graph.enhanced_pgat_layer import EnhancedPGAT_CrossAttn_Layer
from utils.graph_aware_dimension_manager import GraphAwareDimensionManager, create_graph_aware_dimension_manager

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
        
        # Track probabilistic configuration early for loss/decoder wiring
        self._use_mdn_outputs = bool(getattr(config, 'use_mixture_density', True) and mode != 'standard')
        self.mixture_loss = MixtureNLLLoss() if self._use_mdn_outputs else None
        
        # Initialize registries
        self.attention_registry = AttentionRegistry()
        self.decoder_registry = DecoderRegistry()
        self.graph_registry = GraphComponentRegistry()
        
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
        
        # Initialize dynamic graph components
        self.dynamic_graph = None  # Will be initialized dynamically
        self.adaptive_graph = None
        self.spatiotemporal_encoder = None
        self.graph_pos_encoding = None
        self.graph_attention = None
        self.feature_projection = None
        
        # Enhanced spatial encoder with dynamic edge weights
        # Note: Previously forced disabled due to index errors; now gated via config with safe fallback
        use_dynamic_weights = getattr(config, 'use_dynamic_edge_weights', True)
        try:
            if use_dynamic_weights:
                self.spatial_encoder = EnhancedPGAT_CrossAttn_Layer(
                    d_model=config.d_model,
                    num_heads=getattr(config, 'n_heads', 8),
                    use_dynamic_weights=True
                )
                self.enhanced_pgat_enabled = True
            else:
                raise RuntimeError('Enhanced PGAT disabled via config')
        except Exception as e:
            print(f"Info: Enhanced PGAT unavailable ({e}); falling back to Linear spatial encoder")
            self.enhanced_pgat_enabled = False
            # Use simple linear layer for spatial encoding
            self.spatial_encoder = nn.Linear(config.d_model, config.d_model)
            # Feature projection for concatenated features when enhanced PGAT is disabled
            total_features = config.d_model * 3  # wave + transition + target
            self.feature_projection = nn.Linear(total_features, config.d_model)
        
        # Enhanced temporal encoder with autocorrelation attention
        use_autocorr = getattr(config, 'use_autocorr_attention', True)
        if use_autocorr:
            self.temporal_encoder = AutoCorrTemporalAttention(
                d_model=config.d_model,
                n_heads=getattr(config, 'n_heads', 8),
                dropout=getattr(config, 'dropout', 0.1),
                factor=getattr(config, 'autocorr_factor', 1)
            )
        else:
            self.temporal_encoder = self.attention_registry.get('temporal_attention')(
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
            self.decoder = self.decoder_registry.get('custom_standard')(config.d_model)
        elif self._use_mdn_outputs:
            self.decoder = MixtureDensityDecoder(
                d_model=config.d_model,
                pred_len=getattr(config, 'pred_len', 96),
                num_components=getattr(config, 'mdn_components', 3)
            )
        else:
            self.decoder = self.decoder_registry.get('probabilistic')(config.d_model)
        
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
    
    def forward(self, wave_window, target_window, graph):
        """
        Forward pass through the SOTA Temporal PGAT model
        
        Args:
            wave_window: Wave input data
            target_window: Target input data
            graph: Graph adjacency matrix
            
        Returns:
            Model output (single tensor for standard mode, tuple for probabilistic mode)
        """
        # Initial embedding - concatenate wave and target windows
        # wave_window: [batch, seq_len, features], target_window: [batch, pred_len, features]
        combined_input = torch.cat([wave_window, target_window], dim=1)  # [batch, seq_len+pred_len, features]
        batch_size, seq_len, features = combined_input.shape
        
        # Check if input is already embedded (features == d_model) or needs embedding
        if features == getattr(self, 'd_model', 512):
            embedded = combined_input  # Already embedded
        else:
            embedded = self.embedding(combined_input.view(-1, features)).view(batch_size, seq_len, -1)
        
        # Apply enhanced temporal positional encoding
        embedded = self.temporal_pos_encoding(embedded)
        
        # Spatial encoding with graph attention
        # Split embedded tensor back into wave and target parts
        wave_len = wave_window.shape[1]
        target_len = target_window.shape[1]
        wave_embedded = embedded[:, :wave_len, :]
        target_embedded = embedded[:, wave_len:wave_len+target_len, :]
        
        graph_counts = getattr(self.dim_manager, 'node_counts', {})
        wave_nodes = graph_counts.get('wave', wave_len)
        target_nodes = graph_counts.get('target', target_len)
        transition_nodes = max(1, graph_counts.get('transition', min(wave_nodes, target_nodes)))
        spatial_nodes = graph_counts.get('spatial', wave_nodes + target_nodes + transition_nodes)
        total_graph_nodes = max(spatial_nodes, wave_nodes + target_nodes + transition_nodes)
        
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
        
        # Create a simple config for graph construction if not available
        if not hasattr(self.config, 'num_waves'):
            self.config.num_waves = wave_len
            self.config.num_targets = target_len
            self.config.num_transitions = min(wave_len, target_len)
        
        # Get graph structure
        graph_data = get_pyg_graph(self.config, wave_embedded.device)
        
        # Prepare node features for graph processing
        x_dict = {
            'wave': wave_embedded.mean(dim=0),  # [seq_len, d_model] -> [num_waves, d_model]
            'target': target_embedded.mean(dim=0)  # [pred_len, d_model] -> [num_targets, d_model]
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
        x_dict['transition'] = transition_broadcast.mean(dim=0)
        
        # Prepare topology features
        t_dict = {
            'wave': graph_data['wave'].t if hasattr(graph_data['wave'], 't') else torch.zeros_like(x_dict['wave']),
            'target': graph_data['target'].t if hasattr(graph_data['target'], 't') else torch.zeros_like(x_dict['target']),
            'transition': torch.zeros_like(x_dict['transition'])
        }
        
        # Prepare edge indices
        edge_index_dict = {
            ('wave', 'interacts_with', 'transition'): graph_data['wave', 'interacts_with', 'transition'].edge_index,
            ('transition', 'influences', 'target'): graph_data['transition', 'influences', 'target'].edge_index
        }
        
        # Enhanced spatial encoding with dynamic graph components
        num_nodes = total_graph_nodes

        # Initialize dynamic graph components if not exists
        if self.dynamic_graph is None:
            self.dynamic_graph = DynamicGraphConstructor(
                d_model=self.d_model,
                num_waves=wave_nodes,
                num_targets=target_nodes,
                num_transitions=transition_nodes
            ).to(combined_input.device)

        if self.adaptive_graph is None:
            self.adaptive_graph = AdaptiveGraphStructure(
                d_model=self.d_model,
                num_waves=wave_nodes,
                num_targets=target_nodes,
                num_transitions=transition_nodes
            ).to(combined_input.device)

        if self.spatiotemporal_encoder is None:
            self.spatiotemporal_encoder = AdaptiveSpatioTemporalEncoder(
                d_model=self.d_model,
                max_seq_len=seq_len,
                max_nodes=num_nodes,
                num_layers=2,
                num_heads=self.n_heads
            ).to(combined_input.device)

        if self.graph_pos_encoding is None:
            self.graph_pos_encoding = GraphAwarePositionalEncoding(
                d_model=self.d_model,
                max_nodes=num_nodes,
                max_seq_len=seq_len,
                encoding_types=['distance', 'centrality', 'spectral']
            ).to(combined_input.device)

        if self.graph_attention is None:
            self.graph_attention = MultiHeadGraphAttention(
                d_model=self.d_model,
                num_heads=self.n_heads
            ).to(combined_input.device)

        # Create dynamic adjacency matrix
        node_features_dict = {
            'wave': wave_embedded,
            'transition': transition_broadcast,
            'target': target_embedded
        }
        
        # Dynamic graph construction with enhanced edge weights
        # Replace hard-disabled path with config gate and fallback
        if getattr(self.config, 'enable_dynamic_graph', True):
            dyn_result = self.dynamic_graph(node_features_dict)
            if isinstance(dyn_result, (tuple, list)):
                adjacency_matrix, edge_weights = dyn_result[0], dyn_result[1]
            else:
                adjacency_matrix, edge_weights = dyn_result, None
            # Update graph structure adaptively
            adapt_result = self.adaptive_graph(node_features_dict)
            if isinstance(adapt_result, (tuple, list)):
                adjacency_matrix, edge_weights = adapt_result[0], adapt_result[1]
            else:
                adjacency_matrix = adapt_result
        else:
            # Use simple adjacency matrix for now
            adjacency_matrix = self._create_adjacency_matrix(seq_len, num_nodes, combined_input.device)
            edge_weights = None
        
        # Reshape embedded features for spatial-temporal processing
        spatiotemporal_input = embedded.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [batch, seq, nodes, d_model]
        # Add structural positional encoding for enhanced graph structure awareness
        if getattr(self.config, 'enable_structural_pos_encoding', True):
            try:
                # Lazily initialize structural positional encoding module if needed
                if not hasattr(self, 'structural_pos_encoding') or self.structural_pos_encoding is None:
                    self.structural_pos_encoding = StructuralPositionalEncoding(
                        d_model=self.d_model,
                        num_eigenvectors=getattr(self.config, 'max_eigenvectors', 16),
                        dropout=getattr(self.config, 'dropout', 0.1),
                        learnable_projection=True
                    ).to(combined_input.device)

                # Use homogeneous simple path: compute node-wise structural encoding once and broadcast
                adj_matrix_tensor = self._create_adjacency_matrix(seq_len, num_nodes, combined_input.device)
                d_model = spatiotemporal_input.size(-1)
                base_x = torch.zeros(num_nodes, d_model, device=combined_input.device, dtype=spatiotemporal_input.dtype)
                node_struct_encoding = self.structural_pos_encoding(base_x, adj_matrix_tensor)  # [num_nodes, d_model]
                # Broadcast across batch and sequence length
                node_struct_encoding = node_struct_encoding.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
                spatiotemporal_input = spatiotemporal_input + node_struct_encoding
            except Exception as e:
                print(f"Info: Structural positional encoding skipped: {e}")
        # Add graph-aware positional encoding (config gated)
        if getattr(self.config, 'enable_graph_positional_encoding', True):
            try:
                adj_matrix_tensor = self._create_adjacency_matrix(seq_len, num_nodes, combined_input.device)
                pos_encoding = self.graph_pos_encoding(
                    batch_size, seq_len, num_nodes, adj_matrix_tensor, combined_input.device
                )
                spatiotemporal_input = spatiotemporal_input + pos_encoding
            except Exception as e:
                print(f"Info: Graph positional encoding skipped: {e}")
        # Create adjacency matrix for spatiotemporal encoder
        adj_matrix_tensor = self._create_adjacency_matrix(seq_len, num_nodes, combined_input.device)
        
        # Apply joint spatial-temporal encoding
        spatiotemporal_encoded = self.spatiotemporal_encoder(
            spatiotemporal_input, adj_matrix_tensor
        )
        
        # Derive enhanced node features and edge indices before validations
        enhanced_x_dict = {
            'wave': spatiotemporal_encoded[:, :wave_len, :, :].mean(dim=2).mean(dim=0),
            'transition': spatiotemporal_encoded[:, wave_len:, :, :].mean(dim=2).mean(dim=0),
            'target': spatiotemporal_encoded[:, wave_len:, :, :].mean(dim=2).mean(dim=0)
        }

        enhanced_edge_index_dict = {
            ('wave', 'interacts_with', 'transition'): self._create_edge_index(wave_nodes, transition_nodes, combined_input.device),
            ('transition', 'influences', 'target'): self._create_edge_index(transition_nodes, target_nodes, combined_input.device)
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
                    max_src = int(ei[0].max().item())
                    max_tgt = int(ei[1].max().item())
                    min_src = int(ei[0].min().item())
                    min_tgt = int(ei[1].min().item())
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
            spatial_x_dict, spatial_t_dict = self.spatial_encoder(x_dict, t_dict, edge_index_dict)
            spatial_encoded = {
                'wave': spatial_x_dict['wave'].unsqueeze(0).expand(batch_size, -1, -1) + attended_features['wave'],
                'transition': spatial_x_dict['transition'].unsqueeze(0).expand(batch_size, -1, -1) + attended_features['transition'],
                'target': spatial_x_dict['target'].unsqueeze(0).expand(batch_size, -1, -1) + attended_features['target']
            }
        else:
            # Simple linear spatial encoder
            wave_features = self.spatial_encoder(attended_features['wave']).unsqueeze(0).expand(batch_size, -1, -1)
            transition_features = self.spatial_encoder(attended_features['transition']).unsqueeze(0).expand(batch_size, -1, -1)
            target_features = self.spatial_encoder(attended_features['target']).unsqueeze(0).expand(batch_size, -1, -1)
            spatial_encoded = {
                'wave': wave_features,
                'transition': transition_features,
                'target': target_features
            }
        
        # Store graph information if enabled
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
            out = self.temporal_encoder(query=target_spatial, keys=target_spatial, values=target_spatial)
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
        
        # Decode to final output
        return self.decoder(final_embedding)
    
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
    
    def _initialize_embedding(self, config: object) -> nn.Module:
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

    def _infer_input_feature_dim(self, config: object) -> Optional[int]:
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
            edge_index = torch.stack([source_nodes, target_nodes], dim=0)
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
        adj_matrix.fill_diagonal_(0.1)
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
    
    def configure_for_training(self):
        """
        Configure model for training mode
        """
        self.train()
        return self
    
    def configure_for_inference(self):
        """
        Configure model for inference mode
        """
        self.eval()
        return self

        self.dim_manager.edge_bounds[('wave', 'transition')] = (wave_nodes, transition_nodes)
        self.dim_manager.edge_bounds[('transition', 'target')] = (transition_nodes, target_nodes)
        self.dim_manager.edge_bounds['spatial'] = (total_graph_nodes, total_graph_nodes)
        
        if self._embedding_source != 'registry':
            raise RuntimeError("Attention weight inspection requires a registry-backed embedding module.")