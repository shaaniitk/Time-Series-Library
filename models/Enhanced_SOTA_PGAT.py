import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from typing import Any, Dict, List, Optional, Tuple, cast

# Import original model and new modular components
from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
from layers.modular.graph.gated_graph_combiner import GatedGraphCombiner
from layers.modular.embedding.hierarchical_mapper import HierarchicalTemporalSpatialMapper
from layers.modular.embedding.multi_scale_patching import MultiScalePatchingComposer
from layers.modular.graph.stochastic_learner import StochasticGraphLearner
from layers.modular.decoder.mixture_density_decoder import MixtureDensityDecoder, MixtureNLLLoss
from utils.graph_utils import (
    convert_hetero_to_dense_adj,
    prepare_graph_proposal,
    validate_graph_proposals,
    adjacency_to_edge_indices,
)


class Enhanced_SOTA_PGAT(SOTA_Temporal_PGAT):
    """
    This class enhances the SOTA_Temporal_PGAT model by overriding its init and forward methods 
    to incorporate new modular components:
    1. MultiScalePatchingComposer: For multi-scale patch-based processing.
    2. HierarchicalTemporalSpatialMapper: For a more advanced temporal-to-spatial conversion.
    3. StochasticGraphLearner: For learning a stochastic graph structure.
    4. GatedGraphCombiner: A meta-controller to combine multiple graph proposals.
    5. MixtureDensityDecoder: For probabilistic forecasting.
    """
    
    def __init__(self, config, mode='probabilistic'):
        # Ensure config has required attributes for parent class
        self._ensure_config_attributes(config)
        
        super().__init__(config, mode)

        # --- OVERRIDE AND ENHANCE COMPONENTS ---

        # Initialize internal logging
        self.internal_logs: Dict[str, Any] = {}
        
        # CRITICAL FIX: Pre-allocate all projection layers to prevent dynamic creation
        self.context_projection_layers = nn.ModuleDict()
        
        # Pre-allocate common projection sizes to avoid dynamic layer creation
        common_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
        for size in common_sizes:
            self.context_projection_layers[f'proj_{size}'] = nn.Linear(size, self.d_model)
        
        # Pre-allocate context fusion layer with reasonable max size
        max_fusion_dim = self.d_model * 6  # Reasonable upper bound for fusion
        self.context_fusion_layer = nn.Linear(max_fusion_dim, self.d_model)

        # Determine wave feature count early and use consistently across components
        self.num_wave_features = getattr(config, 'num_wave_features', None)
        if self.num_wave_features is None:
            total_features = getattr(config, 'enc_in', 7)
            target_features = getattr(config, 'c_out', 3)
            self.num_wave_features = max(1, total_features - target_features)

        wave_nodes_default = self.num_wave_features
        target_nodes_default = getattr(config, 'c_out', 3)
        transition_nodes_default = max(1, min(wave_nodes_default, target_nodes_default))

        self.enable_phase_features = getattr(config, 'enable_phase_features', True)
        if self.enable_phase_features:
            self.phase_feature_projector = nn.Linear(6, self.d_model)

        self.enable_delayed_influence = getattr(config, 'enable_delayed_influence', True)
        self.delayed_max_lag = max(1, int(getattr(config, 'delayed_max_lag', 3)))
        if self.enable_delayed_influence:
            self.delay_feature_projector = nn.Linear(self.delayed_max_lag, self.d_model)
            self.delay_wave_to_transition = nn.Parameter(torch.randn(wave_nodes_default, transition_nodes_default))

        self.enable_group_interactions = getattr(config, 'enable_group_interactions', True)
        if self.enable_group_interactions:
            self.group_interaction_wave = nn.Parameter(torch.randn(wave_nodes_default, transition_nodes_default))
            self.group_interaction_transition = nn.Parameter(torch.randn(transition_nodes_default, target_nodes_default))
            self.group_interaction_scale = nn.Parameter(torch.tensor(1.0))

        self._validate_enhanced_config(config)

        total_nodes = wave_nodes_default + target_nodes_default + transition_nodes_default

        # 1. Stochastic Graph Learner
        if getattr(self.config, 'use_stochastic_learner', True):
            self.stochastic_learner = StochasticGraphLearner(
                d_model=self.d_model,
                num_nodes=total_nodes
            )
        else:
            self.stochastic_learner = None

        # 2. Gated Graph Combiner (as a meta-controller)
        if getattr(self.config, 'use_gated_graph_combiner', True):
            num_graphs = 2 + (1 if self.stochastic_learner is not None else 0) # Base, Adaptive, Stochastic
            self.graph_combiner = GatedGraphCombiner(
                num_nodes=total_nodes,
                d_model=self.d_model,
                num_graphs=num_graphs
            )
        
        # 3. Hierarchical Temporal-to-Spatial Conversion
        self.use_hierarchical_mapper = getattr(self.config, 'use_hierarchical_mapper', True)
        if self.use_hierarchical_mapper:
            self.wave_temporal_to_spatial = HierarchicalTemporalSpatialMapper(
                d_model=self.d_model, 
                num_nodes=self.num_wave_features,
                n_heads=getattr(config, 'n_heads', 8)
            )
            self.target_temporal_to_spatial = HierarchicalTemporalSpatialMapper(
                d_model=self.d_model, 
                num_nodes=getattr(config, 'c_out', 3),
                n_heads=getattr(config, 'n_heads', 8)
            )

        # 4. Multi-Scale Patching Layer (Separate for wave and target)
        if getattr(self.config, 'use_multi_scale_patching', True):
            # Adaptive patch configurations based on sequence lengths
            seq_len = getattr(config, 'seq_len', 24)
            pred_len = getattr(config, 'pred_len', 6)
            
            # Create adaptive patch configs for wave (longer sequence)
            wave_patch_configs = self._create_adaptive_patch_configs(seq_len)
            
            # Create adaptive patch configs for target (shorter sequence)  
            target_patch_configs = self._create_adaptive_patch_configs(pred_len)
            
            # Feature dimensions for patching already determined above
            
            num_target_features = getattr(config, 'c_out', 3)
            
            self.wave_patching_composer = MultiScalePatchingComposer(
                patch_configs=wave_patch_configs,
                d_model=self.d_model,
                input_features=self.num_wave_features,
                num_latents=getattr(config, 'num_wave_patch_latents', 64),
                n_heads=getattr(config, 'n_heads', 8)
            )
            self.target_patching_composer = MultiScalePatchingComposer(
                patch_configs=target_patch_configs,
                d_model=self.d_model,
                input_features=num_target_features,
                num_latents=getattr(config, 'num_target_patch_latents', 24),
                n_heads=getattr(config, 'n_heads', 8)
            )
            # Use Identity for patched mode (patching composers handle embedding)
            self.embedding = nn.Identity()
        else:
            self.wave_patching_composer = None
            self.target_patching_composer = None
            # Use proper embedding for non-patched mode
            self.embedding = self._initialize_embedding(config)

        # 5. Decoder - Fixed MDN wiring
        if getattr(config, 'use_mixture_decoder', True):
            self.decoder = MixtureDensityDecoder(
                d_model=self.d_model,
                pred_len=getattr(config, 'pred_len', 24),
                num_components=getattr(config, 'mdn_components', 3),
                num_targets=getattr(config, 'c_out', 3)
            )
            # Initialize mixture loss with multivariate support
            multivariate_mode = getattr(config, 'mixture_multivariate_mode', 'independent')
            self.mixture_loss: Optional[MixtureNLLLoss] = MixtureNLLLoss(
                multivariate_mode=multivariate_mode
            )
        else:
            self.decoder = nn.Linear(self.d_model, getattr(config, 'c_out', 3))
            self.mixture_loss = None


    def forward(self, wave_window, target_window, graph=None):
        self._validate_forward_inputs(wave_window, target_window)
        batch_size = wave_window.shape[0]

        # --- 1. Patching and Embedding ---
        wave_patched_outputs = None
        target_patched_outputs = None
        
        if self.wave_patching_composer is not None and self.target_patching_composer is not None:
            # Extract appropriate features for each composer
            # Wave composer gets the configured number of wave features
            wave_features = wave_window[:, :, :self.num_wave_features]
            # Target composer gets the next c_out target features
            target_start_idx = self.num_wave_features
            target_end_idx = target_start_idx + getattr(self.config, 'c_out', 3)
            target_features = target_window[:, :, target_start_idx:target_end_idx]
            
            # Patching composers handle their own temporal encoding
            wave_embedded, wave_patched_outputs = self.wave_patching_composer(wave_features)
            target_embedded, target_patched_outputs = self.target_patching_composer(target_features)
            # Skip redundant temporal encoding for patched data
        else:
            # Apply embedding and temporal encoding for non-patched data
            wave_embedded = self.embedding(wave_window.reshape(-1, wave_window.shape[-1])).view(batch_size, wave_window.shape[1], -1)
            target_embedded = self.embedding(target_window.reshape(-1, target_window.shape[-1])).view(batch_size, target_window.shape[1], -1)
            # Apply temporal encoding only for non-patched data
            wave_embedded = self.temporal_pos_encoding(wave_embedded)
            target_embedded = self.temporal_pos_encoding(target_embedded)

        wave_nodes = self.num_wave_features
        target_nodes = getattr(self.config, 'c_out', 3)
        transition_nodes = max(1, min(wave_nodes, target_nodes))

        # --- 2. Temporal-to-Spatial Conversion ---
        if self.use_hierarchical_mapper:
            # CRITICAL FIX: Align sequence lengths before hierarchical mapping
            wave_embedded, target_embedded = self._align_sequence_lengths(wave_embedded, target_embedded)
            wave_spatial = self.wave_temporal_to_spatial(wave_embedded)
            target_spatial = self.target_temporal_to_spatial(target_embedded)
        else: 
            wave_spatial = wave_embedded.mean(dim=1).unsqueeze(1).expand(-1, wave_nodes, -1)
            target_spatial = target_embedded.mean(dim=1).unsqueeze(1).expand(-1, target_nodes, -1)

        # Use the same wave features that were used for patching
        if self.wave_patching_composer is not None:
            wave_features_for_augment = wave_window[:, :, :self.num_wave_features]
        else:
            wave_features_for_augment = wave_window
        wave_spatial = self._augment_wave_features(wave_features_for_augment, wave_spatial)

        # --- 3. Dynamic Graph Construction & Gated Combination ---
        if not hasattr(self, 'transition_features'):
            transition_dim = self.d_model  # Use self.d_model consistently
            transition_init = torch.randn(transition_nodes, transition_dim, device=wave_embedded.device, dtype=wave_embedded.dtype)
            self.register_parameter('transition_features', nn.Parameter(transition_init))
        
        transition_broadcast = self.transition_features.unsqueeze(0).expand(batch_size, -1, -1)
        delayed_transition = self._compute_delayed_influence_features(
            wave_window,
            wave_spatial,
            transition_nodes,
        )
        if delayed_transition is not None:
            transition_broadcast = transition_broadcast + delayed_transition

        node_features_dict = {'wave': wave_spatial, 'transition': transition_broadcast, 'target': target_spatial}
        all_node_features = torch.cat([wave_spatial, transition_broadcast, target_spatial], dim=1)

        # Get total nodes for consistent tensor shapes
        total_nodes = wave_nodes + transition_nodes + target_nodes
        
        # Get graph proposals using improved utilities with validation
        
        # Base graph (dynamic) with validation
        dyn_result = self.dynamic_graph(node_features_dict)
        dyn_hetero, dyn_weights = self._validate_graph_output(dyn_result, "dynamic_graph")
        dyn_proposal = prepare_graph_proposal(dyn_hetero, dyn_weights, batch_size, total_nodes, preserve_weights=True)
        
        # Adaptive graph with validation
        adapt_result = self.adaptive_graph(node_features_dict)
        adapt_hetero, adapt_weights = self._validate_graph_output(adapt_result, "adaptive_graph")
        adapt_proposal = prepare_graph_proposal(adapt_hetero, adapt_weights, batch_size, total_nodes, preserve_weights=True)

        # Prepare graph proposals list for gated combiner
        graph_proposals = [dyn_proposal, adapt_proposal]

        higher_order = self._build_higher_order_adjacency(
            wave_spatial,
            transition_nodes,
            target_nodes,
            total_nodes,
        )
        if higher_order is not None:
            hyper_adj, hyper_weights = higher_order
            hyper_proposal = prepare_graph_proposal(
                hyper_adj,
                hyper_weights,
                batch_size,
                total_nodes,
                preserve_weights=True,
            )
            graph_proposals.append(hyper_proposal)
        
        # Handle stochastic learner - integration strategy: append to proposals
        stochastic_adj = None
        if self.stochastic_learner:
            stoch_adj, stoch_logits = self.stochastic_learner(all_node_features, self.training)
            self.latest_stochastic_loss = self.stochastic_learner.regularization_loss(stoch_logits)
            
            # Prepare stochastic proposal and append to list
            stoch_proposal = prepare_graph_proposal(stoch_adj, None, batch_size, total_nodes, preserve_weights=True)
            graph_proposals.append(stoch_proposal)
            stochastic_adj = stoch_proposal[0]
        
        # Validate all proposals before gated combination
        proposals_valid = validate_graph_proposals(graph_proposals, batch_size, total_nodes)

        # Combine graphs using the restored gated combiner contract
        if hasattr(self, 'graph_combiner') and self.graph_combiner is not None and proposals_valid:
            # Create rich context tensor preserving multi-scale and stochastic information
            # Instead of simple mean, use attention-weighted pooling to preserve important features
            context = self._create_rich_context(all_node_features, wave_patched_outputs, target_patched_outputs)  # [batch_size, d_model]
            
            # DEBUG: Memory check before graph_combiner
            import torch
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated() / 1024**3
                reserved_before = torch.cuda.memory_reserved() / 1024**3
                print(f"🔍 MEMORY BEFORE graph_combiner: Allocated={allocated_before:.2f}GB, Reserved={reserved_before:.2f}GB")
            else:
                import psutil
                process = psutil.Process()
                cpu_before = process.memory_info().rss / 1024**3
                print(f"🔍 MEMORY BEFORE graph_combiner: CPU={cpu_before:.2f}GB")
            
            # Debug input shapes
            print(f"🔍 INPUT SHAPES: graph_proposals={len(graph_proposals)} items, context={context.shape}")
            for i, proposal in enumerate(graph_proposals):
                if isinstance(proposal, tuple) and len(proposal) >= 2:
                    adj, weights = proposal[0], proposal[1]
                    print(f"   proposal[{i}]: adj={adj.shape}, weights={weights.shape}")
            
            # The improved gated combiner can handle variable number of graphs
            adjacency_matrix, edge_weights = self.graph_combiner(graph_proposals, context)
            
            # DEBUG: Memory check after graph_combiner
            if torch.cuda.is_available():
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                reserved_after = torch.cuda.memory_reserved() / 1024**3
                print(f"✅ MEMORY AFTER graph_combiner: Allocated={allocated_after:.2f}GB, Reserved={reserved_after:.2f}GB")
                print(f"📊 MEMORY DELTA: Allocated={allocated_after-allocated_before:+.2f}GB, Reserved={reserved_after-reserved_before:+.2f}GB")
            else:
                cpu_after = process.memory_info().rss / 1024**3
                print(f"✅ MEMORY AFTER graph_combiner: CPU={cpu_after:.2f}GB")
                print(f"📊 MEMORY DELTA: CPU={cpu_after-cpu_before:+.2f}GB")
            
            # Debug output shapes
            print(f"✅ OUTPUT SHAPES: adjacency_matrix={adjacency_matrix.shape}, edge_weights={edge_weights.shape}")
            self.internal_logs = {
                'graph_combination': 'success',
                'num_proposals': len(graph_proposals),
                'includes_stochastic': stochastic_adj is not None,
                'proposals_valid': proposals_valid
            }
        else: 
            # Use adaptive graph as fallback
            adjacency_matrix = adapt_proposal[0]
            edge_weights = adapt_proposal[1]
            self.internal_logs = {
                'graph_combination': 'no_combiner' if not hasattr(self, 'graph_combiner') else 'invalid_proposals',
                'proposals_valid': proposals_valid
            }

        # --- Use LEARNED adjacency for message passing (CRITICAL FIX) ---
        
        # Convert learned adjacency matrix to edge indices for graph attention
        adjacency_result = adjacency_to_edge_indices(
            adjacency_matrix, wave_nodes, target_nodes, transition_nodes, edge_weights
        )

        edge_index_batches, edge_weight_batches = self._normalize_edge_batch_outputs(
            adjacency_result,
            batch_size,
        )
        self.internal_logs['edge_weights_preserved'] = any(
            weight is not None for weight in edge_weight_batches
        )

        enhanced_x_dict = {
            'wave': wave_spatial,
            'transition': transition_broadcast,
            'target': target_spatial,
        }

        if getattr(self.config, 'enable_graph_attention', True):
            batch_attended_features = {'wave': [], 'transition': [], 'target': []}

            for b in range(batch_size):
                single_batch_x_dict = {
                    'wave': enhanced_x_dict['wave'][b],
                    'transition': enhanced_x_dict['transition'][b],
                    'target': enhanced_x_dict['target'][b],
                }
                edge_index_single = edge_index_batches[b]
                edge_weight_single = edge_weight_batches[b]

                attended_single = self.graph_attention(
                    single_batch_x_dict,
                    edge_index_single,
                    edge_weight_single,
                )

                for key in batch_attended_features:
                    batch_attended_features[key].append(attended_single[key])

            spatial_encoded = {
                'wave': torch.stack(batch_attended_features['wave'], dim=0),
                'transition': torch.stack(batch_attended_features['transition'], dim=0),
                'target': torch.stack(batch_attended_features['target'], dim=0),
            }
        else:
            spatial_encoded = enhanced_x_dict

        target_spatial_encoded = spatial_encoded['target']
        
        # Use the same temporal encoder calling pattern as base model
        out = self.temporal_encoder(target_spatial_encoded, target_spatial_encoded, target_spatial_encoded)
        
        # Unpack output if needed
        temporal_encoded = out[0] if isinstance(out, tuple) else out
        
        final_embedding = temporal_encoded + target_spatial_encoded
        
        if isinstance(self.decoder, MixtureDensityDecoder):
            # Fixed: Use correct MDN output format
            means, log_stds, log_weights = self.decoder(final_embedding)
            return means, log_stds, log_weights
        else:
            output = self.decoder(final_embedding)
            return output

    def loss(self, forward_output, targets):
        if isinstance(self.decoder, MixtureDensityDecoder):
            # Fixed: Use correct MDN loss computation
            means, log_stds, log_weights = forward_output
            if self.mixture_loss is None:
                raise RuntimeError("Mixture loss requested but not initialized")
            return self.mixture_loss(forward_output, targets)
        else:
            # Standard MSE loss for the base decoder
            loss_fn = nn.MSELoss()
            return loss_fn(forward_output, targets)

    def get_regularization_loss(self):
        loss = 0
        if hasattr(self, 'latest_stochastic_loss'):
            loss += self.latest_stochastic_loss
        return loss
    
    def configure_optimizer_loss(self, base_criterion, verbose=False):
        """Configure the appropriate loss function for the model."""
        if isinstance(self.decoder, MixtureDensityDecoder):
            if self.mixture_loss is None:
                raise RuntimeError("Mixture loss requested but not initialized")
            if verbose:
                print("Enhanced PGAT using MixtureNLLLoss for MDN outputs")
            return self.mixture_loss
        else:
            if verbose:
                print("Enhanced PGAT using standard loss function")
            return base_criterion

    def _ensure_config_attributes(self, config):
        """Ensure config has all required attributes for parent class."""
        # CRITICAL FIX: Use actual training values instead of small defaults
        required_attrs = {
            'seq_len': getattr(config, 'seq_len', 256),  # Use training sequence length
            'pred_len': getattr(config, 'pred_len', 24), # Use training prediction length
            'enc_in': getattr(config, 'enc_in', 118),    # Use actual feature count
            'c_out': getattr(config, 'c_out', 4),        # Use actual target count
            'd_model': getattr(config, 'd_model', 128),   # Use training model dimension
            'n_heads': getattr(config, 'n_heads', 4),    # Use training head count
            'dropout': getattr(config, 'dropout', 0.1)
        }
        
        for attr, default_value in required_attrs.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                setattr(config, attr, default_value)
        
        # Enhanced PGAT specific attributes
        enhanced_attrs = {
            'num_wave_features': None,  # Will be inferred if not provided
            'use_multi_scale_patching': True,
            'use_hierarchical_mapper': True,
            'use_stochastic_learner': True,
            'use_gated_graph_combiner': True,
            'use_mixture_decoder': True
        }
        
        for attr, default_value in enhanced_attrs.items():
            if not hasattr(config, attr):
                setattr(config, attr, default_value)

    def _validate_enhanced_config(self, config):
        """Validate enhanced model configuration parameters."""
        if getattr(config, 'use_multi_scale_patching', True):
            patch_configs = getattr(config, 'patch_configs', [])
            if not isinstance(patch_configs, list) or not all(isinstance(c, dict) for c in patch_configs):
                raise ValueError("patch_configs must be a list of dictionaries.")
            
            # Validate feature dimensions make sense
            if hasattr(self, 'num_wave_features'):
                total_features = getattr(config, 'enc_in', 7)
                target_features = getattr(config, 'c_out', 3)
                if self.num_wave_features + target_features > total_features:
                    raise ValueError(
                        f"num_wave_features ({self.num_wave_features}) + c_out ({target_features}) "
                        f"cannot exceed enc_in ({total_features})"
                    )

        if getattr(config, 'use_hierarchical_mapper', True):
            n_heads = getattr(config, 'n_heads', 8)
            d_model = getattr(config, 'd_model', 512)
            if d_model % n_heads != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        if getattr(config, 'use_mixture_decoder', True):
            if not hasattr(config, 'c_out') or not hasattr(config, 'd_model'):
                raise ValueError("config must have c_out and d_model for MixtureDensityDecoder")

    def _create_adaptive_patch_configs(self, seq_len: int):
        """Create patch configurations that are compatible with the sequence length."""
        configs = []
        
        # Ensure patch lengths don't exceed sequence length
        max_patch_len = seq_len // 2  # At most half the sequence length
        
        if seq_len >= 8:
            # Small patches (fine-grained)
            patch_len = min(4, max_patch_len)
            if patch_len >= 2:
                configs.append({'patch_len': patch_len, 'stride': max(1, patch_len // 2)})
        
        if seq_len >= 12:
            # Medium patches
            patch_len = min(8, max_patch_len)
            if patch_len >= 4:
                configs.append({'patch_len': patch_len, 'stride': max(2, patch_len // 2)})
        
        if seq_len >= 16:
            # Large patches (coarse-grained)
            patch_len = min(12, max_patch_len)
            if patch_len >= 6:
                configs.append({'patch_len': patch_len, 'stride': max(3, patch_len // 2)})
        
        # Fallback: if no configs generated, create a minimal one
        if not configs:
            patch_len = max(1, seq_len // 3)
            stride = max(1, patch_len // 2)
            configs.append({'patch_len': patch_len, 'stride': stride})
        
        return configs

    def _create_rich_context(self, all_node_features: torch.Tensor, 
                           wave_patched_outputs: Optional[List[torch.Tensor]] = None,
                           target_patched_outputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Create rich context tensor that preserves multi-scale and stochastic information
        instead of simple averaging that loses important features.
        
        Args:
            all_node_features: [batch_size, total_nodes, d_model]
            wave_patched_outputs: List of wave patch outputs (optional)
            target_patched_outputs: List of target patch outputs (optional)
            
        Returns:
            Rich context tensor [batch_size, d_model]
        """
        batch_size, total_nodes, d_model = all_node_features.shape
        
        # CRITICAL FIX: Use pre-allocated attention layer
        if not hasattr(self, 'context_attention'):
            self.context_attention = nn.Linear(d_model, 1)

        attention_logits = self.context_attention(all_node_features)
        attention_weights = torch.softmax(attention_logits, dim=1)
        base_context = torch.sum(all_node_features * attention_weights, dim=1)

        node_mean = all_node_features.mean(dim=1)
        node_std = all_node_features.std(dim=1, unbiased=False)

        # Convert patch output dictionaries to lists of tensors
        wave_patch_list = list(wave_patched_outputs.values()) if wave_patched_outputs else []
        target_patch_list = list(target_patched_outputs.values()) if target_patched_outputs else []
        
        wave_summary = self._aggregate_patch_collection(
            wave_patch_list,
            "wave_patch",
            batch_size,
            all_node_features.device,
            all_node_features.dtype,
        )
        target_summary = self._aggregate_patch_collection(
            target_patch_list,
            "target_patch",
            batch_size,
            all_node_features.device,
            all_node_features.dtype,
        )

        fusion_input = torch.cat([base_context, node_mean, node_std, wave_summary, target_summary], dim=-1)

        # CRITICAL FIX: Use pre-allocated fusion layer with padding/truncation
        fusion_dim = fusion_input.size(-1)
        max_fusion_dim = self.context_fusion_layer.in_features
        
        if fusion_dim > max_fusion_dim:
            # Truncate if input is too large
            fusion_input = fusion_input[:, :max_fusion_dim]
        elif fusion_dim < max_fusion_dim:
            # Pad if input is too small
            padding = torch.zeros(batch_size, max_fusion_dim - fusion_dim, 
                                device=fusion_input.device, dtype=fusion_input.dtype)
            fusion_input = torch.cat([fusion_input, padding], dim=-1)

        context_vector = self.context_fusion_layer(fusion_input)
        return context_vector

    def _aggregate_patch_collection(
        self,
        patch_outputs: Optional[List[torch.Tensor]],
        key_prefix: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Aggregate multi-scale patch outputs into a fixed-size context vector."""

        if not patch_outputs:
            return torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)

        projections: List[torch.Tensor] = []
        for idx, patch_output in enumerate(patch_outputs):
            if patch_output is None:
                continue
            summary = patch_output
            while summary.dim() > 2:
                summary = summary.mean(dim=1)
            summary = summary.reshape(batch_size, -1)
            projector_key = f"{key_prefix}_{idx}_{summary.size(-1)}"
            projections.append(self._project_context_summary(summary, projector_key, device))

        if not projections:
            return torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)

        stacked = torch.stack(projections, dim=0).mean(dim=0)
        return stacked

    def _project_context_summary(
        self,
        summary: torch.Tensor,
        key: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Project an arbitrary summary tensor to ``d_model`` dimensions."""

        # CRITICAL FIX: Use pre-allocated projectors instead of dynamic creation
        summary_dim = summary.size(-1)
        
        # Find the closest pre-allocated projector
        projector_key = f'proj_{summary_dim}'
        if projector_key in self.context_projection_layers:
            projector = self.context_projection_layers[projector_key]
            return projector(summary)
        
        # Find the next larger projector and pad input
        available_sizes = [int(k.split('_')[1]) for k in self.context_projection_layers.keys() if k.startswith('proj_')]
        larger_sizes = [s for s in available_sizes if s >= summary_dim]
        
        if larger_sizes:
            target_size = min(larger_sizes)
            projector_key = f'proj_{target_size}'
            projector = self.context_projection_layers[projector_key]
            
            # Pad input to match projector size
            if summary_dim < target_size:
                batch_size = summary.size(0)
                padding = torch.zeros(batch_size, target_size - summary_dim, 
                                    device=device, dtype=summary.dtype)
                summary_padded = torch.cat([summary, padding], dim=-1)
                return projector(summary_padded)
        
        # Fallback: use largest available projector and truncate input
        max_size = max(available_sizes)
        projector_key = f'proj_{max_size}'
        projector = self.context_projection_layers[projector_key]
        
        if summary_dim > max_size:
            summary_truncated = summary[:, :max_size]
            return projector(summary_truncated)
        
        return projector(summary)

    def _augment_wave_features(
        self,
        wave_window: torch.Tensor,
        wave_spatial: torch.Tensor,
    ) -> torch.Tensor:
        """Inject phase-aware features into wave node representations."""

        if not getattr(self, 'enable_phase_features', False):
            return wave_spatial
        if not hasattr(self, 'phase_feature_projector'):
            return wave_spatial
        if wave_window.dim() != 3:
            return wave_spatial

        phase_features = self._compute_phase_features(
            wave_window,
            wave_spatial.device,
            wave_spatial.dtype,
        )
        projected = self.phase_feature_projector(phase_features)
        # Align node dimension if mismatch occurs (e.g., enc_in includes targets)
        if projected.size(1) != wave_spatial.size(1):
            projected = projected.transpose(1, 2)
            projected = F.interpolate(projected, size=wave_spatial.size(1), mode='nearest')
            projected = projected.transpose(1, 2)
        return wave_spatial + projected

    def _compute_phase_features(
        self,
        wave_window: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute harmonic embeddings and relative phase statistics for wave nodes."""

        batch_size, seq_len, wave_nodes = wave_window.shape
        float_window = wave_window.float()
        freq_domain = torch.fft.rfft(float_window, dim=1)

        if freq_domain.size(1) > 1:
            dominant = freq_domain[:, 1, :]
        else:
            dominant = freq_domain[:, 0, :]

        amplitude = torch.log1p(torch.abs(dominant))
        phase = torch.angle(dominant)
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)

        if freq_domain.size(1) > 2:
            phase_spectrum = torch.angle(freq_domain[:, 1:, :])
            phase_diff = phase_spectrum[:, 1:, :] - phase_spectrum[:, :-1, :]
            phase_velocity = phase_diff.mean(dim=1)
        else:
            phase_velocity = torch.zeros_like(amplitude)

        phase_matrix = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        relative_sin = torch.sin(phase_matrix).mean(dim=-1)
        relative_cos = torch.cos(phase_matrix).mean(dim=-1)

        features = torch.stack(
            [amplitude, sin_phase, cos_phase, phase_velocity, relative_sin, relative_cos],
            dim=-1,
        )
        return features.to(device=device, dtype=dtype)

    def _compute_delayed_influence_features(
        self,
        wave_window: torch.Tensor,
        wave_spatial: torch.Tensor,
        transition_nodes: int,
    ) -> Optional[torch.Tensor]:
        """Generate lag-aware features for transition nodes."""

        if not getattr(self, 'enable_delayed_influence', False):
            return None
        if not hasattr(self, 'delay_feature_projector') or not hasattr(self, 'delay_wave_to_transition'):
            return None
        if wave_window.dim() != 3:
            return None

        batch_size, seq_len, wave_nodes = wave_window.shape
        if seq_len < 2:
            return None

        lags = min(self.delayed_max_lag, max(1, seq_len - 1))
        lag_features: List[torch.Tensor] = []
        for lag in range(1, lags + 1):
            shifted = torch.roll(wave_window, shifts=lag, dims=1)
            corr = (wave_window * shifted).mean(dim=1)
            lag_features.append(corr)

        lag_tensor = torch.stack(lag_features, dim=-1)
        if lag_tensor.size(-1) < self.delayed_max_lag:
            pad_width = self.delayed_max_lag - lag_tensor.size(-1)
            lag_tensor = F.pad(lag_tensor, (0, pad_width))

        delay_wave = self.delay_feature_projector(lag_tensor)

        weight_matrix = self.delay_wave_to_transition.to(delay_wave.dtype)
        if weight_matrix.size(0) != wave_nodes or weight_matrix.size(1) != transition_nodes:
            weight_matrix = weight_matrix.unsqueeze(0).unsqueeze(0)
            weight_matrix = F.interpolate(
                weight_matrix,
                size=(wave_nodes, transition_nodes),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0).squeeze(0)

        weight_matrix = torch.softmax(weight_matrix, dim=0)
        delayed_transition = torch.einsum('bwd,wt->btd', delay_wave, weight_matrix)
        return delayed_transition.to(device=wave_spatial.device, dtype=wave_spatial.dtype)

    def _build_higher_order_adjacency(
        self,
        wave_spatial: torch.Tensor,
        transition_nodes: int,
        target_nodes: int,
        total_nodes: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Construct a higher-order adjacency capturing group interactions."""

        if not getattr(self, 'enable_group_interactions', False):
            return None
        required_attrs = (
            hasattr(self, 'group_interaction_wave'),
            hasattr(self, 'group_interaction_transition'),
            hasattr(self, 'group_interaction_scale'),
        )
        if not all(required_attrs):
            return None

        batch_size, wave_nodes, _ = wave_spatial.shape
        if self.group_interaction_wave.size(0) != wave_nodes:
            return None
        if self.group_interaction_transition.size(0) != transition_nodes or self.group_interaction_transition.size(1) != target_nodes:
            return None

        synergy = torch.einsum('bid,bjd->bij', wave_spatial, wave_spatial) / math.sqrt(self.d_model)
        wave_weights = torch.softmax(self.group_interaction_wave.to(wave_spatial.dtype), dim=0)
        wave_to_transition = torch.relu(torch.einsum('bij,jk->bik', synergy, wave_weights))

        transition_weights = torch.softmax(self.group_interaction_transition.to(wave_spatial.dtype), dim=0)
        # Aggregate wave_to_transition from [batch, wave_nodes, transition_nodes] to [batch, transition_nodes, target_nodes]
        # First sum over wave dimension to get transition features
        transition_features = wave_to_transition.sum(dim=1)  # [batch, transition_nodes]
        # Then expand and apply transition weights
        transition_context = torch.relu(torch.einsum('bi,ij->bij', transition_features, transition_weights))

        adjacency = wave_spatial.new_zeros((batch_size, total_nodes, total_nodes))
        wave_start = 0
        transition_start = wave_nodes
        target_start = wave_nodes + transition_nodes

        adjacency[:, wave_start:transition_start, transition_start:target_start] = wave_to_transition
        adjacency[:, transition_start:target_start, target_start:] = transition_context

        scale = torch.relu(self.group_interaction_scale).to(device=wave_spatial.device, dtype=wave_spatial.dtype)
        adjacency = adjacency * scale
        weights = adjacency.clone()
        return adjacency, weights

    def _normalize_edge_batch_outputs(
        self,
        adjacency_result: Any,
        batch_size: int,
    ) -> Tuple[List[Dict[Tuple[str, str, str], torch.Tensor]], List[Optional[Dict[Tuple[str, str, str], torch.Tensor]]]]:
        """Coerce adjacency_to_edge_indices output into per-batch lists."""

        if isinstance(adjacency_result, tuple):
            edge_indices_raw, edge_weights_raw = adjacency_result
        else:
            edge_indices_raw = adjacency_result
            edge_weights_raw = None

        if isinstance(edge_indices_raw, dict):
            edge_indices = [edge_indices_raw]
        else:
            edge_indices = list(edge_indices_raw)

        if edge_weights_raw is None:
            edge_weights: List[Optional[Dict[Tuple[str, str, str], torch.Tensor]]] = [None] * len(edge_indices)
        elif isinstance(edge_weights_raw, dict):
            edge_weights = [edge_weights_raw]
        else:
            edge_weights = list(edge_weights_raw)

        if len(edge_indices) != batch_size:
            if len(edge_indices) == 1:
                edge_indices = edge_indices * batch_size
                edge_weights = edge_weights * batch_size
            else:
                raise ValueError(
                    f"Edge index batch size {len(edge_indices)} does not match input batch size {batch_size}."
                )

        return edge_indices, edge_weights

    def get_enhanced_config_info(self):
        """Get a comprehensive summary of the enhanced model's configuration."""
        info = {
            'use_multi_scale_patching': hasattr(self, 'wave_patching_composer') and self.wave_patching_composer is not None,
            'use_hierarchical_mapper': self.use_hierarchical_mapper,
            'use_stochastic_learner': hasattr(self, 'stochastic_learner') and self.stochastic_learner is not None,
            'use_gated_graph_combiner': hasattr(self, 'graph_combiner') and self.graph_combiner is not None,
            'use_mixture_decoder': isinstance(self.decoder, MixtureDensityDecoder),
        }
        
        # Multi-scale patching information
        if hasattr(self, 'wave_patching_composer') and self.wave_patching_composer is not None:
            wave_config = self.wave_patching_composer.get_config_info()
            info['wave_patch_configs'] = wave_config['patch_configs']
            info['wave_patch_scales'] = wave_config['num_scales']
            info['wave_patch_latents'] = wave_config['num_latents']
        
        if hasattr(self, 'target_patching_composer') and self.target_patching_composer is not None:
            target_config = self.target_patching_composer.get_config_info()
            info['target_patch_configs'] = target_config['patch_configs']
            info['target_patch_scales'] = target_config['num_scales']
            info['target_patch_latents'] = target_config['num_latents']
        
        # Hierarchical mapper information
        if self.use_hierarchical_mapper:
            if hasattr(self, 'wave_temporal_to_spatial'):
                info['wave_mapper_nodes'] = self.wave_temporal_to_spatial.num_nodes
            if hasattr(self, 'target_temporal_to_spatial'):
                info['target_mapper_nodes'] = self.target_temporal_to_spatial.num_nodes
        
        # Graph combiner information
        if hasattr(self, 'graph_combiner') and self.graph_combiner is not None:
            info['num_graphs_combined'] = self.graph_combiner.num_graphs
            info['graph_combiner_d_model'] = getattr(self.graph_combiner, 'd_model', self.d_model)
        
        # Stochastic learner information
        if hasattr(self, 'stochastic_learner') and self.stochastic_learner is not None:
            info['stochastic_learner_active'] = True
            if hasattr(self, 'latest_stochastic_loss'):
                info['latest_stochastic_loss'] = float(self.latest_stochastic_loss.item()) if self.latest_stochastic_loss is not None else 0.0
        
        # Mixture decoder information
        if isinstance(self.decoder, MixtureDensityDecoder):
            info['mixture_decoder_components'] = getattr(self.decoder, 'num_components', 3)
            info['mixture_decoder_targets'] = getattr(self.decoder, 'num_targets', 1)
            info['mixture_multivariate_mode'] = getattr(self.mixture_loss, 'multivariate_mode', 'unknown') if hasattr(self, 'mixture_loss') else 'none'
        
        # Internal logging information
        if hasattr(self, 'internal_logs'):
            info['internal_logs'] = self.internal_logs
            
        return info

    def _validate_graph_output(self, graph_result, component_name: str):
        """Validate graph component output format and provide robust fallbacks."""
        if isinstance(graph_result, (tuple, list)):
            if len(graph_result) >= 2:
                return graph_result[0], graph_result[1]
            elif len(graph_result) == 1:
                print(f"Warning: {component_name} returned single element tuple, using None for weights")
                return graph_result[0], None
            else:
                raise ValueError(f"Invalid tuple length from {component_name}: {len(graph_result)}")
        elif isinstance(graph_result, torch.Tensor):
            return graph_result, None
        else:
            raise ValueError(f"Invalid output format from {component_name}: {type(graph_result)}")

    def _align_sequence_lengths(self, wave_embedded, target_embedded):
        """Ensure consistent sequence lengths for hierarchical mapping."""
        wave_seq_len = wave_embedded.shape[1]
        target_seq_len = target_embedded.shape[1]
        
        if wave_seq_len != target_seq_len:
            # Align to shorter sequence to avoid dimension mismatches
            min_len = min(wave_seq_len, target_seq_len)
            wave_embedded = wave_embedded[:, -min_len:, :]
            target_embedded = target_embedded[:, -min_len:, :]
            print(f"Warning: Aligned sequence lengths from {wave_seq_len}, {target_seq_len} to {min_len}")
        
        return wave_embedded, target_embedded

    def get_internal_logs(self) -> Dict[str, torch.Tensor]:
        """Returns a dictionary of internal model states for logging and visualization."""
        return getattr(self, 'internal_logs', {})