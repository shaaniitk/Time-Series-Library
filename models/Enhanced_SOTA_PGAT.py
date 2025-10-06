import torch
import torch.nn as nn
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
        self.internal_logs = {}
        
        self._validate_enhanced_config(config)

        total_nodes = getattr(config, 'enc_in', 7) + getattr(config, 'c_out', 3) + min(getattr(config, 'enc_in', 7), getattr(config, 'c_out', 3))

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
                num_nodes=getattr(config, 'enc_in', 7),
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
            
            # Create two separate composers for wave and target
            self.wave_patching_composer = MultiScalePatchingComposer(
                patch_configs=wave_patch_configs,
                d_model=self.d_model,
                input_features=getattr(config, 'enc_in'),
                num_latents=getattr(config, 'num_wave_patch_latents', 64),
                n_heads=getattr(config, 'n_heads', 8)
            )
            self.target_patching_composer = MultiScalePatchingComposer(
                patch_configs=target_patch_configs,
                d_model=self.d_model,
                input_features=getattr(config, 'c_out'),
                num_latents=getattr(config, 'num_target_patch_latents', 24),
                n_heads=getattr(config, 'n_heads', 8)
            )
            # Use Identity for patched mode (patching composers handle embedding)
            self.embedding = nn.Identity()
        else:
            self.wave_patching_composer = None
            self.target_patching_composer = None
            # Use proper embedding for non-patched mode
            try:
                self.embedding = self._initialize_embedding(config)
            except:
                # Fallback to simple linear embedding
                self.embedding = nn.Linear(getattr(config, 'enc_in', 7), self.d_model)

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
            self.mixture_loss = MixtureNLLLoss(multivariate_mode=multivariate_mode)
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
            # Patching composers handle their own temporal encoding
            wave_embedded, wave_patched_outputs = self.wave_patching_composer(wave_window)
            target_embedded, target_patched_outputs = self.target_patching_composer(target_window)
            # Skip redundant temporal encoding for patched data
        else:
            # Apply embedding and temporal encoding for non-patched data
            wave_embedded = self.embedding(wave_window.reshape(-1, wave_window.shape[-1])).view(batch_size, wave_window.shape[1], -1)
            target_embedded = self.embedding(target_window.reshape(-1, target_window.shape[-1])).view(batch_size, target_window.shape[1], -1)
            # Apply temporal encoding only for non-patched data
            wave_embedded = self.temporal_pos_encoding(wave_embedded)
            target_embedded = self.temporal_pos_encoding(target_embedded)

        wave_nodes = getattr(self.config, 'enc_in', 7)
        target_nodes = getattr(self.config, 'c_out', 3)
        transition_nodes = max(1, min(wave_nodes, target_nodes))

        # --- 2. Temporal-to-Spatial Conversion ---
        if self.use_hierarchical_mapper:
            wave_spatial = self.wave_temporal_to_spatial(wave_embedded)
            target_spatial = self.target_temporal_to_spatial(target_embedded)
        else: 
            wave_spatial = wave_embedded.mean(dim=1).unsqueeze(1).expand(-1, wave_nodes, -1)
            target_spatial = target_embedded.mean(dim=1).unsqueeze(1).expand(-1, target_nodes, -1)

        # --- 3. Dynamic Graph Construction & Gated Combination ---
        if not hasattr(self, 'transition_features'):
            transition_dim = self.d_model  # Use self.d_model consistently
            transition_init = torch.randn(transition_nodes, transition_dim, device=wave_embedded.device, dtype=wave_embedded.dtype)
            self.register_parameter('transition_features', nn.Parameter(transition_init))
        
        transition_broadcast = self.transition_features.unsqueeze(0).expand(batch_size, -1, -1)
        
        
        node_features_dict = {'wave': wave_spatial, 'transition': transition_broadcast, 'target': target_spatial}
        all_node_features = torch.cat([wave_spatial, transition_broadcast, target_spatial], dim=1)

        # Get total nodes for consistent tensor shapes
        total_nodes = wave_nodes + transition_nodes + target_nodes
        
        # Get graph proposals using improved utilities
        
        # Base graph (dynamic)
        dyn_result = self.dynamic_graph(node_features_dict)
        if isinstance(dyn_result, (tuple, list)):
            dyn_hetero, dyn_weights = dyn_result[0], dyn_result[1]
        else:
            dyn_hetero, dyn_weights = dyn_result, None
        
        dyn_proposal = prepare_graph_proposal(dyn_hetero, dyn_weights, batch_size, total_nodes, preserve_weights=True)
        
        # Adaptive graph  
        adapt_result = self.adaptive_graph(node_features_dict)
        if isinstance(adapt_result, (tuple, list)):
            adapt_hetero, adapt_weights = adapt_result[0], adapt_result[1]
        else:
            adapt_hetero, adapt_weights = adapt_result, None
            
        adapt_proposal = prepare_graph_proposal(adapt_hetero, adapt_weights, batch_size, total_nodes, preserve_weights=True)

        # Prepare graph proposals list for gated combiner
        graph_proposals = [dyn_proposal, adapt_proposal]
        
        # Handle stochastic learner - integration strategy: append to proposals
        stochastic_adj = None
        if self.stochastic_learner:
            try:
                stoch_adj, stoch_logits = self.stochastic_learner(all_node_features, self.training)
                self.latest_stochastic_loss = self.stochastic_learner.regularization_loss(stoch_logits)
                
                # Prepare stochastic proposal and append to list
                stoch_proposal = prepare_graph_proposal(stoch_adj, None, batch_size, total_nodes, preserve_weights=True)
                graph_proposals.append(stoch_proposal)
                stochastic_adj = stoch_proposal[0]
                
            except Exception as e:
                self.latest_stochastic_loss = torch.tensor(0.0, device=wave_embedded.device)
                self.internal_logs['stochastic_error'] = str(e)
        
        # Validate all proposals before gated combination
        proposals_valid = validate_graph_proposals(graph_proposals, batch_size, total_nodes)

        # Combine graphs using the restored gated combiner contract
        if hasattr(self, 'graph_combiner') and self.graph_combiner is not None and proposals_valid:
            try:
                # Create rich context tensor preserving multi-scale and stochastic information
                # Instead of simple mean, use attention-weighted pooling to preserve important features
                context = self._create_rich_context(all_node_features, wave_patched_outputs, target_patched_outputs)  # [batch_size, d_model]
                
                # The improved gated combiner can handle variable number of graphs
                adjacency_matrix, edge_weights = self.graph_combiner(graph_proposals, context)
                self.internal_logs = {
                    'graph_combination': 'success',
                    'num_proposals': len(graph_proposals),
                    'includes_stochastic': stochastic_adj is not None,
                    'proposals_valid': proposals_valid
                }
            except Exception as e:
                # Fallback to adaptive graph if combiner fails
                adjacency_matrix = adapt_proposal[0]
                edge_weights = adapt_proposal[1]
                self.internal_logs = {'graph_combination': f'fallback: {str(e)}'}
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
        enhanced_edge_index_dict = adjacency_to_edge_indices(
            adjacency_matrix, wave_nodes, target_nodes, transition_nodes, edge_weights
        )
        
        # CRITICAL: Use per-sample node features instead of batch-averaged
        # This preserves the multi-scale and stochastic learning signals
        enhanced_x_dict = {
            'wave': wave_spatial,      # Keep batch dimension [batch, wave_nodes, d_model]
            'transition': transition_broadcast,  # [batch, transition_nodes, d_model]
            'target': target_spatial   # Keep batch dimension [batch, target_nodes, d_model]
        }

        # Handle graph attention with batched features
        if getattr(self.config, 'enable_graph_attention', True):
            # Graph attention expects non-batched features, so we process each batch sample
            batch_attended_features = {'wave': [], 'transition': [], 'target': []}
            
            for b in range(batch_size):
                # Extract single batch features
                single_batch_x_dict = {
                    'wave': enhanced_x_dict['wave'][b],      # [wave_nodes, d_model]
                    'transition': enhanced_x_dict['transition'][b],  # [transition_nodes, d_model]
                    'target': enhanced_x_dict['target'][b]   # [target_nodes, d_model]
                }
                
                # Apply graph attention to single batch
                attended_single = self.graph_attention(single_batch_x_dict, enhanced_edge_index_dict)
                
                # Collect results
                for key in batch_attended_features:
                    batch_attended_features[key].append(attended_single[key])
            
            # Stack back to batch dimension
            spatial_encoded = {
                'wave': torch.stack(batch_attended_features['wave'], dim=0),      # [batch, wave_nodes, d_model]
                'transition': torch.stack(batch_attended_features['transition'], dim=0),  # [batch, transition_nodes, d_model]
                'target': torch.stack(batch_attended_features['target'], dim=0)   # [batch, target_nodes, d_model]
            }
        else:
            # No graph attention - use features as-is
            spatial_encoded = enhanced_x_dict

        target_spatial_encoded = spatial_encoded['target']
        temporal_encoded, _ = self.temporal_encoder(target_spatial_encoded, target_spatial_encoded, target_spatial_encoded)
        
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
            if verbose:
                print("Enhanced PGAT using MixtureNLLLoss for MDN outputs")
            return self.mixture_loss
        else:
            if verbose:
                print("Enhanced PGAT using standard loss function")
            return base_criterion

    def _ensure_config_attributes(self, config):
        """Ensure config has all required attributes for parent class."""
        required_attrs = {
            'seq_len': 24,
            'pred_len': 6, 
            'enc_in': 7,
            'c_out': 3,
            'd_model': 512,
            'n_heads': 8,
            'dropout': 0.1
        }
        
        for attr, default_value in required_attrs.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                setattr(config, attr, default_value)

    def _validate_enhanced_config(self, config):
        """Validate enhanced model configuration parameters."""
        if getattr(config, 'use_multi_scale_patching', True):
            patch_configs = getattr(config, 'patch_configs', [])
            if not isinstance(patch_configs, list) or not all(isinstance(c, dict) for c in patch_configs):
                raise ValueError("patch_configs must be a list of dictionaries.")

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
        
        # Start with attention-weighted pooling instead of simple mean
        # Use learnable attention weights for different node types
        if not hasattr(self, 'context_attention'):
            self.context_attention = nn.Linear(d_model, 1).to(all_node_features.device)
        
        # Compute attention weights for each node
        attention_logits = self.context_attention(all_node_features)  # [batch_size, total_nodes, 1]
        attention_weights = torch.softmax(attention_logits, dim=1)  # [batch_size, total_nodes, 1]
        
        # Attention-weighted pooling
        base_context = torch.sum(all_node_features * attention_weights, dim=1)  # [batch_size, d_model]
        
        # Enhance with multi-scale information if available
        if wave_patched_outputs and len(wave_patched_outputs) > 0:
            # Aggregate multi-scale wave information
            wave_scales = []
            for wave_output in wave_patched_outputs:
                if wave_output is not None and wave_output.numel() > 0:
                    # Pool each scale to [batch_size, d_model]
                    if wave_output.dim() == 3:  # [batch_size, seq_len, d_model]
                        scale_pooled = wave_output.mean(dim=1)
                    else:
                        scale_pooled = wave_output.view(batch_size, -1)[:, :d_model]
                    wave_scales.append(scale_pooled)
            
            if wave_scales:
                # Combine multi-scale information
                multi_scale_wave = torch.stack(wave_scales, dim=1).mean(dim=1)  # [batch_size, d_model]
                base_context = base_context + 0.1 * multi_scale_wave  # Small contribution
        
        # Add target information if available
        if target_patched_outputs and len(target_patched_outputs) > 0:
            target_scales = []
            for target_output in target_patched_outputs:
                if target_output is not None and target_output.numel() > 0:
                    if target_output.dim() == 3:
                        scale_pooled = target_output.mean(dim=1)
                    else:
                        scale_pooled = target_output.view(batch_size, -1)[:, :d_model]
                    target_scales.append(scale_pooled)
            
            if target_scales:
                multi_scale_target = torch.stack(target_scales, dim=1).mean(dim=1)
                base_context = base_context + 0.1 * multi_scale_target
        
        return base_context

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

    def get_internal_logs(self) -> Dict[str, torch.Tensor]:
        """Returns a dictionary of internal model states for logging and visualization."""
        return getattr(self, 'internal_logs', {})