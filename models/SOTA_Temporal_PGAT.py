import torch
import torch.nn as nn
from layers.modular.attention.registry import AttentionRegistry
from layers.modular.decoder.registry import DecoderRegistry
from layers.modular.graph.registry import GraphComponentRegistry
from layers.modular.embedding.registry import EmbeddingRegistry
from layers.modular.graph.dynamic_graph import DynamicGraphConstructor, AdaptiveGraphStructure
from layers.modular.attention.multihead_graph_attention import MultiHeadGraphAttention, GraphTransformerLayer
from layers.modular.encoder.spatiotemporal_encoding import JointSpatioTemporalEncoding, AdaptiveSpatioTemporalEncoder
from layers.modular.embedding.graph_positional_encoding import GraphAwarePositionalEncoding, HierarchicalGraphPositionalEncoding
# New enhanced components
from layers.modular.decoder.mixture_density_decoder import MixtureDensityDecoder
from layers.modular.attention.autocorr_temporal_attention import AutoCorrTemporalAttention
from layers.modular.embedding.structural_positional_encoding import StructuralPositionalEncoding
from layers.modular.embedding.enhanced_temporal_encoding import EnhancedTemporalEncoding
from layers.modular.graph.enhanced_pgat_layer import EnhancedPGAT_CrossAttn_Layer

class SOTA_Temporal_PGAT(nn.Module):
    """
    State-of-the-Art Temporal Probabilistic Graph Attention Transformer
    Refactored to use modular components from registries
    """
    
    def __init__(self, config, mode='probabilistic'):
        super().__init__()
        self.config = config
        self.mode = mode
        
        # Initialize registries
        self.attention_registry = AttentionRegistry()
        self.decoder_registry = DecoderRegistry()
        self.graph_registry = GraphComponentRegistry()
        
        # Get components from registries
        try:
            # Try to get embedding from registry, fallback to direct import if not available
            try:
                embedding_registry = EmbeddingRegistry()
                self.embedding = embedding_registry.get('initial_embedding')(config)
            except:
                from layers.modular.embedding.initial_embedding import InitialEmbedding
                self.embedding = InitialEmbedding(config)
        except:
            # Simple fallback embedding implementation
            # For multivariate ('M'), assume 7 features as default
            features = getattr(config, 'features', 'M')
            d_model = getattr(config, 'd_model', 512)
            num_features = 7 if features == 'M' else 1
            self.embedding = nn.Linear(num_features, d_model)
        
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
        use_dynamic_weights = getattr(config, 'use_dynamic_edge_weights', True)
        if use_dynamic_weights:
            self.spatial_encoder = EnhancedPGAT_CrossAttn_Layer(
                d_model=config.d_model,
                num_heads=getattr(config, 'n_heads', 8),
                use_dynamic_weights=True
            )
        else:
            self.spatial_encoder = self.graph_registry.get('pgat_cross_attn_layer')(
                d_model=config.d_model
            )
        
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
        use_mdn = getattr(config, 'use_mixture_density', True)
        if self.mode == 'standard':
            self.decoder = self.decoder_registry.get('custom_standard')(config.d_model)
        else:  # probabilistic mode
            if use_mdn:
                self.decoder = MixtureDensityDecoder(
                    input_dim=config.d_model,
                    output_dim=getattr(config, 'c_out', 1),
                    num_components=getattr(config, 'mdn_components', 3),
                    hidden_dim=getattr(config, 'mdn_hidden_dim', 256)
                )
            else:
                self.decoder = self.decoder_registry.get('probabilistic')(config.d_model)
        
        # Add structural positional encoding
        self.structural_pos_encoding = StructuralPositionalEncoding(
            d_model=config.d_model,
            max_nodes=getattr(config, 'max_nodes', 100),
            k_eigenvectors=getattr(config, 'k_eigenvectors', 16)
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
        if not hasattr(self, 'transition_features'):
            self.transition_features = nn.Parameter(
                torch.randn(self.config.num_transitions, getattr(self.config, 'd_model', 512))
            )
        
        x_dict['transition'] = self.transition_features.expand(batch_size, -1, -1).mean(dim=0)
        
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
        num_nodes = combined_input.shape[-1]
        
        # Initialize dynamic graph components if not exists
        if self.dynamic_graph is None:
            self.dynamic_graph = DynamicGraphConstructor(
                num_nodes=num_nodes,
                feature_dim=seq_len,
                hidden_dim=self.d_model
            ).to(combined_input.device)
        
        if self.adaptive_graph is None:
            self.adaptive_graph = AdaptiveGraphStructure(
                num_nodes=num_nodes,
                feature_dim=seq_len
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
        node_features = combined_input.transpose(1, 2)  # [batch_size, features, seq_len]
        adjacency_matrix, edge_weights = self.dynamic_graph(node_features)
        
        # Update graph structure adaptively
        adjacency_matrix = self.adaptive_graph(adjacency_matrix, node_features)
        
        # Reshape embedded features for spatial-temporal processing
        spatiotemporal_input = embedded.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [batch, seq, nodes, d_model]
        
        # Add structural positional encoding for enhanced graph structure awareness
        if hasattr(self, 'structural_pos_encoding'):
            structural_encoding = self.structural_pos_encoding(adjacency_matrix, spatiotemporal_input)
            spatiotemporal_input = spatiotemporal_input + structural_encoding
        
        # Add graph-aware positional encoding
        pos_encoding = self.graph_pos_encoding(
            batch_size, seq_len, num_nodes, adjacency_matrix, combined_input.device
        )
        spatiotemporal_input = spatiotemporal_input + pos_encoding
        
        # Apply joint spatial-temporal encoding
        spatiotemporal_encoded = self.spatiotemporal_encoder(
            spatiotemporal_input, adjacency_matrix
        )
        
        # Apply enhanced graph attention
        enhanced_x_dict = {
            'wave': spatiotemporal_encoded[:, :wave_len, :, :].mean(dim=2),
            'transition': spatiotemporal_encoded[:, wave_len:, :, :].mean(dim=2),
            'target': spatiotemporal_encoded[:, wave_len:, :, :].mean(dim=2)
        }
        
        enhanced_edge_index_dict = {
            ('wave', 'interacts_with', 'transition'): self._create_edge_index(wave_len, target_len, combined_input.device),
            ('transition', 'influences', 'target'): self._create_edge_index(target_len, target_len, combined_input.device)
        }
        
        # Apply multi-head graph attention
        attended_features = self.graph_attention(enhanced_x_dict, enhanced_edge_index_dict)
        
        # Apply original spatial encoding through graph attention
        spatial_encoded_dict = self.spatial_encoder(x_dict, t_dict, edge_index_dict)
        
        # Combine enhanced and original features
        spatial_encoded = {
            'wave': spatial_encoded_dict['wave'].unsqueeze(0).expand(batch_size, -1, -1) + attended_features['wave'],
            'transition': spatial_encoded_dict['transition'].unsqueeze(0).expand(batch_size, -1, -1) + attended_features['transition'],
            'target': spatial_encoded_dict['target'].unsqueeze(0).expand(batch_size, -1, -1) + attended_features['target']
        }
        
        # Store graph information if enabled
        if self.store_graph_info:
            self.last_adjacency_matrix = adjacency_matrix.detach()
            self.last_edge_weights = edge_weights.detach() if edge_weights is not None else None
        
        # Temporal encoding with temporal attention
        # Extract target embeddings from spatial encoded output
        target_spatial = spatial_encoded['target']
        temporal_encoded, _ = self.temporal_encoder(
            target_spatial,
            spatial_encoded['wave']
        )
        
        # Combine spatial and temporal features
        # Use target spatial features for combination
        final_embedding = temporal_encoded + target_spatial
        
        # Project features if dimension mismatch
        if final_embedding.size(-1) != self.d_model:
            if self.feature_projection is None:
                self.feature_projection = nn.Linear(final_embedding.size(-1), self.d_model).to(final_embedding.device)
            final_embedding = self.feature_projection(final_embedding)
        
        # Decode to final output
        return self.decoder(final_embedding)
    
    def _create_edge_index(self, num_source: int, num_target: int, device: torch.device) -> torch.Tensor:
        """
        Create edge indices for heterogeneous graph attention
        """
        # Create fully connected bipartite graph
        source_nodes = torch.arange(num_source, device=device).repeat(num_target)
        target_nodes = torch.arange(num_target, device=device).repeat_interleave(num_source)
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        return edge_index
    
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