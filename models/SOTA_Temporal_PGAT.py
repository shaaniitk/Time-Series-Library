import torch
import torch.nn as nn
from layers.modular.attention.registry import AttentionRegistry
from layers.modular.decoder.registry import DecoderRegistry
from layers.modular.graph.registry import GraphComponentRegistry
from layers.modular.embedding.registry import EmbeddingRegistry

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
        
        # Spatial encoder using graph components
        self.spatial_encoder = self.graph_registry.get('pgat_cross_attn_layer')(
            d_model=config.d_model
        )
        
        # Temporal encoder using attention components
        self.temporal_encoder = self.attention_registry.get('temporal_attention')(
            d_model=config.d_model,
            n_heads=getattr(config, 'n_heads', 8),
            dropout=getattr(config, 'dropout', 0.1)
        )
        
        # Decoder selection based on mode
        if self.mode == 'standard':
            self.decoder = self.decoder_registry.get('custom_standard')(config.d_model)
        else:  # probabilistic mode
            self.decoder = self.decoder_registry.get('probabilistic')(config.d_model)
    
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
        
        # Spatial encoding with graph attention
        # Split embedded tensor back into wave and target parts
        wave_len = wave_window.shape[1]
        target_len = target_window.shape[1]
        wave_embedded = embedded[:, :wave_len, :]
        target_embedded = embedded[:, wave_len:wave_len+target_len, :]
        
        # For now, skip spatial encoding and use embedded features directly
        # TODO: Implement proper graph structure when available
        spatial_encoded = {
            'wave': wave_embedded,
            'transition': wave_embedded,
            'target': target_embedded
        }
        
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
        
        # Decode to final output
        return self.decoder(final_embedding)
    
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