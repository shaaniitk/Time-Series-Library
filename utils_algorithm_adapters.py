"""
Phase 2: Algorithm Adaptation - Utils System Integration

This module creates BaseAttention wrappers for our restored sophisticated algorithms,
allowing them to work within the utils/ modular system while preserving their
algorithmic sophistication.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Import utils base interfaces
from utils.modular_components.base_interfaces import BaseAttention
from utils.modular_components.config_schemas import AttentionConfig

# Import our restored algorithms
from layers.modular.attention.fourier_attention import FourierAttention as OriginalFourierAttention
from layers.EfficientAutoCorrelation import EfficientAutoCorrelation
from layers.modular.attention.adaptive_components import MetaLearningAdapter as OriginalMetaLearningAdapter


@dataclass
class RestoredFourierConfig(AttentionConfig):
    """Configuration for restored Fourier attention with all sophisticated features"""
    seq_len: int = 96
    frequency_selection: str = 'adaptive'
    temperature: float = 1.0
    learnable_filter: bool = True


@dataclass
class RestoredAutoCorrelationConfig(AttentionConfig):
    """Configuration for restored AutoCorrelation with learned k-predictor"""
    factor: float = 1.0
    adaptive_k: bool = True
    multi_scale: bool = True
    scales: list = None
    use_checkpoint: bool = True
    max_seq_len: int = 1024


@dataclass
class RestoredMetaLearningConfig(AttentionConfig):
    """Configuration for restored MAML meta-learning adapter"""
    adaptation_steps: int = 3
    meta_lr: float = 0.01
    inner_lr: float = 0.1


class RestoredFourierAttention(BaseAttention):
    """
    Utils-compatible wrapper for our restored Fourier attention with 
    sophisticated complex frequency filtering.
    
    Preserves the sophisticated algorithmic implementation while providing
    utils system compatibility.
    """
    
    def __init__(self, config: RestoredFourierConfig):
        super().__init__(config)
        self.config = config
        
        # Create our sophisticated restored algorithm
        self.fourier_attention = OriginalFourierAttention(
            d_model=config.d_model,
            n_heads=getattr(config, 'num_heads', config.d_model // 64),
            seq_len=config.seq_len,
            frequency_selection=config.frequency_selection,
            dropout=config.dropout,
            temperature=config.temperature,
            learnable_filter=config.learnable_filter
        )
        
        self.d_model = config.d_model
        self.num_heads = getattr(config, 'num_heads', config.d_model // 64)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass preserving sophisticated frequency filtering algorithm
        
        Args:
            query: Query tensor [batch, seq_len, d_model]
            key: Key tensor [batch, seq_len, d_model]  
            value: Value tensor [batch, seq_len, d_model]
            mask: Optional attention mask (alias of attn_mask)
            attn_mask: Optional attention mask (alias of mask)
            
        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        # Normalize alias arguments
        if attn_mask is not None and mask is None:
            mask = attn_mask
        
        # Our restored algorithm uses sophisticated complex frequency filtering
        output, attn_weights = self.fourier_attention(query, key, value, mask)
        return output, attn_weights
    
    def apply_attention(self, 
                       queries: torch.Tensor, 
                       keys: torch.Tensor, 
                       values: torch.Tensor, 
                       attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """BaseAttention interface method"""
        output, attn_weights = self.forward(queries, keys, values, attn_mask)
        return output, attn_weights
    
    def get_output_dim(self) -> int:
        """BaseAttention interface method"""
        return self.d_model
    
    def get_attention_type(self) -> str:
        """BaseAttention interface method"""
        return "restored_fourier"
    
    @staticmethod
    def get_capabilities() -> list:
        """Return capabilities of this component"""
        return [
            'sophisticated_frequency_filtering',
            'complex_fourier_transform',
            'adaptive_frequency_selection',
            'learnable_filters',
            'restored_algorithm'
        ]
    
    @staticmethod
    def get_requirements() -> dict:
        """Return requirements for this component"""
        return {
            'min_seq_len': 4,
            'supports_variable_length': True,
            'requires_complex_math': True,
            'algorithmic_sophistication': 'high'
        }


class RestoredAutoCorrelationAttention(BaseAttention):
    """
    Utils-compatible wrapper for our restored Enhanced AutoCorrelation with
    sophisticated learned k-predictor and multi-scale analysis.
    
    Preserves the sophisticated algorithmic implementation including the
    learned neural network for adaptive k-selection.
    """
    
    def __init__(self, config: RestoredAutoCorrelationConfig):
        super().__init__(config)
        self.config = config
        
        # Set default scales if not provided
        if config.scales is None:
            config.scales = [1, 2, 4]
        
        # Create our sophisticated restored algorithm with learned k-predictor
        self.autocorr_attention = EfficientAutoCorrelation(
            mask_flag=True,
            factor=config.factor,
            attention_dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            use_checkpoint=config.use_checkpoint,
            adaptive_k=config.adaptive_k,
            multi_scale=config.multi_scale,
            scales=config.scales
        )
        
        self.d_model = config.d_model
        self.num_heads = getattr(config, 'num_heads', config.d_model // 64)
        
        # Verify our sophisticated k_predictor exists
        assert hasattr(self.autocorr_attention, 'k_predictor'), "Sophisticated k_predictor missing!"
        assert len(list(self.autocorr_attention.k_predictor.modules())) > 1, "k_predictor too simple!"
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass preserving sophisticated learned k-predictor and multi-scale analysis
        
        Args:
            query: Query tensor [batch, seq_len, d_model]
            key: Key tensor [batch, seq_len, d_model]
            value: Value tensor [batch, seq_len, d_model] 
            mask: Optional attention mask (alias of attn_mask)
            attn_mask: Optional attention mask (alias of mask)
            
        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        if attn_mask is not None and mask is None:
            mask = attn_mask
        B, L, D = query.shape
        H = self.num_heads
        E = D // H
        
        # Reshape for our algorithm's expected format [batch, seq_len, heads, head_dim]
        query_reshaped = query.view(B, L, H, E)
        key_reshaped = key.view(B, L, H, E)
        value_reshaped = value.view(B, L, H, E)
        
        # Our restored algorithm uses sophisticated learned k-selection
        output, attn_weights = self.autocorr_attention(query_reshaped, key_reshaped, value_reshaped, mask)
        
        # Reshape back to expected format
        output = output.view(B, L, D)
        
        return output, attn_weights
    
    def apply_attention(self, 
                       queries: torch.Tensor, 
                       keys: torch.Tensor, 
                       values: torch.Tensor, 
                       attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """BaseAttention interface method"""
        output, attn_weights = self.forward(queries, keys, values, attn_mask)
        return output, attn_weights
    
    def get_output_dim(self) -> int:
        """BaseAttention interface method"""
        return self.d_model
    
    def get_attention_type(self) -> str:
        """BaseAttention interface method"""
        return "restored_autocorrelation"
    
    @staticmethod
    def get_capabilities() -> list:
        """Return capabilities of this component"""
        return [
            'sophisticated_k_predictor',
            'learned_adaptive_selection',
            'multi_scale_analysis',
            'fft_correlation',
            'neural_network_k_prediction',
            'restored_algorithm'
        ]
    
    @staticmethod
    def get_requirements() -> dict:
        """Return requirements for this component"""
        return {
            'min_seq_len': 2,
            'supports_variable_length': True,
            'requires_fft': True,
            'requires_neural_networks': True,
            'algorithmic_sophistication': 'very_high'
        }


class RestoredMetaLearningAttention(BaseAttention):
    """
    Utils-compatible wrapper for our restored MAML meta-learning adapter with
    sophisticated gradient-based fast adaptation.
    
    Preserves the sophisticated MAML implementation with fast_weights system.
    """
    
    def __init__(self, config: RestoredMetaLearningConfig):
        super().__init__(config)
        self.config = config
        
        # Create our sophisticated restored MAML algorithm
        self.meta_attention = OriginalMetaLearningAdapter(
            d_model=config.d_model,
            n_heads=getattr(config, 'num_heads', config.d_model // 64),
            adaptation_steps=config.adaptation_steps,
            meta_lr=config.meta_lr,
            inner_lr=config.inner_lr,
            dropout=config.dropout
        )
        
        self.d_model = config.d_model
        self.num_heads = getattr(config, 'num_heads', config.d_model // 64)
        
        # Verify our sophisticated MAML components exist
        assert hasattr(self.meta_attention, 'fast_weights'), "MAML fast_weights missing!"
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass preserving sophisticated MAML gradient-based adaptation
        
        Args:
            query: Query tensor [batch, seq_len, d_model]
            key: Key tensor [batch, seq_len, d_model]
            value: Value tensor [batch, seq_len, d_model]
            mask: Optional attention mask (alias of attn_mask)
            attn_mask: Optional attention mask (alias of mask)
            
        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        if attn_mask is not None and mask is None:
            mask = attn_mask
        # Our restored algorithm uses sophisticated MAML implementation
        output, attn_weights = self.meta_attention(query, key, value, mask)
        return output, attn_weights
    
    def apply_attention(self, 
                       queries: torch.Tensor, 
                       keys: torch.Tensor, 
                       values: torch.Tensor, 
                       attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """BaseAttention interface method"""
        output, attn_weights = self.forward(queries, keys, values, attn_mask)
        return output, attn_weights
    
    def get_output_dim(self) -> int:
        """BaseAttention interface method"""
        return self.d_model
    
    def get_attention_type(self) -> str:
        """BaseAttention interface method"""
        return "restored_meta_learning"
    
    @staticmethod
    def get_capabilities() -> list:
        """Return capabilities of this component"""
        return [
            'maml_meta_learning',
            'gradient_based_adaptation',
            'fast_weights_system',
            'few_shot_learning',
            'adaptive_parameters',
            'restored_algorithm'
        ]
    
    @staticmethod  
    def get_requirements() -> dict:
        """Return requirements for this component"""
        return {
            'min_seq_len': 1,
            'supports_variable_length': True,
            'requires_gradients': True,
            'requires_meta_learning': True,
            'algorithmic_sophistication': 'very_high'
        }


# Registration helper functions
def register_restored_algorithms():
    """Register all our restored algorithms in the utils system"""
    from utils.modular_components.registry import register_component
    
    # Register our sophisticated restored algorithms
    register_component(
        'attention', 
        'restored_fourier_attention',
        RestoredFourierAttention,
        metadata={
            'description': 'Restored Fourier attention with sophisticated complex frequency filtering',
            'sophistication_level': 'high',
            'features': ['complex_filtering', 'adaptive_selection', 'learnable_filters'],
            'algorithm_source': 'layers.modular.attention.fourier_attention'
        }
    )
    
    register_component(
        'attention',
        'restored_autocorrelation_attention', 
        RestoredAutoCorrelationAttention,
        metadata={
            'description': 'Restored AutoCorrelation with sophisticated learned k-predictor',
            'sophistication_level': 'very_high', 
            'features': ['learned_k_predictor', 'multi_scale', 'neural_network'],
            'algorithm_source': 'layers.EfficientAutoCorrelation'
        }
    )
    
    register_component(
        'attention',
        'restored_meta_learning_attention',
        RestoredMetaLearningAttention,
        metadata={
            'description': 'Restored MAML meta-learning adapter with gradient-based adaptation',
            'sophistication_level': 'very_high',
            'features': ['maml', 'fast_weights', 'gradient_adaptation', 'meta_learning'],
            'algorithm_source': 'layers.modular.attention.adaptive_components'
        }
    )
    
    print("✅ Registered 3 restored sophisticated algorithms in utils system")
    print("   • restored_fourier_attention")
    print("   • restored_autocorrelation_attention") 
    print("   • restored_meta_learning_attention")


if __name__ == "__main__":
    # Test the wrappers
    print("🔧 PHASE 2: ALGORITHM ADAPTATION - TESTING WRAPPERS")
    print("=" * 60)
    
    # Test 1: Fourier Attention Wrapper
    try:
        config = RestoredFourierConfig(d_model=64, num_heads=4, dropout=0.1)
        fourier = RestoredFourierAttention(config)
        
        x = torch.randn(2, 32, 64)
        output, _ = fourier.apply_attention(x, x, x)
        assert output.shape == x.shape
        print("✅ RestoredFourierAttention wrapper working")
        
    except Exception as e:
        print(f"❌ RestoredFourierAttention wrapper failed: {e}")
    
    # Test 2: AutoCorrelation Wrapper
    try:
        config = RestoredAutoCorrelationConfig(d_model=64, num_heads=4, dropout=0.1)
        autocorr = RestoredAutoCorrelationAttention(config)
        
        x = torch.randn(2, 32, 64)
        output, _ = autocorr.apply_attention(x, x, x)
        assert output.shape == x.shape
        print("✅ RestoredAutoCorrelationAttention wrapper working")
        
    except Exception as e:
        print(f"❌ RestoredAutoCorrelationAttention wrapper failed: {e}")
    
    # Test 3: Meta Learning Wrapper
    try:
        config = RestoredMetaLearningConfig(d_model=64, num_heads=4, dropout=0.1)
        meta = RestoredMetaLearningAttention(config)
        
        x = torch.randn(2, 32, 64)
        output, _ = meta.apply_attention(x, x, x)
        assert output.shape == x.shape
        print("✅ RestoredMetaLearningAttention wrapper working")
        
    except Exception as e:
        print(f"❌ RestoredMetaLearningAttention wrapper failed: {e}")
    
    # Test 4: Registration
    try:
        register_restored_algorithms()
        print("✅ Registration successful")
        
    except Exception as e:
        print(f"❌ Registration failed: {e}")
    
    print("\n🎯 Phase 2 Algorithm Adaptation Complete!")
