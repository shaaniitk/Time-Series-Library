
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelationLayer, AdaptiveAutoCorrelation
from .cross_resolution_attention import CrossResolutionAttention

# Import new Phase 2 components
from .fourier_attention import FourierAttention, FourierBlock, FourierCrossAttention
from .wavelet_attention import WaveletAttention, WaveletDecomposition, AdaptiveWaveletAttention, MultiScaleWaveletAttention
from .enhanced_autocorrelation import EnhancedAutoCorrelation, AdaptiveAutoCorrelationLayer as NewAdaptiveAutoCorrelationLayer, HierarchicalAutoCorrelation
from .bayesian_attention import BayesianAttention, BayesianMultiHeadAttention, VariationalAttention, BayesianCrossAttention
from .adaptive_components import MetaLearningAdapter, AdaptiveMixture
from .temporal_conv_attention import CausalConvolution, TemporalConvNet, ConvolutionalAttention

from utils.logger import logger

class AttentionRegistry:
    """
    A registry for all available attention components.
    """
    _registry = {
        # Legacy components
        "autocorrelation_layer": AutoCorrelationLayer,
        "adaptive_autocorrelation_layer": AdaptiveAutoCorrelationLayer,
        "cross_resolution_attention": CrossResolutionAttention,
        
        # Phase 2: Fourier Attention Components
        "fourier_attention": FourierAttention,
        "fourier_block": FourierBlock,
        "fourier_cross_attention": FourierCrossAttention,
        
        # Phase 2: Wavelet Attention Components
        "wavelet_attention": WaveletAttention,
        "wavelet_decomposition": WaveletDecomposition,
        "adaptive_wavelet_attention": AdaptiveWaveletAttention,
        "multi_scale_wavelet_attention": MultiScaleWaveletAttention,
        
        # Phase 2: Enhanced AutoCorrelation Components
        "enhanced_autocorrelation": EnhancedAutoCorrelation,
        "new_adaptive_autocorrelation_layer": NewAdaptiveAutoCorrelationLayer,
        "hierarchical_autocorrelation": HierarchicalAutoCorrelation,
        
        # Phase 2: Bayesian Attention Components
        "bayesian_attention": BayesianAttention,
        "bayesian_multi_head_attention": BayesianMultiHeadAttention,
        "variational_attention": VariationalAttention,
        "bayesian_cross_attention": BayesianCrossAttention,
        
        # Phase 2: Adaptive Components
        "meta_learning_adapter": MetaLearningAdapter,
        "adaptive_mixture": AdaptiveMixture,
        
        # Phase 2: Temporal Convolution Attention Components
        "causal_convolution": CausalConvolution,
        "temporal_conv_net": TemporalConvNet,
        "convolutional_attention": ConvolutionalAttention,
    }

    @classmethod
    def register(cls, name, component_class):
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered attention component: {name}")

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Attention component '{name}' not found.")
            raise ValueError(f"Attention component '{name}' not found.")
        return component
    
    @classmethod
    def create(cls, name, **kwargs):
        """Create an attention component instance with given parameters"""
        component_class = cls.get(name)
        
        # For most attention components, we need d_model and n_heads
        d_model = kwargs.get('d_model', 64)
        n_heads = kwargs.get('n_heads', 4)
        
        try:
            # Try to instantiate with common parameters
            if name in ['fourier_attention', 'wavelet_attention', 'bayesian_attention']:
                return component_class(d_model, n_heads, **{k: v for k, v in kwargs.items() if k not in ['d_model', 'n_heads']})
            elif name == 'enhanced_autocorrelation':
                # This one has different parameter structure
                return component_class(**kwargs)
            elif name == 'meta_learning_adapter':
                return component_class(d_model, n_heads, **{k: v for k, v in kwargs.items() if k not in ['d_model', 'n_heads']})
            else:
                # Try generic instantiation
                return component_class(**kwargs)
        except Exception as e:
            logger.warning(f"Failed to create {name} with provided kwargs: {e}")
            # Fallback: try with minimal parameters
            try:
                return component_class()
            except:
                raise ValueError(f"Could not instantiate {name} with any parameter combination")

    @classmethod
    def list_available(cls):
        """List all available attention component names"""
        return list(cls._registry.keys())

    @classmethod
    def list_components(cls):
        return list(cls._registry.keys())

def get_attention_component(name, **kwargs):
    component_class = AttentionRegistry.get(name)
    
    # Legacy components
    if name == "autocorrelation_layer":
        autocorrelation = AutoCorrelation(
            mask_flag=kwargs.get('mask_flag', True),
            factor=kwargs.get('factor', 1),
            attention_dropout=kwargs.get('dropout', 0.1),
            output_attention=kwargs.get('output_attention', False)
        )
        return component_class(
            autocorrelation,
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads')
        )
    elif name == "adaptive_autocorrelation_layer":
        autocorrelation = AdaptiveAutoCorrelation(
            mask_flag=kwargs.get('mask_flag', True),
            factor=kwargs.get('factor', 1),
            attention_dropout=kwargs.get('dropout', 0.1),
            output_attention=kwargs.get('output_attention', False)
        )
        return component_class(
            autocorrelation,
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads')
        )
    elif name == "cross_resolution_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_levels=kwargs.get('n_levels'),
            n_heads=kwargs.get('n_heads')
        )
    
    # Phase 2: Fourier Attention Components
    elif name == "fourier_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            seq_len=kwargs.get('seq_len', 96),
            frequency_selection=kwargs.get('frequency_selection', 'adaptive'),
            dropout=kwargs.get('dropout', 0.1),
            temperature=kwargs.get('temperature', 1.0),
            learnable_filter=kwargs.get('learnable_filter', True)
        )
    elif name == "fourier_block":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            seq_len_q=kwargs.get('seq_len_q', 96),
            dropout=kwargs.get('dropout', 0.1),
            activation=kwargs.get('activation', 'gelu')
        )
    elif name == "fourier_cross_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            dropout=kwargs.get('dropout', 0.1),
            temperature=kwargs.get('temperature', 1.0)
        )
    
    # Phase 2: Wavelet Attention Components
    elif name == "wavelet_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            levels=kwargs.get('levels', 3),
            n_levels=kwargs.get('n_levels'),  # Pass as alias
            wavelet_type=kwargs.get('wavelet_type', 'learnable'),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name == "wavelet_decomposition":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            n_levels=kwargs.get('n_levels', 4),
            wavelet_type=kwargs.get('wavelet_type', 'learnable'),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name == "adaptive_wavelet_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            max_levels=kwargs.get('max_levels', 5),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name == "multi_scale_wavelet_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            scales=kwargs.get('scales', [1, 2, 4, 8]),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    # Phase 2: Enhanced AutoCorrelation Components
    elif name == "enhanced_autocorrelation":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            factor=kwargs.get('factor', 1),
            attention_dropout=kwargs.get('dropout', 0.1),
            output_attention=kwargs.get('output_attention', False),
            adaptive_k=kwargs.get('adaptive_k', True),
            multi_scale=kwargs.get('multi_scale', True)
        )
    elif name == "new_adaptive_autocorrelation_layer":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            factor=kwargs.get('factor', 1),
            attention_dropout=kwargs.get('dropout', 0.1),
            output_attention=kwargs.get('output_attention', False),
            adaptive_k=kwargs.get('adaptive_k', True),
            multi_scale=kwargs.get('multi_scale', True)
        )
    elif name == "hierarchical_autocorrelation":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            hierarchy_levels=kwargs.get('hierarchy_levels', [1, 4, 16]),
            factor=kwargs.get('factor', 1)
        )
    
    # Phase 2: Bayesian Attention Components
    elif name == "bayesian_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            dropout=kwargs.get('dropout', 0.1),
            prior_std=kwargs.get('prior_std', 1.0),
            temperature=kwargs.get('temperature', 1.0),
            output_attention=kwargs.get('output_attention', False)
        )
    elif name == "bayesian_multi_head_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            dropout=kwargs.get('dropout', 0.1),
            prior_std=kwargs.get('prior_std', 1.0),
            n_samples=kwargs.get('n_samples', 5),
            output_attention=kwargs.get('output_attention', False)
        )
    elif name == "variational_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            dropout=kwargs.get('dropout', 0.1),
            learn_variance=kwargs.get('learn_variance', True)
        )
    elif name == "bayesian_cross_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            dropout=kwargs.get('dropout', 0.1),
            prior_std=kwargs.get('prior_std', 1.0)
        )
    
    # Phase 2: Adaptive Components
    elif name == "meta_learning_adapter":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            adaptation_steps=kwargs.get('adaptation_steps', 3),
            meta_lr=kwargs.get('meta_lr', 0.01),
            inner_lr=kwargs.get('inner_lr', 0.1),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name == "adaptive_mixture":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            mixture_components=kwargs.get('mixture_components', 3),
            gate_hidden_dim=kwargs.get('gate_hidden_dim', None),
            dropout=kwargs.get('dropout', 0.1),
            temperature=kwargs.get('temperature', 1.0)
        )
    
    # Phase 2: Temporal Convolution Attention Components
    elif name == "causal_convolution":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            kernel_sizes=kwargs.get('kernel_sizes', [3, 5, 7]),
            dilation_rates=kwargs.get('dilation_rates', [1, 2, 4]),
            dropout=kwargs.get('dropout', 0.1),
            activation=kwargs.get('activation', 'gelu')
        )
    elif name == "temporal_conv_net":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            num_levels=kwargs.get('num_levels', 4),
            kernel_size=kwargs.get('kernel_size', 3),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name == "convolutional_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            conv_kernel_size=kwargs.get('conv_kernel_size', 3),
            pool_size=kwargs.get('pool_size', 2),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    raise ValueError(f"Factory not implemented for attention type: {name}")
