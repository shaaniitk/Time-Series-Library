"""
Modular Dimension Manager

Extends the base DimensionManager to handle complex dimension requirements
for modular components (encoders, decoders, decomposition, etc.).

This manager serves as the single source of truth for all dimension calculations
across the modular framework, ensuring components receive the correct dimensions
based on their type and position in the data flow.
"""

import dataclasses
from typing import List, Dict, Any, Optional, Tuple
from layers.modular.dimensions.dimension_manager import DimensionManager


@dataclasses.dataclass
class ModularDimensionManager(DimensionManager):
    """
    Enhanced dimension manager for modular components.
    
    Extends the base DimensionManager to handle component-specific dimension
    requirements and data flow transformations.
    """
    
    # Additional modular framework parameters
    d_model: int = 512  # Model embedding dimension
    seq_len: int = 96   # Input sequence length
    pred_len: int = 24  # Prediction length
    label_len: int = 48 # Label length for decoder
    
    def __post_init__(self):
        """Calculate all derived properties including modular component dimensions."""
        # Call parent's post_init for base dimension calculations
        super().__post_init__()
        
        # Calculate component-specific dimensions
        self._calculate_embedding_dimensions()
        self._calculate_decomposition_dimensions()
        self._calculate_encoder_dimensions()
        self._calculate_decoder_dimensions()
        self._calculate_attention_dimensions()
        self._calculate_output_dimensions()
    
    def _calculate_embedding_dimensions(self):
        """Calculate dimensions for embedding layers."""
        self.enc_embedding_in = self.enc_in
        self.dec_embedding_in = self.dec_in
        self.embedding_out = self.d_model
    
    def _calculate_decomposition_dimensions(self):
        """Calculate dimensions for decomposition components."""
        # Init decomposition works with raw input dimensions
        self.init_decomp_in = self.dec_in
        self.init_decomp_out = self.dec_in  # Output maintains input feature count
        
        # Encoder/Decoder decomposition works with embedded dimensions
        self.encoder_decomp_in = self.d_model
        self.encoder_decomp_out = self.d_model
        
        self.decoder_decomp_in = self.d_model
        self.decoder_decomp_out = self.d_model
    
    def _calculate_encoder_dimensions(self):
        """Calculate dimensions for encoder components."""
        self.encoder_in = self.d_model  # From embedding
        self.encoder_out = self.d_model # Maintains d_model
        self.encoder_seq_len = self.seq_len
    
    def _calculate_decoder_dimensions(self):
        """Calculate dimensions for decoder components."""
        self.decoder_in = self.d_model  # From embedding
        self.decoder_out = self.d_model # Maintains d_model
        self.decoder_seq_len = self.label_len + self.pred_len
    
    def _calculate_attention_dimensions(self):
        """Calculate dimensions for attention components."""
        self.attention_d_model = self.d_model
        # Attention components maintain d_model dimension
        self.attention_in = self.d_model
        self.attention_out = self.d_model
    
    def _calculate_output_dimensions(self):
        """Calculate dimensions for output head components."""
        self.output_head_in = self.d_model  # From decoder
        
        # For quantile output heads, the internal model output should be c_out * num_quantiles
        # but the evaluation/expected output remains c_out_evaluation
        if self.loss_function in ['quantile', 'pinball'] and self.quantiles:
            self.output_head_out = self.c_out_evaluation * len(self.quantiles)  # Internal model output
            self.output_head_eval_out = self.c_out_evaluation  # Expected evaluation output
        else:
            self.output_head_out = self.c_out_model  # Final model output
            self.output_head_eval_out = self.c_out_model
    
    def get_component_dimensions(self, component_type: str, component_stage: str = None) -> Dict[str, Any]:
        """
        Get dimension parameters for a specific component type.
        
        Args:
            component_type: Type of component ('decomposition', 'encoder', 'decoder', etc.)
            component_stage: Stage in pipeline ('init', 'encoder', 'decoder', etc.)
            
        Returns:
            Dictionary of dimension parameters for the component
        """
        if component_type == 'decomposition':
            return self._get_decomposition_dimensions(component_stage)
        elif component_type == 'encoder':
            return self._get_encoder_dimensions()
        elif component_type == 'decoder':
            return self._get_decoder_dimensions()
        elif component_type == 'attention':
            return self._get_attention_dimensions()
        elif component_type == 'embedding':
            return self._get_embedding_dimensions(component_stage)
        elif component_type == 'output_head':
            return self._get_output_head_dimensions()
        elif component_type == 'sampling':
            return self._get_sampling_dimensions()
        else:
            raise ValueError(f"Unknown component type: {component_type}")
    
    def _get_decomposition_dimensions(self, stage: str = None) -> Dict[str, Any]:
        """Get dimensions for decomposition components."""
        if stage == 'init':
            return {
                'input_dim': self.init_decomp_in,
                'seq_len': self.decoder_seq_len,
                'd_model': self.init_decomp_in,  # For wavelet decomp in raw space
            }
        elif stage == 'encoder':
            return {
                'input_dim': self.encoder_decomp_in,
                'seq_len': self.encoder_seq_len,
                'd_model': self.encoder_decomp_out,
            }
        elif stage == 'decoder':
            return {
                'input_dim': self.decoder_decomp_in,
                'seq_len': self.decoder_seq_len,
                'd_model': self.decoder_decomp_out,
            }
        else:
            # Default to encoder stage
            return {
                'input_dim': self.d_model,
                'seq_len': self.seq_len,
                'd_model': self.d_model,
            }
    
    def _get_encoder_dimensions(self) -> Dict[str, Any]:
        """Get dimensions for encoder components."""
        return {
            'd_model': self.encoder_out,
            'seq_len': self.encoder_seq_len,
            'input_dim': self.encoder_in,
        }
    
    def _get_decoder_dimensions(self) -> Dict[str, Any]:
        """Get dimensions for decoder components."""
        return {
            'd_model': self.decoder_out,
            'seq_len': self.decoder_seq_len,
            'input_dim': self.decoder_in,
            'c_out': self.c_out_evaluation,  # For trend component
        }
    
    def _get_attention_dimensions(self) -> Dict[str, Any]:
        """Get dimensions for attention components."""
        return {
            'd_model': self.attention_d_model,
            'input_dim': self.attention_in,
        }
    
    def _get_embedding_dimensions(self, stage: str = None) -> Dict[str, Any]:
        """Get dimensions for embedding components."""
        if stage == 'encoder':
            return {
                'c_in': self.enc_embedding_in,
                'd_model': self.embedding_out,
            }
        elif stage == 'decoder':
            return {
                'c_in': self.dec_embedding_in,
                'd_model': self.embedding_out,
            }
        else:
            return {
                'c_in': self.enc_in,
                'd_model': self.embedding_out,
            }
    
    def _get_output_head_dimensions(self) -> Dict[str, Any]:
        """Get dimensions for output head components."""
        dimensions = {
            'd_model': self.output_head_in,
            'c_out': self.output_head_out,
        }
        
        # Add quantile-specific information if applicable
        if self.loss_function in ['quantile', 'pinball'] and self.quantiles:
            dimensions.update({
                'num_quantiles': len(self.quantiles),
                'c_out_eval': getattr(self, 'output_head_eval_out', self.c_out_evaluation),
                'quantile_levels': self.quantiles,
            })
            
        return dimensions
    
    def _get_sampling_dimensions(self) -> Dict[str, Any]:
        """Get dimensions for sampling components."""
        return {
            'input_dim': self.d_model,
            'output_dim': self.c_out_model,
            'num_quantiles': len(self.quantiles) if self.quantiles else 1,
        }
    
    def get_data_flow_info(self) -> Dict[str, Any]:
        """
        Get complete data flow dimension information.
        
        Returns:
            Dictionary describing dimension transformations through the pipeline
        """
        return {
            'input_data': {
                'encoder': f"[batch, {self.seq_len}, {self.enc_in}]",
                'decoder': f"[batch, {self.decoder_seq_len}, {self.dec_in}]",
            },
            'after_embedding': {
                'encoder': f"[batch, {self.seq_len}, {self.d_model}]",
                'decoder': f"[batch, {self.decoder_seq_len}, {self.d_model}]",
            },
            'after_encoder': f"[batch, {self.seq_len}, {self.d_model}]",
            'after_decoder': f"[batch, {self.decoder_seq_len}, {self.d_model}]",
            'final_output': f"[batch, {self.pred_len}, {self.c_out_model}]",
            'evaluation_output': f"[batch, {self.pred_len}, {self.c_out_evaluation}]",
        }
    
    def validate_component_compatibility(self, components: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate that component choices are dimensionally compatible and follow proper patterns.
        
        Args:
            components: Dictionary mapping component types to component names
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check basic dimension compatibility
        if self.d_model <= 0:
            issues.append("d_model must be positive")
        
        if self.seq_len <= 0:
            issues.append("seq_len must be positive")
        
        if self.pred_len <= 0:
            issues.append("pred_len must be positive")
        
        # Cross-component validation for Bayesian models
        sampling_type = components.get('sampling_type', 'none')
        output_head_type = components.get('output_head_type', 'standard')
        
        if sampling_type == 'bayesian':
            # Bayesian sampling requires Bayesian loss functions
            if self.loss_function not in ['bayesian', 'bayesian_adaptive', 'bayesian_frequency', 'bayesian_quantile']:
                issues.append(f"Bayesian sampling requires Bayesian loss function, got '{self.loss_function}'. "
                             f"Use: 'bayesian', 'bayesian_adaptive', 'bayesian_frequency', or 'bayesian_quantile'")
        
        # Quantile-specific validations
        if self.loss_function in ['quantile', 'pinball', 'bayesian_quantile']:
            if not self.quantiles:
                issues.append(f"Quantile loss '{self.loss_function}' requires quantiles to be specified")
            else:
                # For quantile losses, output head should handle quantile dimensions
                if output_head_type == 'quantile':
                    # Quantile output head should output c_out * num_quantiles
                    expected_output = self.c_out_evaluation * len(self.quantiles)
                    if self.c_out_model != expected_output:
                        issues.append(f"Quantile output head dimension mismatch: "
                                     f"expected {expected_output} (c_out * num_quantiles), got {self.c_out_model}")
                elif output_head_type == 'standard':
                    # Standard output head with quantile loss should output c_out_evaluation
                    if self.c_out_model != self.c_out_evaluation:
                        issues.append(f"Standard output head with quantile loss should output {self.c_out_evaluation}, "
                                     f"got {self.c_out_model}")
        
        # Bayesian + Quantile combination validation
        if sampling_type == 'bayesian' and self.loss_function == 'bayesian_quantile':
            if not self.quantiles:
                issues.append("Bayesian quantile model requires both Bayesian sampling AND quantiles to be specified")
            
            # For Bayesian quantile, we expect quantile output head
            if output_head_type != 'quantile':
                issues.append(f"Bayesian quantile model should use 'quantile' output head, got '{output_head_type}'")
        
        # MSE with Bayesian is invalid
        if sampling_type == 'bayesian' and self.loss_function == 'mse':
            issues.append("Bayesian sampling with MSE loss is invalid. MSE doesn't provide uncertainty quantification. "
                         "Use 'bayesian' (wraps MSE with KL) or 'bayesian_quantile' for proper uncertainty.")
        
        # Check mode consistency
        if self.mode not in ['S', 'MS', 'M']:
            issues.append(f"Unknown mode: {self.mode}")
        
        # Validate loss function types
        valid_loss_functions = [
            'mse', 'mae', 'huber', 'quantile', 'pinball',
            'bayesian', 'bayesian_adaptive', 'bayesian_frequency', 'bayesian_quantile',
            'adaptive_autoformer', 'frequency_aware'
        ]
        if self.loss_function not in valid_loss_functions:
            issues.append(f"Unknown loss function: {self.loss_function}. Valid options: {valid_loss_functions}")
        
        # Validate output head types
        valid_output_heads = ['standard', 'quantile', 'adaptive', 'hierarchical']
        if output_head_type not in valid_output_heads:
            issues.append(f"Unknown output head type: {output_head_type}. Valid options: {valid_output_heads}")
        
        # Validate sampling types
        valid_sampling_types = ['none', 'bayesian', 'monte_carlo', 'ensemble']
        if sampling_type not in valid_sampling_types:
            issues.append(f"Unknown sampling type: {sampling_type}. Valid options: {valid_sampling_types}")
        
        return len(issues) == 0, issues
    
    def __repr__(self):
        base_repr = super().__repr__()
        modular_info = (
            f"\n  - Modular Dims: d_model={self.d_model}, seq_len={self.seq_len}, pred_len={self.pred_len}\n"
            f"  - Flow: Raw({self.enc_in}) -> Embed({self.d_model}) -> Output({self.c_out_model})"
        )
        return base_repr + modular_info


def create_modular_dimension_manager(configs) -> ModularDimensionManager:
    """
    Create a ModularDimensionManager from configuration.
    
    Args:
        configs: Configuration object with necessary parameters
        
    Returns:
        Configured ModularDimensionManager instance
    """
    # Extract core dimension parameters from config
    mode = getattr(configs, 'features', 'M')  # Default to multivariate
    
    # Use the dimensions already calculated by the base config
    c_out_evaluation = getattr(configs, 'c_out_evaluation', 1)
    c_out_model = getattr(configs, 'c_out', c_out_evaluation)
    enc_in = getattr(configs, 'enc_in', c_out_evaluation)
    dec_in = getattr(configs, 'dec_in', c_out_evaluation)
    
    # Create feature lists based on actual dimensions
    target_features = [f"target_{i}" for i in range(c_out_evaluation)]
    all_features = [f"feature_{i}" for i in range(enc_in)]
    
    # Extract other parameters
    loss_function = getattr(configs, 'loss_function_type', 'mse')
    quantiles = getattr(configs, 'quantile_levels', [])
    
    # Create the manager with explicit dimension parameters
    manager = ModularDimensionManager(
        mode=mode,
        target_features=target_features,
        all_features=all_features,
        loss_function=loss_function,
        quantiles=quantiles,
        d_model=getattr(configs, 'd_model', 512),
        seq_len=getattr(configs, 'seq_len', 96),
        pred_len=getattr(configs, 'pred_len', 24),
        label_len=getattr(configs, 'label_len', 48),
    )
    
    # Override the calculated dimensions with config values to ensure consistency
    manager.c_out_evaluation = c_out_evaluation
    manager.c_out_model = c_out_model
    manager.enc_in = enc_in
    manager.dec_in = dec_in
    
    # Recalculate derived dimensions with the correct base values
    manager._calculate_output_dimensions()
    
    return manager


# Example usage
if __name__ == "__main__":
    from argparse import Namespace
    
    # Test configuration
    test_configs = Namespace(
        features='MS',
        enc_in=10,
        dec_in=10,
        c_out=7,
        c_out_evaluation=7,
        d_model=512,
        seq_len=96,
        pred_len=24,
        label_len=48,
        loss_function_type='mse',
        quantile_levels=[]
    )
    
    # Create dimension manager
    dim_manager = create_modular_dimension_manager(test_configs)
    
    print("=== Modular Dimension Manager ===")
    print(dim_manager)
    print("\n=== Component Dimensions ===")
    
    # Test component dimension calculations
    components = ['decomposition', 'encoder', 'decoder', 'attention', 'embedding', 'output_head']
    for comp_type in components:
        if comp_type == 'decomposition':
            for stage in ['init', 'encoder', 'decoder']:
                dims = dim_manager.get_component_dimensions(comp_type, stage)
                print(f"{comp_type}({stage}): {dims}")
        elif comp_type == 'embedding':
            for stage in ['encoder', 'decoder']:
                dims = dim_manager.get_component_dimensions(comp_type, stage)
                print(f"{comp_type}({stage}): {dims}")
        else:
            dims = dim_manager.get_component_dimensions(comp_type)
            print(f"{comp_type}: {dims}")
    
    print("\n=== Data Flow ===")
    flow_info = dim_manager.get_data_flow_info()
    for stage, dims in flow_info.items():
        print(f"{stage}: {dims}")
    
    print("\n=== Validation ===")
    is_valid, issues = dim_manager.validate_component_compatibility({})
    print(f"Valid: {is_valid}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")