"""
Configuration Manager for Modular HF Architecture

This module provides centralized configuration management with integrated
dependency validation, adapter insertion, and compatibility checking.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json

from .dependency_manager import DependencyValidator, ComponentRequirement, DimensionConstraint
from .registry import ComponentRegistry

logger = logging.getLogger(__name__)


@dataclass
class ModularConfig:
    """Complete configuration for modular HF architecture"""
    
    # Model architecture
    seq_len: int = 96
    pred_len: int = 24
    enc_in: int = 7
    c_out: int = 7
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = 'gelu'
    
    # Component selections
    backbone_type: str = 'chronos'
    processor_type: str = 'time_domain'
    attention_type: str = 'multi_head'
    loss_type: str = 'mse'
    embedding_type: str = 'standard'
    output_type: str = 'linear'
    
    # Suite configuration
    suite_name: str = 'HFEnhancedAutoformer'
    use_bayesian: bool = False
    use_quantile: bool = False
    use_hierarchical: bool = False
    
    # Advanced features
    quantile_levels: Optional[List[float]] = None
    uncertainty_method: str = 'none'
    n_samples: int = 50
    
    # Adapter configuration
    auto_insert_adapters: bool = True
    allowed_adapters: List[str] = field(default_factory=lambda: ['linear', 'mlp'])
    
    # Validation settings
    strict_validation: bool = True
    allow_warnings: bool = True
    
    # Device and optimization
    device: str = 'cpu'
    mixed_precision: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModularConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModularConfig':
        """Load from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path: str):
        """Save to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class ConfigurationManager:
    """
    Centralized configuration manager with dependency validation
    
    This class coordinates component configuration, validates dependencies,
    and can automatically insert adapters when needed.
    """
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.validator = DependencyValidator(registry)
        self.validated_configs: Dict[str, ModularConfig] = {}
        
    def create_configuration(self, **kwargs) -> ModularConfig:
        """Create a new configuration with defaults"""
        return ModularConfig(**kwargs)
    
    def validate_and_fix_configuration(self, config: ModularConfig) -> Tuple[ModularConfig, List[str], List[str]]:
        """
        Validate configuration and attempt to fix issues automatically
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (fixed_config, errors, warnings)
        """
        logger.info(f"Validating configuration for {config.suite_name}")
        
        # Convert to dict for validation
        config_dict = config.to_dict()
        
        # Initial validation
        is_valid, errors, warnings = self.validator.validate_configuration(config_dict)
        
        if is_valid:
            logger.info("✅ Configuration is valid")
            return config, [], warnings
        
        # Attempt automatic fixes
        fixed_config = self._attempt_automatic_fixes(config, errors)
        
        # Re-validate fixed configuration
        fixed_config_dict = fixed_config.to_dict()
        is_valid_after_fix, remaining_errors, new_warnings = self.validator.validate_configuration(fixed_config_dict)
        
        if is_valid_after_fix:
            logger.info("✅ Configuration fixed automatically")
            warnings.extend(new_warnings)
            return fixed_config, [], warnings
        else:
            logger.warning("⚠️ Some issues could not be fixed automatically")
            return fixed_config, remaining_errors, warnings + new_warnings
    
    def _attempt_automatic_fixes(self, config: ModularConfig, errors: List[str]) -> ModularConfig:
        """Attempt to fix configuration errors automatically"""
        fixed_config = ModularConfig(**config.to_dict())
        
        for error in errors:
            if "Dimension mismatch" in error and config.auto_insert_adapters:
                logger.info("🔧 Attempting to fix dimension mismatch with adapters")
                # This would require more sophisticated parsing of error messages
                # For now, we'll implement basic dimension alignment
                self._fix_dimension_mismatches(fixed_config)
            
            elif "requires capability" in error:
                logger.info("🔧 Attempting to fix capability requirements")
                self._fix_capability_requirements(fixed_config, error)
            
            elif "Component" in error and "not found" in error:
                logger.info("🔧 Attempting to fix missing components")
                self._fix_missing_components(fixed_config, error)
        
        return fixed_config
    
    def _fix_dimension_mismatches(self, config: ModularConfig):
        """Attempt to fix dimension mismatches"""
        # For now, ensure all model dimensions are consistent
        # In a more sophisticated implementation, we'd insert adapter layers
        pass
    
    def _fix_capability_requirements(self, config: ModularConfig, error: str):
        """Attempt to fix capability requirement mismatches"""
        # Parse error to understand what capability is needed
        if "frequency_domain_compatible" in error and "attention" in error:
            # Switch to a compatible attention mechanism
            available_attention = self.registry.list_components('attention')['attention']
            for attention_name in available_attention:
                metadata = self.validator.get_component_metadata('attention', attention_name)
                if 'frequency_domain_compatible' in metadata.capabilities:
                    config.attention_type = attention_name
                    logger.info(f"🔧 Switched attention to {attention_name}")
                    break
    
    def _fix_missing_components(self, config: ModularConfig, error: str):
        """Attempt to fix missing component references"""
        # Extract component type and suggest alternatives
        if "backbone" in error.lower():
            available = self.registry.list_components('backbone')['backbone']
            if available:
                config.backbone_type = available[0]
                logger.info(f"🔧 Switched to available backbone: {available[0]}")
        
        elif "processor" in error.lower():
            available = self.registry.list_components('processor')['processor']
            if available:
                config.processor_type = available[0]
                logger.info(f"🔧 Switched to available processor: {available[0]}")
    
    def suggest_configurations(self, base_config: ModularConfig, 
                             max_suggestions: int = 5) -> List[Tuple[ModularConfig, str]]:
        """
        Suggest alternative configurations based on a base configuration
        
        Args:
            base_config: Base configuration to modify
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of (config, description) tuples
        """
        suggestions = []
        
        # Get available components
        all_components = self.registry.list_components()
        
        # Try different backbone combinations
        for backbone in all_components.get('backbone', []):
            if backbone == base_config.backbone_type:
                continue
                
            test_config = ModularConfig(**base_config.to_dict())
            test_config.backbone_type = backbone
            
            is_valid, _, _ = self.validator.validate_configuration(test_config.to_dict())
            if is_valid:
                suggestions.append((
                    test_config,
                    f"Alternative backbone: {backbone}"
                ))
                
            if len(suggestions) >= max_suggestions:
                break
        
        # Try different processor combinations
        for processor in all_components.get('processor', []):
            if processor == base_config.processor_type:
                continue
                
            test_config = ModularConfig(**base_config.to_dict())
            test_config.processor_type = processor
            
            is_valid, _, _ = self.validator.validate_configuration(test_config.to_dict())
            if is_valid:
                suggestions.append((
                    test_config,
                    f"Alternative processor: {processor}"
                ))
                
            if len(suggestions) >= max_suggestions:
                break
        
        return suggestions[:max_suggestions]
    
    def get_compatible_components(self, config: ModularConfig, 
                                component_type: str) -> List[str]:
        """
        Get list of components compatible with current configuration
        
        Args:
            config: Current configuration
            component_type: Type of component to check
            
        Returns:
            List of compatible component names
        """
        existing_components = {
            'backbone': config.backbone_type,
            'processor': config.processor_type,
            'attention': config.attention_type,
            'loss': config.loss_type,
            'embedding': config.embedding_type,
            'output': config.output_type
        }
        
        return self.validator.suggest_compatible_components(
            component_type, existing_components
        )
    
    def create_test_configurations(self) -> List[Tuple[ModularConfig, str]]:
        """Create a set of test configurations for validation"""
        test_configs = []
        
        # Get available components first
        available = self.registry.list_components()
        
        # Use first available components as fallbacks
        default_backbone = available.get('backbone', ['chronos'])[0] if available.get('backbone') else 'chronos'
        default_processor = available.get('processor', ['time_domain'])[0] if available.get('processor') else 'time_domain'
        default_attention = available.get('attention', ['multi_head'])[0] if available.get('attention') else 'multi_head'
        default_loss = available.get('loss', ['mse'])[0] if available.get('loss') else 'mock_loss'
        
        # Basic configurations
        basic_config = ModularConfig(
            backbone_type=default_backbone,
            processor_type=default_processor,
            attention_type=default_attention,
            loss_type=default_loss
        )
        test_configs.append((basic_config, "Basic Configuration"))
        
        # Bayesian configuration (if bayesian loss available)
        bayesian_losses = [l for l in available.get('loss', []) if 'bayesian' in l]
        if bayesian_losses:
            bayesian_config = ModularConfig(
                backbone_type=default_backbone,
                processor_type=default_processor,
                attention_type=default_attention,
                loss_type=bayesian_losses[0],
                use_bayesian=True,
                suite_name='HFBayesianAutoformer'
            )
            test_configs.append((bayesian_config, "Bayesian Configuration"))
        
        # Frequency domain configuration (if available)
        freq_processors = [p for p in available.get('processor', []) if 'frequency' in p]
        freq_attention = [a for a in available.get('attention', []) if 'autocorr' in a]
        
        if freq_processors and freq_attention:
            freq_config = ModularConfig(
                backbone_type=default_backbone,
                processor_type=freq_processors[0],
                attention_type=freq_attention[0],
                loss_type=default_loss
            )
            test_configs.append((freq_config, "Frequency Domain Configuration"))
        
        return test_configs
    
    def export_configuration_report(self, config: ModularConfig, 
                                  output_path: str):
        """Export detailed configuration report"""
        is_valid, errors, warnings = self.validator.validate_configuration(config.to_dict())
        
        report = {
            'configuration': config.to_dict(),
            'validation': {
                'is_valid': is_valid,
                'errors': errors,
                'warnings': warnings
            },
            'component_info': {},
            'compatibility_matrix': {},
            'suggestions': []
        }
        
        # Add component information
        components = {
            'backbone': config.backbone_type,
            'processor': config.processor_type,
            'attention': config.attention_type,
            'loss': config.loss_type
        }
        
        for comp_type, comp_name in components.items():
            try:
                metadata = self.validator.get_component_metadata(comp_type, comp_name)
                report['component_info'][f"{comp_type}:{comp_name}"] = {
                    'capabilities': list(metadata.capabilities),
                    'requirements': [
                        {
                            'target': req.target_component,
                            'capability': req.capability_needed,
                            'optional': req.is_optional
                        }
                        for req in metadata.requirements
                    ],
                    'compatibility_tags': list(metadata.compatibility_tags)
                }
            except Exception as e:
                report['component_info'][f"{comp_type}:{comp_name}"] = {'error': str(e)}
        
        # Add suggestions if configuration is invalid
        if not is_valid:
            suggestions = self.suggest_configurations(config, max_suggestions=3)
            report['suggestions'] = [
                {
                    'configuration': suggestion.to_dict(),
                    'description': description
                }
                for suggestion, description in suggestions
            ]
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Configuration report exported to {output_path}")
