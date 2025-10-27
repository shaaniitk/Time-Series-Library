"""
Component Configuration Manager for Enhanced SOTA PGAT
Provides systematic component enabling/disabling with validation
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ComponentConfig:
    """Configuration for individual components"""
    
    # Core model parameters
    seq_len: int = 24
    pred_len: int = 6
    enc_in: int = 118
    c_out: int = 4
    d_model: int = 128
    n_heads: int = 8
    dropout: float = 0.1
    
    # Component toggles
    use_multi_scale_patching: bool = False
    use_hierarchical_mapper: bool = False
    use_stochastic_learner: bool = False
    use_gated_graph_combiner: bool = False
    use_mixture_decoder: bool = False
    
    # Component-specific parameters
    num_wave_features: Optional[int] = None
    mdn_components: int = 3
    mixture_multivariate_mode: str = 'independent'
    num_wave_patch_latents: int = 64
    num_target_patch_latents: int = 24
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    train_epochs: int = 10
    patience: int = 5


class ComponentConfigurationManager:
    """Manages systematic component configuration"""
    
    def __init__(self):
        self.base_config = ComponentConfig()
        self.component_dependencies = {
            'use_gated_graph_combiner': ['use_stochastic_learner'],  # Combiner works better with stochastic
            'use_mixture_decoder': [],  # No dependencies
            'use_hierarchical_mapper': [],  # No dependencies
            'use_multi_scale_patching': [],  # No dependencies
            'use_stochastic_learner': []  # No dependencies
        }
        
        self.component_order = [
            'use_multi_scale_patching',
            'use_hierarchical_mapper', 
            'use_stochastic_learner',
            'use_gated_graph_combiner',
            'use_mixture_decoder'
        ]
    
    def get_baseline_config(self) -> ComponentConfig:
        """Get baseline configuration with all components disabled"""
        config = ComponentConfig()
        # Ensure all components are disabled
        for component in self.component_order:
            setattr(config, component, False)
        return config
    
    def get_progressive_configs(self) -> List[ComponentConfig]:
        """Get progressive configurations adding one component at a time"""
        configs = []
        
        # Start with baseline
        current_config = self.get_baseline_config()
        configs.append(ComponentConfig(**asdict(current_config)))
        
        # Add components progressively
        for component in self.component_order:
            # Create new config with this component enabled
            new_config = ComponentConfig(**asdict(current_config))
            setattr(new_config, component, True)
            
            # Check and enable dependencies
            if component in self.component_dependencies:
                for dep in self.component_dependencies[component]:
                    setattr(new_config, dep, True)
            
            configs.append(new_config)
            current_config = new_config
        
        return configs
    
    def get_ablation_configs(self) -> List[ComponentConfig]:
        """Get ablation study configurations (full model minus one component)"""
        full_config = self.get_full_config()
        configs = []
        
        for component in self.component_order:
            # Create config with all components except this one
            ablation_config = ComponentConfig(**asdict(full_config))
            setattr(ablation_config, component, False)
            
            # Handle dependencies - if we disable a component that others depend on,
            # we might need to disable dependents too
            if component == 'use_stochastic_learner' and ablation_config.use_gated_graph_combiner:
                # Combiner works better with stochastic, but can work without
                pass  # Keep combiner enabled for this ablation
            
            configs.append(ablation_config)
        
        return configs
    
    def get_full_config(self) -> ComponentConfig:
        """Get full configuration with all components enabled"""
        config = ComponentConfig()
        for component in self.component_order:
            setattr(config, component, True)
        return config
    
    def get_recommended_configs(self) -> Dict[str, ComponentConfig]:
        """Get recommended configurations for different use cases"""
        return {
            'minimal': self.get_baseline_config(),
            'stable': ComponentConfig(
                use_multi_scale_patching=True,
                use_hierarchical_mapper=True,
                use_stochastic_learner=False,
                use_gated_graph_combiner=False,
                use_mixture_decoder=False
            ),
            'advanced': ComponentConfig(
                use_multi_scale_patching=True,
                use_hierarchical_mapper=True,
                use_stochastic_learner=True,
                use_gated_graph_combiner=True,
                use_mixture_decoder=False
            ),
            'full': self.get_full_config()
        }
    
    def validate_config(self, config: ComponentConfig) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        warnings = []
        
        # Check dependencies
        for component, deps in self.component_dependencies.items():
            if getattr(config, component):
                for dep in deps:
                    if not getattr(config, dep):
                        warnings.append(f"{component} enabled but dependency {dep} is disabled")
        
        # Check parameter consistency
        if config.d_model % config.n_heads != 0:
            warnings.append(f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})")
        
        if config.num_wave_features is not None:
            if config.num_wave_features + config.c_out > config.enc_in:
                warnings.append(f"num_wave_features + c_out cannot exceed enc_in")
        
        return warnings
    
    def save_config(self, config: ComponentConfig, filepath: str):
        """Save configuration to YAML file"""
        config_dict = asdict(config)
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def load_config(self, filepath: str) -> ComponentConfig:
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return ComponentConfig(**config_dict)
    
    def create_training_configs(self, output_dir: str = "configs/systematic_testing"):
        """Create a set of training configurations for systematic testing"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Progressive configs
        progressive_configs = self.get_progressive_configs()
        for i, config in enumerate(progressive_configs):
            filename = f"progressive_step_{i:02d}.yaml"
            self.save_config(config, output_path / filename)
        
        # Recommended configs
        recommended_configs = self.get_recommended_configs()
        for name, config in recommended_configs.items():
            filename = f"recommended_{name}.yaml"
            self.save_config(config, output_path / filename)
        
        # Ablation configs
        ablation_configs = self.get_ablation_configs()
        for i, config in enumerate(ablation_configs):
            component_name = self.component_order[i].replace('use_', '')
            filename = f"ablation_without_{component_name}.yaml"
            self.save_config(config, output_path / filename)
        
        print(f"Created {len(progressive_configs) + len(recommended_configs) + len(ablation_configs)} configuration files in {output_path}")
    
    def print_config_summary(self, config: ComponentConfig):
        """Print a summary of the configuration"""
        print("📋 Configuration Summary")
        print("-" * 40)
        print(f"Model: d_model={config.d_model}, n_heads={config.n_heads}")
        print(f"Data: seq_len={config.seq_len}, pred_len={config.pred_len}, enc_in={config.enc_in}, c_out={config.c_out}")
        print(f"Training: lr={config.learning_rate}, batch_size={config.batch_size}, epochs={config.train_epochs}")
        
        print("\n🔧 Components:")
        for component in self.component_order:
            status = "✅" if getattr(config, component) else "❌"
            name = component.replace('use_', '').replace('_', ' ').title()
            print(f"  {status} {name}")
        
        # Check for warnings
        warnings = self.validate_config(config)
        if warnings:
            print(f"\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  - {warning}")


def main():
    """Demonstrate the configuration manager"""
    manager = ComponentConfigurationManager()
    
    print("🚀 Component Configuration Manager Demo")
    print("=" * 50)
    
    # Show recommended configurations
    recommended = manager.get_recommended_configs()
    
    for name, config in recommended.items():
        print(f"\n📋 {name.upper()} Configuration:")
        manager.print_config_summary(config)
    
    # Create training configs
    print(f"\n📁 Creating systematic testing configurations...")
    manager.create_training_configs()
    
    print(f"\n✅ Configuration manager demo complete!")


if __name__ == "__main__":
    main()