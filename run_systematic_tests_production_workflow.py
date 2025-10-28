#!/usr/bin/env python3
"""
🚀 ADVANCED COMPONENT TESTING - Celestial Enhanced PGAT Modular
Uses the exact same data handling, scaling, and training loop as train_celestial_production.py
but with lightweight configs to systematically test advanced components

🎯 ADVANCED COMPONENTS TESTED:
1. Multi-Scale Context Fusion (4 modes: simple, gated, attention, multi_scale)
2. Stochastic Control & Learning (probabilistic graph learning)
3. Hierarchical Temporal-Spatial Mapping (enhanced feature extraction)
4. Advanced Graph Combiners (Petri Net vs Gated vs Standard)
5. Enhanced Decoders (MDN, Mixture, Target Autocorrelation)
6. Celestial-Target Attention (deterministic future conditioning)
7. Efficient Covariate Interaction (partitioned processing)
8. Dynamic Spatiotemporal Encoding (time-varying graphs)
"""

import os
import sys
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the proven production training function
from scripts.train.train_celestial_production import train_celestial_pgat_production


class AdvancedComponentTester:
    """Test advanced components using the exact production workflow with modular Celestial Enhanced PGAT"""
    
    def __init__(self, results_dir='results/advanced_component_tests'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Base lightweight config optimized for advanced component testing
        self.base_config = {
            # Required fields from production config
            'model': 'Celestial_Enhanced_PGAT',
            'data': 'custom',
            'root_path': './data',
            'data_path': 'prepared_financial_data.csv',
            'features': 'MS',
            'target': 'log_Open,log_High,log_Low,log_Close',
            'embed': 'timeF',
            'freq': 'd',
            
            # Sequence settings - Optimized for component testing
            'seq_len': 96,    # Reasonable sequence for testing
            'label_len': 48,
            'pred_len': 24,   # Standard prediction horizon
            
            # Data split settings - Must be larger than pred_len
            'validation_length': 50,  # Larger than pred_len=24
            'test_length': 50,        # Larger than pred_len=24
            
            # Model configuration - Optimized for advanced component testing
            'd_model': 256,   # Larger model for better component differentiation
            'n_heads': 8,     # Compatible heads: 256 ÷ 8 = 32
            'e_layers': 4,    # Deeper for advanced component interactions
            'd_layers': 3,    # More decoder layers for enhanced decoders
            'd_ff': 512,      # Larger feed-forward for richer representations
            'dropout': 0.1,
            
            # Input/Output dimensions - Fixed for Celestial model
            'enc_in': 113,    # Celestial model expects 113 features
            'dec_in': 113,    # Celestial model expects 113 features  
            'c_out': 4,
            
            # Training configuration - Balanced for component comparison
            'train_epochs': 8,  # Enough epochs to see component convergence differences
            'batch_size': 4,    # Larger batch for stable gradient estimates
            'learning_rate': 0.0005,  # Conservative LR for stable training
            'patience': 10,     # Higher patience for advanced components
            'lradj': 'warmup_cosine',
            'warmup_epochs': 2,
            'min_lr': 1e-7,
            'weight_decay': 0.0001,
            'clip_grad_norm': 1.0,
            
            # Production workflow settings
            'use_gpu': True,
            'mixed_precision': False,
            'gradient_accumulation_steps': 1,
            
            # 🌟 CELESTIAL SYSTEM - Advanced Configuration
            'use_celestial_graph': True,
            'aggregate_waves_to_celestial': True,
            'celestial_fusion_layers': 2,
            'num_celestial_bodies': 13,
            'celestial_dim': 32,
            'num_message_passing_steps': 2,
            'edge_feature_dim': 8,
            'use_temporal_attention': True,
            'use_spatial_attention': True,
            'bypass_spatiotemporal_with_petri': True,
            'num_input_waves': 113,
            'target_wave_indices': [0, 1, 2, 3],
            
            # 🎯 ADVANCED COMPONENT DEFAULTS (will be systematically overridden)
            # Context Fusion
            'use_multi_scale_context': True,
            'context_fusion_mode': 'multi_scale',  # Will test: simple, gated, attention, multi_scale
            'short_term_kernel_size': 3,
            'medium_term_kernel_size': 15,
            'long_term_kernel_size': 0,  # Global
            'context_fusion_dropout': 0.1,
            'enable_context_diagnostics': True,
            
            # Stochastic Components
            'use_stochastic_learner': False,
            'use_stochastic_control': False,
            'stochastic_temperature_start': 1.0,
            'stochastic_temperature_end': 0.1,
            'stochastic_decay_steps': 1000,
            
            # Hierarchical Mapping
            'use_hierarchical_mapping': False,
            'use_hierarchical_mapper': False,  # Alias
            
            # Graph Combiners
            'use_petri_net_combiner': True,  # Will test: petri_net vs gated vs standard
            'use_gated_graph_combiner': False,
            
            # Enhanced Decoders
            'use_mixture_decoder': False,
            'use_sequential_mixture_decoder': False,
            'enable_mdn_decoder': False,
            'mdn_components': 5,
            'mdn_sigma_min': 0.001,
            'mdn_use_softplus': True,
            
            # Target Processing
            'use_target_autocorrelation': True,
            'target_autocorr_layers': 2,
            
            # Celestial-Target Attention
            'use_celestial_target_attention': True,
            'celestial_target_use_gated_fusion': True,
            'celestial_target_diagnostics': False,  # Disable for speed
            'use_c2t_edge_bias': True,
            'c2t_edge_bias_weight': 0.2,
            
            # Efficient Processing
            'use_efficient_covariate_interaction': False,
            'enable_target_covariate_attention': False,
            
            # Dynamic Encoding
            'use_dynamic_spatiotemporal_encoder': True,
            
            # Calendar and Additional Effects
            'use_calendar_effects': True,
            'calendar_embedding_dim': 64,  # d_model // 4
            
            # Loss and Regularization
            'reg_loss_weight': 0.0005,
            'kl_weight': 0.00005,
            'c2t_aux_rel_loss_weight': 0.0,
            'loss': {'type': 'mse'},  # Will test advanced losses too
            
            # Training Control
            'save_checkpoints': False,
            'checkpoint_interval': 1,
            'log_interval': 3,
            'save_best_only': True,
            
            # Diagnostics and Logging
            'enable_memory_diagnostics': False,
            'memory_log_interval': 20,
            'collect_diagnostics': True,
            'enable_fusion_diagnostics': False,  # Disable for speed
            'verbose_logging': False,
        }
        
        # 🎯 ADVANCED COMPONENT CONFIGURATIONS
        self.advanced_components = {
            'context_fusion_modes': ['simple', 'gated', 'attention', 'multi_scale'],
            'graph_combiners': ['standard', 'gated', 'petri_net'],
            'decoder_types': ['standard', 'mixture', 'mdn', 'autocorr'],
            'stochastic_modes': ['none', 'learner', 'control', 'both'],
            'encoding_modes': ['static', 'dynamic'],
        }
    
    def create_test_config(self, config_name: str, **component_overrides) -> str:
        """Create a test config file with component overrides"""
        config = self.base_config.copy()
        config.update(component_overrides)
        
        # Create config file
        config_path = self.results_dir / f"{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        return str(config_path)
    
    def run_component_test(self, config_name: str, **component_overrides) -> Dict[str, Any]:
        """Run a single component test using production workflow"""
        print(f"\n🚀 Testing: {config_name}")
        print("=" * 60)
        
        # Create config file
        config_path = self.create_test_config(config_name, **component_overrides)
        
        # Run production training with this config
        start_time = time.time()
        try:
            success = train_celestial_pgat_production(config_path)
            training_time = time.time() - start_time
            
            if success:
                # Load results from the production training
                results_pattern = Path("./checkpoints/*/production_results.json")
                results_files = list(Path(".").glob("checkpoints/*/production_results.json"))
                
                if results_files:
                    # Get the most recent results file
                    latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_results, 'r') as f:
                        production_results = json.load(f)
                    
                    # Extract key metrics
                    result = {
                        'status': 'success',
                        'config_name': config_name,
                        'final_val_loss': production_results.get('best_val_loss', float('inf')),
                        'final_rmse': production_results.get('overall_metrics', {}).get('rmse', float('inf')),
                        'total_params': production_results.get('total_parameters', 0),
                        'training_time': training_time,
                        'train_losses': production_results.get('train_losses', []),
                        'val_losses': production_results.get('val_losses', []),
                        'overall_metrics': production_results.get('overall_metrics', {}),
                        'config_path': config_path
                    }
                    
                    print(f"✅ SUCCESS: {config_name}")
                    print(f"   • Final val loss: {result['final_val_loss']:.6f}")
                    print(f"   • Final RMSE: {result['final_rmse']:.6f}")
                    print(f"   • Parameters: {result['total_params']:,}")
                    print(f"   • Training time: {training_time:.1f}s")
                    
                else:
                    result = {
                        'status': 'success_no_results',
                        'config_name': config_name,
                        'training_time': training_time,
                        'config_path': config_path
                    }
                    print(f"⚠️  SUCCESS but no results file found: {config_name}")
            else:
                result = {
                    'status': 'failed',
                    'config_name': config_name,
                    'training_time': training_time,
                    'config_path': config_path,
                    'error': 'Production training returned False'
                }
                print(f"❌ FAILED: {config_name}")
        
        except Exception as e:
            training_time = time.time() - start_time
            result = {
                'status': 'error',
                'config_name': config_name,
                'training_time': training_time,
                'config_path': config_path,
                'error': str(e)
            }
            print(f"💥 ERROR: {config_name} - {e}")
        
        return result 
   
    def run_context_fusion_tests(self) -> Dict[str, Any]:
        """🌟 Test Multi-Scale Context Fusion modes systematically"""
        print("\n🌟 CONTEXT FUSION MODE TESTS")
        print("=" * 60)
        print("Testing: simple → gated → attention → multi_scale")
        
        context_configs = [
            ("Context_Simple", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "simple",
                "enable_context_diagnostics": True,
            }),
            ("Context_Gated", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "gated",
                "context_fusion_dropout": 0.1,
                "enable_context_diagnostics": True,
            }),
            ("Context_Attention", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "attention",
                "context_fusion_dropout": 0.1,
                "enable_context_diagnostics": True,
            }),
            ("Context_MultiScale", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "multi_scale",
                "short_term_kernel_size": 3,
                "medium_term_kernel_size": 15,
                "long_term_kernel_size": 0,  # Global
                "context_fusion_dropout": 0.1,
                "enable_context_diagnostics": True,
            }),
        ]
        
        results = {}
        for config_name, component_overrides in context_configs:
            result = self.run_component_test(config_name, **component_overrides)
            results[config_name] = result
            self.save_results({'context_fusion': results}, 'context_fusion_results.json')
            time.sleep(2)
        
        return results
    
    def run_stochastic_component_tests(self) -> Dict[str, Any]:
        """🎲 Test Stochastic Learning and Control components"""
        print("\n🎲 STOCHASTIC COMPONENT TESTS")
        print("=" * 60)
        print("Testing: none → learner → control → both")
        
        stochastic_configs = [
            ("Stochastic_None", {
                "use_stochastic_learner": False,
                "use_stochastic_control": False,
            }),
            ("Stochastic_Learner", {
                "use_stochastic_learner": True,
                "use_stochastic_control": False,
            }),
            ("Stochastic_Control", {
                "use_stochastic_learner": False,
                "use_stochastic_control": True,
                "stochastic_temperature_start": 1.0,
                "stochastic_temperature_end": 0.1,
                "stochastic_decay_steps": 500,
            }),
            ("Stochastic_Both", {
                "use_stochastic_learner": True,
                "use_stochastic_control": True,
                "stochastic_temperature_start": 1.0,
                "stochastic_temperature_end": 0.1,
                "stochastic_decay_steps": 500,
            }),
        ]
        
        results = {}
        for config_name, component_overrides in stochastic_configs:
            result = self.run_component_test(config_name, **component_overrides)
            results[config_name] = result
            self.save_results({'stochastic': results}, 'stochastic_results.json')
            time.sleep(2)
        
        return results
    
    def run_graph_combiner_tests(self) -> Dict[str, Any]:
        """🔗 Test Advanced Graph Combiner architectures"""
        print("\n🔗 GRAPH COMBINER TESTS")
        print("=" * 60)
        print("Testing: standard → gated → petri_net")
        
        graph_configs = [
            ("Graph_Standard", {
                "use_petri_net_combiner": False,
                "use_gated_graph_combiner": False,
                # Uses CelestialGraphCombiner (standard)
            }),
            ("Graph_Gated", {
                "use_petri_net_combiner": False,
                "use_gated_graph_combiner": True,
                # Uses GatedGraphCombiner
            }),
            ("Graph_PetriNet", {
                "use_petri_net_combiner": True,
                "use_gated_graph_combiner": False,
                "num_message_passing_steps": 2,
                "edge_feature_dim": 8,
                "use_temporal_attention": True,
                "use_spatial_attention": True,
                # Uses CelestialPetriNetCombiner (most advanced)
            }),
        ]
        
        results = {}
        for config_name, component_overrides in graph_configs:
            result = self.run_component_test(config_name, **component_overrides)
            results[config_name] = result
            self.save_results({'graph_combiners': results}, 'graph_combiner_results.json')
            time.sleep(2)
        
        return results
    
    def run_decoder_enhancement_tests(self) -> Dict[str, Any]:
        """🎯 Test Enhanced Decoder architectures"""
        print("\n🎯 DECODER ENHANCEMENT TESTS")
        print("=" * 60)
        print("Testing: standard → mixture → mdn → autocorr")
        
        decoder_configs = [
            ("Decoder_Standard", {
                "use_mixture_decoder": False,
                "use_sequential_mixture_decoder": False,
                "enable_mdn_decoder": False,
                "use_target_autocorrelation": False,
            }),
            ("Decoder_Mixture", {
                "use_mixture_decoder": True,
                "use_sequential_mixture_decoder": True,
                "enable_mdn_decoder": False,
                "use_target_autocorrelation": False,
            }),
            ("Decoder_MDN", {
                "use_mixture_decoder": False,
                "use_sequential_mixture_decoder": False,
                "enable_mdn_decoder": True,
                "mdn_components": 5,
                "mdn_sigma_min": 0.001,
                "mdn_use_softplus": True,
                "use_target_autocorrelation": False,
            }),
            ("Decoder_Autocorr", {
                "use_mixture_decoder": False,
                "use_sequential_mixture_decoder": False,
                "enable_mdn_decoder": False,
                "use_target_autocorrelation": True,
                "target_autocorr_layers": 2,
            }),
            ("Decoder_Full", {
                "use_mixture_decoder": True,
                "use_sequential_mixture_decoder": True,
                "enable_mdn_decoder": True,
                "mdn_components": 5,
                "use_target_autocorrelation": True,
                "target_autocorr_layers": 2,
            }),
        ]
        
        results = {}
        for config_name, component_overrides in decoder_configs:
            result = self.run_component_test(config_name, **component_overrides)
            results[config_name] = result
            self.save_results({'decoders': results}, 'decoder_results.json')
            time.sleep(2)
        
        return results
    
    def run_progressive_component_tests(self) -> Dict[str, Any]:
        """🚀 Progressive component addition - build up from baseline"""
        print("\n🚀 PROGRESSIVE COMPONENT ADDITION TESTS")
        print("=" * 70)
        print("Building up from baseline to full advanced configuration")
        
        progressive_configs = [
            ("01_Baseline", {
                # Minimal configuration
                "use_multi_scale_context": False,
                "use_hierarchical_mapping": False,
                "use_stochastic_learner": False,
                "use_petri_net_combiner": False,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": False,
                "enable_mdn_decoder": False,
                "use_efficient_covariate_interaction": False,
                "use_dynamic_spatiotemporal_encoder": False,
            }),
            ("02_Plus_ContextFusion", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "multi_scale",
                "use_hierarchical_mapping": False,
                "use_stochastic_learner": False,
                "use_petri_net_combiner": False,
                "use_mixture_decoder": False,
                "enable_mdn_decoder": False,
            }),
            ("03_Plus_HierarchicalMapping", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "multi_scale",
                "use_hierarchical_mapping": True,
                "use_stochastic_learner": False,
                "use_petri_net_combiner": False,
                "use_mixture_decoder": False,
            }),
            ("04_Plus_StochasticLearning", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "multi_scale",
                "use_hierarchical_mapping": True,
                "use_stochastic_learner": True,
                "use_petri_net_combiner": False,
                "use_mixture_decoder": False,
            }),
            ("05_Plus_PetriNetCombiner", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "multi_scale",
                "use_hierarchical_mapping": True,
                "use_stochastic_learner": True,
                "use_petri_net_combiner": True,
                "use_mixture_decoder": False,
            }),
            ("06_Plus_EnhancedDecoder", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "multi_scale",
                "use_hierarchical_mapping": True,
                "use_stochastic_learner": True,
                "use_petri_net_combiner": True,
                "use_mixture_decoder": True,
                "enable_mdn_decoder": True,
            }),
            ("07_Full_Advanced", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "multi_scale",
                "use_hierarchical_mapping": True,
                "use_stochastic_learner": True,
                "use_stochastic_control": True,
                "use_petri_net_combiner": True,
                "use_mixture_decoder": True,
                "enable_mdn_decoder": True,
                "use_efficient_covariate_interaction": True,
                "use_dynamic_spatiotemporal_encoder": True,
            }),
        ]
        
        results = {}
        for config_name, component_overrides in progressive_configs:
            result = self.run_component_test(config_name, **component_overrides)
            results[config_name] = result
            self.save_results({'progressive': results}, 'progressive_results.json')
            time.sleep(2)
        
        return results
    
    def run_ablation_tests(self) -> Dict[str, Any]:
        """🔬 Ablation study - remove one advanced component at a time from full config"""
        print("\n🔬 ABLATION STUDY TESTS")
        print("=" * 60)
        print("Remove one advanced component at a time from full configuration")
        
        # Full advanced configuration as baseline
        full_config = {
            "use_multi_scale_context": True,
            "context_fusion_mode": "multi_scale",
            "use_hierarchical_mapping": True,
            "use_stochastic_learner": True,
            "use_stochastic_control": True,
            "use_petri_net_combiner": True,
            "use_mixture_decoder": True,
            "enable_mdn_decoder": True,
            "use_efficient_covariate_interaction": True,
            "use_dynamic_spatiotemporal_encoder": True,
            "use_target_autocorrelation": True,
            "use_celestial_target_attention": True,
        }
        
        ablation_configs = [
            ("Ablation_No_ContextFusion", {
                **full_config,
                "use_multi_scale_context": False,
            }),
            ("Ablation_No_HierarchicalMapping", {
                **full_config,
                "use_hierarchical_mapping": False,
            }),
            ("Ablation_No_StochasticLearning", {
                **full_config,
                "use_stochastic_learner": False,
                "use_stochastic_control": False,
            }),
            ("Ablation_No_PetriNetCombiner", {
                **full_config,
                "use_petri_net_combiner": False,
                "use_gated_graph_combiner": False,  # Fall back to standard
            }),
            ("Ablation_No_EnhancedDecoders", {
                **full_config,
                "use_mixture_decoder": False,
                "enable_mdn_decoder": False,
            }),
            ("Ablation_No_EfficientProcessing", {
                **full_config,
                "use_efficient_covariate_interaction": False,
            }),
            ("Ablation_No_DynamicEncoding", {
                **full_config,
                "use_dynamic_spatiotemporal_encoder": False,
            }),
            ("Ablation_No_CelestialAttention", {
                **full_config,
                "use_celestial_target_attention": False,
            }),
        ]
        
        results = {}
        for config_name, component_overrides in ablation_configs:
            result = self.run_component_test(config_name, **component_overrides)
            results[config_name] = result
            self.save_results({'ablation': results}, 'ablation_results.json')
            time.sleep(2)
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file"""
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to {filepath}")
    
    def generate_comprehensive_report(self, all_results: Dict[str, Dict[str, Any]]):
        """Generate a comprehensive comparison report for all advanced component tests"""
        report_lines = [
            "# 🚀 Celestial Enhanced PGAT - Advanced Component Analysis Report",
            "",
            "## 🎯 Test Configuration",
            "- **Model**: Celestial Enhanced PGAT Modular (Advanced Components)",
            "- **Workflow**: Exact same as `train_celestial_production.py`",
            "- **Data**: Real financial data with proper scaling",
            "- **Model Size**: d_model=256, n_heads=8, e_layers=4, d_layers=3",
            "- **Training**: 8 epochs for component convergence analysis",
            "- **Batch Size**: 4 (stable gradient estimates)",
            "",
            "## 🌟 Advanced Components Tested",
            "",
            "### 1. Multi-Scale Context Fusion",
            "- **Simple**: Basic additive fusion (baseline)",
            "- **Gated**: Learnable blending mechanism",
            "- **Attention**: Dynamic timestep weighting",
            "- **Multi-Scale**: Multi-temporal pooling (most advanced)",
            "",
            "### 2. Stochastic Learning & Control",
            "- **Stochastic Learner**: Probabilistic graph learning with KL regularization",
            "- **Stochastic Control**: Temperature-modulated noise injection",
            "",
            "### 3. Graph Combiners",
            "- **Standard**: CelestialGraphCombiner (baseline)",
            "- **Gated**: GatedGraphCombiner (learnable fusion)",
            "- **Petri Net**: CelestialPetriNetCombiner (most advanced)",
            "",
            "### 4. Enhanced Decoders",
            "- **Mixture**: Sequential Mixture Density Networks",
            "- **MDN**: Mixture Density Networks with uncertainty",
            "- **Autocorr**: Target Autocorrelation processing",
            "",
            "### 5. Other Advanced Features",
            "- **Hierarchical Mapping**: Temporal-spatial feature enhancement",
            "- **Efficient Covariate Interaction**: Partitioned processing",
            "- **Dynamic Spatiotemporal Encoding**: Time-varying graph processing",
            "- **Celestial-Target Attention**: Future deterministic conditioning",
            "",
        ]
        
        # Add results for each test category
        for test_category, results in all_results.items():
            if not results:
                continue
                
            report_lines.extend([
                f"## 📊 {test_category.replace('_', ' ').title()} Results",
                ""
            ])
            
            if results:
                report_lines.append("| Configuration | Val Loss | RMSE | Parameters | Training Time | Status |")
                report_lines.append("|---------------|----------|------|------------|---------------|--------|")
                
                for config_name, result in results.items():
                    if result['status'] == 'success':
                        val_loss = f"{result.get('final_val_loss', 'N/A'):.6f}" if isinstance(result.get('final_val_loss'), (int, float)) else "N/A"
                        rmse = f"{result.get('final_rmse', 'N/A'):.6f}" if isinstance(result.get('final_rmse'), (int, float)) else "N/A"
                        params = f"{result.get('total_params', 'N/A'):,}" if isinstance(result.get('total_params'), int) else "N/A"
                        time_str = f"{result.get('training_time', 0):.1f}s"
                        status = "✅ Success"
                    else:
                        val_loss = "Failed"
                        rmse = "Failed"
                        params = "Failed"
                        time_str = f"{result.get('training_time', 0):.1f}s"
                        status = f"❌ {result['status']}"
                    
                    report_lines.append(f"| {config_name} | {val_loss} | {rmse} | {params} | {time_str} | {status} |")
            
            report_lines.append("")
        
        # Find best configurations across all test categories
        all_successful = {}
        for test_category, results in all_results.items():
            for name, result in results.items():
                if result.get('status') == 'success' and 'final_val_loss' in result:
                    all_successful[f"{test_category}_{name}"] = result
        
        if all_successful:
            # Find best overall
            best_config = min(all_successful.items(), key=lambda x: x[1]['final_val_loss'])
            
            # Find best in each category
            category_bests = {}
            for test_category, results in all_results.items():
                successful_in_category = {name: result for name, result in results.items() 
                                        if result.get('status') == 'success' and 'final_val_loss' in result}
                if successful_in_category:
                    best_in_category = min(successful_in_category.items(), key=lambda x: x[1]['final_val_loss'])
                    category_bests[test_category] = best_in_category
            
            report_lines.extend([
                "## 🏆 Key Findings & Recommendations",
                "",
                f"### 🥇 Overall Best Configuration: {best_config[0]}",
                f"- **Validation Loss**: {best_config[1]['final_val_loss']:.6f}",
                f"- **RMSE**: {best_config[1]['final_rmse']:.6f}",
                f"- **Parameters**: {best_config[1]['total_params']:,}",
                f"- **Training Time**: {best_config[1]['training_time']:.1f}s",
                "",
                "### 🎯 Best in Each Category:",
                ""
            ])
            
            for category, (name, result) in category_bests.items():
                report_lines.append(f"- **{category.replace('_', ' ').title()}**: {name} (Val Loss: {result['final_val_loss']:.6f})")
            
            report_lines.extend([
                "",
                "### 📈 Component Impact Analysis:",
                "",
                "**Most Beneficial Components** (based on validation loss improvement):",
            ])
            
            # Analyze component impact (simplified)
            if 'context_fusion' in all_results:
                context_results = all_results['context_fusion']
                successful_context = {name: result for name, result in context_results.items() 
                                    if result.get('status') == 'success'}
                if len(successful_context) > 1:
                    best_context = min(successful_context.items(), key=lambda x: x[1]['final_val_loss'])
                    report_lines.append(f"- **Context Fusion**: {best_context[0]} performs best")
            
            if 'stochastic' in all_results:
                stoch_results = all_results['stochastic']
                successful_stoch = {name: result for name, result in stoch_results.items() 
                                  if result.get('status') == 'success'}
                if len(successful_stoch) > 1:
                    best_stoch = min(successful_stoch.items(), key=lambda x: x[1]['final_val_loss'])
                    report_lines.append(f"- **Stochastic Learning**: {best_stoch[0]} performs best")
        
        report_lines.extend([
            "",
            "## 🔬 Technical Notes",
            "",
            "- **Data Handling**: Uses proven production workflow with proper scaling",
            "- **Model Architecture**: Celestial Enhanced PGAT Modular with advanced components",
            "- **Training Stability**: All components tested with identical training conditions",
            "- **Component Isolation**: Each test isolates specific advanced features",
            "- **Convergence Analysis**: 8 epochs allows component differences to emerge",
            "",
            "## 🚀 Production Recommendations",
            "",
            "1. **For Maximum Performance**: Use the overall best configuration",
            "2. **For Balanced Performance/Speed**: Consider category-specific best components",
            "3. **For Production Deployment**: Validate best components on longer training runs",
            "4. **For Research**: Focus on components showing consistent improvements",
            ""
        ])
        
        # Save comprehensive report
        report_path = self.results_dir / 'advanced_component_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📋 Comprehensive Advanced Component Analysis Report saved to {report_path}")
        return '\n'.join(report_lines)
    
    def run_all_tests(self):
        """🚀 Run comprehensive advanced component testing suite"""
        print("🌟 CELESTIAL ENHANCED PGAT - ADVANCED COMPONENT TESTING SUITE")
        print("=" * 80)
        print("🎯 SYSTEMATIC EVALUATION OF ADVANCED COMPONENTS")
        print("Using the EXACT same workflow as train_celestial_production.py")
        print("This ensures proper data scaling and realistic performance comparison")
        print("=" * 80)
        
        all_results = {}
        
        # 1. Context Fusion Tests
        print("\n🔥 PHASE 1: Context Fusion Analysis")
        all_results['context_fusion'] = self.run_context_fusion_tests()
        
        # 2. Stochastic Component Tests  
        print("\n🔥 PHASE 2: Stochastic Learning Analysis")
        all_results['stochastic'] = self.run_stochastic_component_tests()
        
        # 3. Graph Combiner Tests
        print("\n🔥 PHASE 3: Graph Combiner Analysis")
        all_results['graph_combiners'] = self.run_graph_combiner_tests()
        
        # 4. Decoder Enhancement Tests
        print("\n🔥 PHASE 4: Decoder Enhancement Analysis")
        all_results['decoders'] = self.run_decoder_enhancement_tests()
        
        # 5. Progressive Component Tests
        print("\n🔥 PHASE 5: Progressive Component Addition")
        all_results['progressive'] = self.run_progressive_component_tests()
        
        # 6. Ablation Study
        print("\n🔥 PHASE 6: Ablation Study")
        all_results['ablation'] = self.run_ablation_tests()
        
        # Save comprehensive results
        final_results = {
            **all_results,
            'test_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_configuration': 'advanced_component_analysis',
                'model_version': 'Celestial_Enhanced_PGAT_Modular',
                'total_test_categories': len(all_results),
                'total_configurations_tested': sum(len(results) for results in all_results.values()),
            }
        }
        self.save_results(final_results, 'comprehensive_results.json')
        
        # Generate comprehensive analysis report
        report = self.generate_comprehensive_report(all_results)
        
        print("\n" + "=" * 80)
        print("🎉 ADVANCED COMPONENT TESTING SUITE COMPLETED")
        print("=" * 80)
        print("🏆 Key Achievements:")
        print("✅ Systematic evaluation of 8+ advanced components")
        print("✅ Production-grade data handling and scaling")
        print("✅ Comprehensive performance comparison")
        print("✅ Component interaction analysis")
        print("✅ Ablation study for component importance")
        print(f"📊 Total configurations tested: {final_results['test_metadata']['total_configurations_tested']}")
        print(f"📁 All results saved to: {self.results_dir}")
        print(f"📋 Detailed report: {self.results_dir}/advanced_component_analysis_report.md")
        
        return final_results


def main():
    """🚀 Main execution function for advanced component testing"""
    print("🌟 CELESTIAL ENHANCED PGAT - ADVANCED COMPONENT TESTING")
    print("🎯 Systematic evaluation of cutting-edge AI components")
    print("🔬 Production-grade testing with comprehensive analysis")
    
    # Create advanced component tester
    tester = AdvancedComponentTester()
    
    # Run comprehensive testing suite
    results = tester.run_all_tests()
    
    return results


if __name__ == "__main__":
    results = main()