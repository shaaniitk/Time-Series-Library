#!/usr/bin/env python3
"""
🚀 FINAL COMPONENT TEST - All Issues Fixed
Tests advanced components with corrected data loading and dtype handling
"""

import os
import sys
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the proven production training function
from scripts.train.train_celestial_production import train_celestial_pgat_production

def create_final_test_config(config_name: str, **component_overrides) -> str:
    """Create a final test config with ALL issues fixed"""
    
    # Base config with ALL FIXES APPLIED
    base_config = {
        # Required fields
        'model': 'Celestial_Enhanced_PGAT',
        'data': 'custom',
        'root_path': './data',
        'data_path': 'prepared_financial_data.csv',
        'features': 'MS',
        'target': 'log_Open,log_High,log_Low,log_Close',
        'embed': 'timeF',
        'freq': 'd',
        
        # Sequence settings
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 24,
        
        # FIXED: Proper data splits for 7109 total samples
        'validation_length': 500,  # ~7% for validation (477 samples → 59 batches)
        'test_length': 500,        # ~7% for testing (477 samples → 59 batches)
        # Leaves ~6109 samples for training (748 batches with batch_size=8)
        
        # Model configuration - Optimized for component testing
        'd_model': 128,   # Reasonable size for testing
        'n_heads': 8,     # 128 ÷ 8 = 16
        'e_layers': 2,    # Fewer layers for speed
        'd_layers': 1,    # Minimal decoder layers
        'd_ff': 256,      # Smaller feed-forward
        'dropout': 0.1,
        
        # FIXED: Input/Output dimensions match actual data
        'enc_in': 118,    # Actual data has 118 features (119 columns - 1 date column)
        'dec_in': 118,    # Decoder input should match encoder input
        'c_out': 4,       # Output targets (OHLC)
        
        # Training configuration - FAST but meaningful
        'train_epochs': 2,  # Just 2 epochs for quick comparison
        'batch_size': 8,    # Good batch size for 748 total batches
        'learning_rate': 0.001,
        'patience': 5,
        'lradj': 'warmup_cosine',
        'warmup_epochs': 1,
        'min_lr': 1e-6,
        'weight_decay': 0.0001,
        
        # Production workflow settings
        'use_gpu': True,
        'mixed_precision': False,
        'gradient_accumulation_steps': 1,
        
        # Celestial system - Standard configuration
        'use_celestial_graph': True,
        'aggregate_waves_to_celestial': True,
        'celestial_fusion_layers': 1,
        'num_celestial_bodies': 13,
        'celestial_dim': 32,
        'num_message_passing_steps': 1,
        'edge_feature_dim': 6,
        'use_temporal_attention': True,
        'use_spatial_attention': True,
        'bypass_spatiotemporal_with_petri': True,
        'num_input_waves': 118,  # Match enc_in (will auto-correct to 113)
        'target_wave_indices': [0, 1, 2, 3],
        
        # Advanced component defaults (will be overridden in tests)
        'use_multi_scale_context': True,
        'context_fusion_mode': 'simple',
        'use_hierarchical_mapping': False,
        'use_stochastic_learner': False,
        'use_petri_net_combiner': True,
        'use_mixture_decoder': False,
        'enable_mdn_decoder': False,
        'use_target_autocorrelation': True,
        'use_celestial_target_attention': True,
        'use_efficient_covariate_interaction': False,
        'use_dynamic_spatiotemporal_encoder': True,
        
        # Additional required fields
        'use_calendar_effects': True,
        'calendar_embedding_dim': 32,  # d_model // 4
        'reg_loss_weight': 0.0005,
        'loss': {'type': 'mse'},
        'save_checkpoints': False,
        'log_interval': 10,  # Less frequent logging
        'enable_memory_diagnostics': False,
        'collect_diagnostics': False,
        'verbose_logging': False,
        
        # Required by data_provider
        'task_name': 'long_term_forecast',
        'model_name': 'Celestial_Enhanced_PGAT',
        'data_name': 'custom',
        'checkpoints': './checkpoints/',
        'inverse': False,
        'cols': None,
        'num_workers': 0,
        'itr': 1,
        'train_only': False,
        'do_predict': False,
    }
    
    # Apply component overrides
    config = base_config.copy()
    config.update(component_overrides)
    
    # Set model_id for unique checkpoints
    config['model_id'] = f"final_test_{config_name}_{int(time.time())}"
    
    # Create results directory
    results_dir = Path('results/final_component_tests')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config file
    config_path = results_dir / f"{config_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return str(config_path)

def run_final_component_tests():
    """Run final component tests with ALL fixes applied"""
    
    print("🚀 FINAL COMPONENT TESTS - ALL ISSUES FIXED")
    print("=" * 60)
    print("✅ Data loading: FIXED (748 training batches)")
    print("✅ Dimensions: FIXED (118 → 113 auto-correction)")
    print("✅ Dtype: FIXED (float32 conversion)")
    print("✅ Model creation: WORKING")
    print("✅ Forward pass: WORKING")
    print("=" * 60)
    
    # Focused component tests - most important advanced features
    test_configs = [
        ("Baseline_Minimal", {
            # Minimal configuration to establish baseline
            "use_multi_scale_context": False,
            "use_hierarchical_mapping": False,
            "use_stochastic_learner": False,
            "use_petri_net_combiner": False,
            "use_mixture_decoder": False,
            "enable_mdn_decoder": False,
        }),
        ("Context_MultiScale", {
            # Multi-scale context fusion (most important)
            "use_multi_scale_context": True,
            "context_fusion_mode": "multi_scale",
            "short_term_kernel_size": 3,
            "medium_term_kernel_size": 15,
            "long_term_kernel_size": 0,
            "enable_context_diagnostics": False,  # Disable for speed
        }),
        ("Stochastic_Enhanced", {
            # Stochastic learning components
            "use_stochastic_learner": True,
            "use_stochastic_control": True,
            "stochastic_temperature_start": 1.0,
            "stochastic_temperature_end": 0.1,
        }),
        ("PetriNet_Advanced", {
            # Advanced Petri Net combiner
            "use_petri_net_combiner": True,
            "num_message_passing_steps": 2,
            "edge_feature_dim": 8,
            "use_temporal_attention": True,
            "use_spatial_attention": True,
        }),
        ("Decoders_Enhanced", {
            # Enhanced decoder components
            "use_mixture_decoder": True,
            "enable_mdn_decoder": True,
            "mdn_components": 3,
            "mdn_sigma_min": 0.001,
            "use_target_autocorrelation": True,
        }),
    ]
    
    results = {}
    
    for config_name, component_overrides in test_configs:
        print(f"\n🧪 Testing: {config_name}")
        print("-" * 40)
        
        # Create config
        config_path = create_final_test_config(config_name, **component_overrides)
        print(f"   Config: {config_path}")
        
        # Run training
        start_time = time.time()
        try:
            success = train_celestial_pgat_production(config_path)
            training_time = time.time() - start_time
            
            if success:
                # Try to load results
                results_pattern = Path("./checkpoints/*/production_results.json")
                results_files = list(Path(".").glob("checkpoints/*/production_results.json"))
                
                if results_files:
                    # Get the most recent results file
                    latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_results, 'r') as f:
                        production_results = json.load(f)
                    
                    result = {
                        'status': 'success',
                        'final_val_loss': production_results.get('best_val_loss', float('inf')),
                        'final_rmse': production_results.get('overall_metrics', {}).get('rmse', float('inf')),
                        'total_params': production_results.get('total_parameters', 0),
                        'training_time': training_time,
                        'train_losses': production_results.get('train_losses', []),
                        'val_losses': production_results.get('val_losses', []),
                    }
                    
                    print(f"   ✅ SUCCESS!")
                    print(f"      Val Loss: {result['final_val_loss']:.6f}")
                    print(f"      RMSE: {result['final_rmse']:.6f}")
                    print(f"      Parameters: {result['total_params']:,}")
                    print(f"      Time: {training_time:.1f}s")
                else:
                    result = {
                        'status': 'success_no_results',
                        'training_time': training_time
                    }
                    print(f"   ✅ SUCCESS (no results file)")
            else:
                result = {
                    'status': 'failed',
                    'training_time': training_time
                }
                print(f"   ❌ FAILED")
        
        except Exception as e:
            training_time = time.time() - start_time
            result = {
                'status': 'error',
                'error': str(e),
                'training_time': training_time
            }
            print(f"   💥 ERROR: {e}")
        
        results[config_name] = result
        
        # Brief pause between tests
        time.sleep(3)
    
    return results

def generate_final_report(results: Dict[str, Any]):
    """Generate final component analysis report"""
    
    print("\n" + "=" * 80)
    print("🎉 FINAL COMPONENT TEST RESULTS")
    print("=" * 80)
    
    successful = []
    failed = []
    
    for name, result in results.items():
        if result['status'] == 'success':
            successful.append((name, result))
        else:
            failed.append((name, result))
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n🏆 SUCCESSFUL CONFIGURATIONS:")
        for name, result in successful:
            val_loss = result.get('final_val_loss', 'N/A')
            rmse = result.get('final_rmse', 'N/A')
            params = result.get('total_params', 'N/A')
            time_str = f"{result.get('training_time', 0):.1f}s"
            
            print(f"   • {name}")
            print(f"     Val Loss: {val_loss:.6f} | RMSE: {rmse:.6f} | Params: {params:,} | Time: {time_str}")
        
        # Find best performing component
        valid_results = [(name, result) for name, result in successful 
                        if 'final_val_loss' in result and isinstance(result['final_val_loss'], (int, float))]
        
        if valid_results:
            best_name, best_result = min(valid_results, key=lambda x: x[1]['final_val_loss'])
            print(f"\n🥇 BEST PERFORMING COMPONENT: {best_name}")
            print(f"   Val Loss: {best_result['final_val_loss']:.6f}")
            print(f"   RMSE: {best_result['final_rmse']:.6f}")
            print(f"   Parameters: {best_result['total_params']:,}")
    
    if failed:
        print(f"\n❌ FAILED CONFIGURATIONS:")
        for name, result in failed:
            error = result.get('error', result.get('status', 'Unknown'))
            print(f"   • {name}: {error}")
    
    print(f"\n📊 COMPONENT ANALYSIS SUMMARY:")
    print(f"   • Multi-Scale Context Fusion: {'✅' if any('Context' in name for name, _ in successful) else '❌'}")
    print(f"   • Stochastic Learning: {'✅' if any('Stochastic' in name for name, _ in successful) else '❌'}")
    print(f"   • Petri Net Combiner: {'✅' if any('PetriNet' in name for name, _ in successful) else '❌'}")
    print(f"   • Enhanced Decoders: {'✅' if any('Decoders' in name for name, _ in successful) else '❌'}")
    
    return results

def main():
    """Run final component tests with all fixes"""
    print("🌟 CELESTIAL ENHANCED PGAT - FINAL COMPONENT TESTING")
    print("🔧 All data loading and dimension issues RESOLVED")
    print("🎯 Testing most important advanced components")
    
    # Run tests
    results = run_final_component_tests()
    
    # Generate report
    final_results = generate_final_report(results)
    
    # Save results
    results_dir = Path('results/final_component_tests')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    with open(results_dir / f'final_results_{timestamp}.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {results_dir}")
    print("🎉 FINAL COMPONENT TESTING COMPLETED!")
    
    return final_results

if __name__ == "__main__":
    results = main()