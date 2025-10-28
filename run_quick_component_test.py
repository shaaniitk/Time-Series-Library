#!/usr/bin/env python3
"""
🚀 Quick Component Test with FIXED Configuration
Tests a few key advanced components with the corrected data loading
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

def create_quick_test_config(config_name: str, **component_overrides) -> str:
    """Create a quick test config with FIXED data loading"""
    
    # Base config with FIXED data splits
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
        'validation_length': 500,  # ~7% for validation (477 samples)
        'test_length': 500,        # ~7% for testing (477 samples)
        # Leaves ~6109 samples for training (748 batches with batch_size=8)
        
        # Model configuration - Optimized for quick testing
        'd_model': 128,   # Smaller for speed
        'n_heads': 8,     # 128 ÷ 8 = 16
        'e_layers': 2,    # Fewer layers for speed
        'd_layers': 1,    # Minimal decoder layers
        'd_ff': 256,      # Smaller feed-forward
        'dropout': 0.1,
        
        # Input/Output dimensions
        'enc_in': 113,
        'dec_in': 113,
        'c_out': 4,
        
        # Training configuration - FAST
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
        
        # Celestial system - Standard
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
        'num_input_waves': 113,
        'target_wave_indices': [0, 1, 2, 3],
        
        # Advanced component defaults
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
        'calendar_embedding_dim': 32,
        'reg_loss_weight': 0.0005,
        'loss': {'type': 'mse'},
        'save_checkpoints': False,
        'log_interval': 5,
        'enable_memory_diagnostics': False,
        'collect_diagnostics': False,
    }
    
    # Apply component overrides
    config = base_config.copy()
    config.update(component_overrides)
    
    # Create results directory
    results_dir = Path('results/quick_component_tests')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config file
    config_path = results_dir / f"{config_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return str(config_path)

def run_quick_component_tests():
    """Run a few key component tests with FIXED configuration"""
    
    print("🚀 QUICK COMPONENT TESTS - FIXED Configuration")
    print("=" * 60)
    print("Testing key advanced components with corrected data loading")
    
    # Key component tests
    test_configs = [
        ("Baseline", {
            "use_multi_scale_context": False,
            "use_hierarchical_mapping": False,
            "use_stochastic_learner": False,
            "use_petri_net_combiner": False,
            "use_mixture_decoder": False,
        }),
        ("MultiScale_Context", {
            "use_multi_scale_context": True,
            "context_fusion_mode": "multi_scale",
            "short_term_kernel_size": 3,
            "medium_term_kernel_size": 15,
            "long_term_kernel_size": 0,
        }),
        ("Stochastic_Learning", {
            "use_stochastic_learner": True,
            "use_stochastic_control": True,
        }),
        ("PetriNet_Combiner", {
            "use_petri_net_combiner": True,
            "num_message_passing_steps": 2,
            "edge_feature_dim": 8,
        }),
        ("Enhanced_Decoders", {
            "use_mixture_decoder": True,
            "enable_mdn_decoder": True,
            "mdn_components": 3,
        }),
    ]
    
    results = {}
    
    for config_name, component_overrides in test_configs:
        print(f"\n🧪 Testing: {config_name}")
        print("-" * 40)
        
        # Create config
        config_path = create_quick_test_config(config_name, **component_overrides)
        
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
                    latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_results, 'r') as f:
                        production_results = json.load(f)
                    
                    result = {
                        'status': 'success',
                        'final_val_loss': production_results.get('best_val_loss', float('inf')),
                        'final_rmse': production_results.get('overall_metrics', {}).get('rmse', float('inf')),
                        'total_params': production_results.get('total_parameters', 0),
                        'training_time': training_time,
                    }
                    
                    print(f"✅ SUCCESS: Val Loss = {result['final_val_loss']:.6f}, RMSE = {result['final_rmse']:.6f}")
                else:
                    result = {'status': 'success_no_results', 'training_time': training_time}
                    print(f"✅ SUCCESS (no results file)")
            else:
                result = {'status': 'failed', 'training_time': training_time}
                print(f"❌ FAILED")
        
        except Exception as e:
            training_time = time.time() - start_time
            result = {'status': 'error', 'error': str(e), 'training_time': training_time}
            print(f"💥 ERROR: {e}")
        
        results[config_name] = result
        
        # Brief pause
        time.sleep(2)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 QUICK COMPONENT TEST RESULTS")
    print("=" * 60)
    
    successful = [name for name, result in results.items() if result['status'] == 'success']
    failed = [name for name, result in results.items() if result['status'] != 'success']
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    if successful:
        print(f"   {', '.join(successful)}")
    
    if failed:
        print(f"❌ Failed: {len(failed)}")
        print(f"   {', '.join(failed)}")
    
    # Find best performing component
    successful_results = {name: result for name, result in results.items() 
                         if result['status'] == 'success' and 'final_val_loss' in result}
    
    if successful_results:
        best_config = min(successful_results.items(), key=lambda x: x[1]['final_val_loss'])
        print(f"\n🏆 Best Component: {best_config[0]}")
        print(f"   Val Loss: {best_config[1]['final_val_loss']:.6f}")
        print(f"   RMSE: {best_config[1]['final_rmse']:.6f}")
    
    return results

def main():
    """Run quick component tests"""
    print("🌟 CELESTIAL ENHANCED PGAT - QUICK COMPONENT TESTING")
    print("🔧 Using FIXED data loading configuration")
    
    results = run_quick_component_tests()
    
    # Save results
    results_dir = Path('results/quick_component_tests')
    with open(results_dir / 'quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {results_dir}")
    return results

if __name__ == "__main__":
    results = main()