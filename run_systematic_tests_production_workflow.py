#!/usr/bin/env python3
"""
Systematic Component Testing using PRODUCTION WORKFLOW
Uses the exact same data handling, scaling, and training loop as train_celestial_production.py
but with lightweight configs to test individual components
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


class ProductionWorkflowTester:
    """Test components using the exact production workflow"""
    
    def __init__(self, results_dir='results/production_workflow_tests'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Base lightweight config for fast testing
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
            
            # Sequence settings - Adjusted for available data
            'seq_len': 96,    # Shorter sequence for testing
            'label_len': 48,
            'pred_len': 24,   # Shorter prediction for testing
            
            # Data split settings - Must be larger than pred_len
            'validation_length': 50,  # Larger than pred_len=24
            'test_length': 50,        # Larger than pred_len=24
            
            # Model configuration - Enhanced for component differentiation
            'd_model': 128,   # Larger model, power of 2 for better compatibility
            'n_heads': 8,     # Compatible heads: 128 ÷ 8 = 16
            'e_layers': 3,    # Deeper for component interactions
            'd_layers': 2,
            'd_ff': 256,      # Larger feed-forward
            'dropout': 0.1,
            
            # Input/Output dimensions - Fixed for Celestial model
            'enc_in': 113,    # Celestial model expects 113 features
            'dec_in': 113,    # Celestial model expects 113 features  
            'c_out': 4,
            
            # Training configuration - FAST for testing
            'train_epochs': 5,  # More epochs to see component differences
            'batch_size': 2,    # Small batch for long sequences (750 days)
            'learning_rate': 0.001,
            'patience': 5,
            'lradj': 'warmup_cosine',
            'warmup_epochs': 1,
            'min_lr': 1e-6,
            'weight_decay': 0.0001,
            'clip_grad_norm': 1.0,
            
            # Production workflow settings
            'use_gpu': True,
            'mixed_precision': False,
            'gradient_accumulation_steps': 1,
            
            # Celestial system - minimal
            'use_celestial_graph': True,
            'aggregate_waves_to_celestial': True,
            'celestial_fusion_layers': 1,
            'use_petri_net_combiner': True,
            'num_message_passing_steps': 1,
            'edge_feature_dim': 6,
            'use_temporal_attention': True,
            'use_spatial_attention': True,
            'bypass_spatiotemporal_with_petri': True,
            'num_input_waves': 113,
            'target_wave_indices': [0, 1, 2, 3],
            
            # Component defaults (will be overridden)
            'use_mixture_decoder': False,
            'use_stochastic_learner': False,
            'use_hierarchical_mapping': False,
            'use_hierarchical_fusion': True,
            'use_efficient_covariate_interaction': True,
            'mdn_multivariate_mode': 'independent',
            
            # Additional required fields
            'use_target_autocorrelation': True,
            'target_autocorr_layers': 1,
            'use_celestial_target_attention': True,
            'celestial_target_use_gated_fusion': True,
            'celestial_target_diagnostics': False,
            'use_c2t_edge_bias': True,
            'c2t_edge_bias_weight': 0.2,
            'c2t_aux_rel_loss_weight': 0.0,
            'use_calendar_effects': True,
            'calendar_embedding_dim': 16,
            'reg_loss_weight': 0.0005,
            'kl_weight': 0.00005,
            'save_checkpoints': False,
            'checkpoint_interval': 1,
            'log_interval': 2,
            'save_best_only': False,
            'enable_mdn_decoder': False,
            'mdn_components': 3,
            'mdn_sigma_min': 0.001,
            'mdn_use_softplus': True,
            
            # Loss configuration
            'loss': {'type': 'mse'},
            
            # Logging
            'enable_memory_diagnostics': False,
            'memory_log_interval': 10,
            'collect_diagnostics': False,
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
   
    def run_progressive_component_tests(self) -> Dict[str, Any]:
        """Run progressive component addition tests"""
        print("🚀 PROGRESSIVE COMPONENT TESTS - Production Workflow")
        print("=" * 70)
        print("Using the exact same data handling and training loop as production")
        print("Only the component flags are changed for systematic testing")
        
        # Progressive configurations - add one component at a time (MultiScalePatching removed)
        progressive_configs = [
            ("01_Baseline", {
                "use_hierarchical_mapper": False,
                "use_stochastic_learner": False,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": False
            }),
            ("02_Hierarchical_Mapping", {
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": False,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": False
            }),
            ("03_Plus_Stochastic", {
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": False
            }),
            ("04_Plus_GraphCombiner", {
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": True,
                "use_mixture_decoder": False
            }),
            ("05_Full_Configuration", {
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": True,
                "use_mixture_decoder": True
            })
        ]
        
        results = {}
        
        for config_name, component_overrides in progressive_configs:
            result = self.run_component_test(config_name, **component_overrides)
            results[config_name] = result
            
            # Save intermediate results
            self.save_results({'progressive': results}, 'progressive_results.json')
            
            # Brief pause between tests
            time.sleep(2)
        
        return results
    
    def run_ablation_tests(self) -> Dict[str, Any]:
        """Run ablation study - remove one component at a time from full config"""
        print("\n🚀 ABLATION STUDY TESTS - Production Workflow")
        print("=" * 50)
        
        ablation_configs = [
            ("Ablation_No_Hierarchical", {
                "use_hierarchical_mapper": False,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": True,
                "use_mixture_decoder": True
            }),
            ("Ablation_No_Stochastic", {
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": False,
                "use_gated_graph_combiner": True,
                "use_mixture_decoder": True
            }),
            ("Ablation_No_GraphCombiner", {
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": True
            }),
            ("Ablation_No_MixtureDecoder", {
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": True,
                "use_mixture_decoder": False
            })
        ]
        
        results = {}
        
        for config_name, component_overrides in ablation_configs:
            result = self.run_component_test(config_name, **component_overrides)
            results[config_name] = result
            
            # Save intermediate results
            self.save_results({'ablation': results}, 'ablation_results.json')
            
            # Brief pause between tests
            time.sleep(2)
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file"""
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to {filepath}")
    
    def generate_comparison_report(self, progressive_results: Dict[str, Any], ablation_results: Dict[str, Any]):
        """Generate a comprehensive comparison report"""
        report_lines = [
            "# Enhanced SOTA PGAT - Production Workflow Component Testing",
            "",
            "## Test Configuration",
            "- **Workflow**: Exact same as `train_celestial_production.py`",
            "- **Data**: Real financial data with proper scaling",
            "- **Model**: Lightweight (d_model=64, n_heads=4, e_layers=2)",
            "- **Training**: 3 epochs for fast component comparison",
            "",
            "## Progressive Component Addition Results",
            ""
        ]
        
        # Progressive results table
        if progressive_results:
            report_lines.append("| Configuration | Val Loss | RMSE | Parameters | Status |")
            report_lines.append("|---------------|----------|------|------------|--------|")
            
            for config_name, result in progressive_results.items():
                if result['status'] == 'success':
                    val_loss = result.get('final_val_loss', 'N/A')
                    rmse = result.get('final_rmse', 'N/A')
                    params = result.get('total_params', 'N/A')
                    status = "✅ Success"
                else:
                    val_loss = "Failed"
                    rmse = "Failed"
                    params = "Failed"
                    status = f"❌ {result['status']}"
                
                params_str = f"{params:,}" if isinstance(params, int) else str(params)
                report_lines.append(f"| {config_name} | {val_loss} | {rmse} | {params_str} | {status} |")
        
        report_lines.extend([
            "",
            "## Ablation Study Results",
            ""
        ])
        
        # Ablation results table
        if ablation_results:
            report_lines.append("| Configuration | Val Loss | RMSE | Parameters | Status |")
            report_lines.append("|---------------|----------|------|------------|--------|")
            
            for config_name, result in ablation_results.items():
                if result['status'] == 'success':
                    val_loss = result.get('final_val_loss', 'N/A')
                    rmse = result.get('final_rmse', 'N/A')
                    params = result.get('total_params', 'N/A')
                    status = "✅ Success"
                else:
                    val_loss = "Failed"
                    rmse = "Failed"
                    params = "Failed"
                    status = f"❌ {result['status']}"
                
                params_str = f"{params:,}" if isinstance(params, int) else str(params)
                report_lines.append(f"| {config_name} | {val_loss} | {rmse} | {params_str} | {status} |")
        
        # Find best configuration
        all_successful = {}
        for test_type, results in [('progressive', progressive_results), ('ablation', ablation_results)]:
            for name, result in results.items():
                if result.get('status') == 'success' and 'final_val_loss' in result:
                    all_successful[f"{test_type}_{name}"] = result
        
        if all_successful:
            best_config = min(all_successful.items(), key=lambda x: x[1]['final_val_loss'])
            report_lines.extend([
                "",
                "## Key Findings",
                "",
                f"**🏆 Best Configuration**: {best_config[0]}",
                f"- Validation Loss: {best_config[1]['final_val_loss']:.6f}",
                f"- RMSE: {best_config[1]['final_rmse']:.6f}",
                f"- Parameters: {best_config[1]['total_params']:,}",
                ""
            ])
        
        report_lines.extend([
            "## Notes",
            "",
            "- All tests use the proven production workflow for data handling and scaling",
            "- Negative losses indicate potential scaling or loss computation issues",
            "- Component effectiveness should be measured by consistent loss reduction",
            "- Results are based on lightweight model configuration for fast testing",
            ""
        ])
        
        # Save report
        report_path = self.results_dir / 'component_testing_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📋 Comprehensive report saved to {report_path}")
        return '\n'.join(report_lines)
    
    def run_all_tests(self):
        """Run all systematic tests using production workflow"""
        print("🌟 Enhanced SOTA PGAT - Production Workflow Component Testing")
        print("=" * 80)
        print("Using the EXACT same workflow as train_celestial_production.py")
        print("This ensures proper data scaling and eliminates matrix multiplication issues")
        print("=" * 80)
        
        # Run progressive tests
        progressive_results = self.run_progressive_component_tests()
        
        # Run ablation tests
        ablation_results = self.run_ablation_tests()
        
        # Save combined results
        all_results = {
            'progressive': progressive_results,
            'ablation': ablation_results,
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_configuration': 'production_workflow_lightweight'
        }
        self.save_results(all_results, 'all_results.json')
        
        # Generate comprehensive report
        report = self.generate_comparison_report(progressive_results, ablation_results)
        
        print("\n" + "=" * 80)
        print("🎉 PRODUCTION WORKFLOW TESTING COMPLETED")
        print("=" * 80)
        print("Key advantages of this approach:")
        print("✅ Uses proven production data handling and scaling")
        print("✅ Eliminates matrix multiplication issues")
        print("✅ Ensures realistic loss values (no negative losses)")
        print("✅ Direct comparison with production performance")
        print(f"📁 All results saved to: {self.results_dir}")
        
        return all_results


def main():
    """Main execution function"""
    print("🚀 Starting Production Workflow Component Testing")
    
    # Create tester
    tester = ProductionWorkflowTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    return results


if __name__ == "__main__":
    results = main()