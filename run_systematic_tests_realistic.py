"""
Systematic Training Tests for Enhanced SOTA PGAT with Realistic Configuration
seq_len=750, pred_len=20 - Production-ready testing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt

from Enhanced_SOTA_PGAT_Refactored import Enhanced_SOTA_PGAT
from component_configuration_manager import ComponentConfigurationManager


class RealisticSystematicTester:
    """Runs systematic training tests with realistic sequence lengths"""
    
    def __init__(self, device='cpu', results_dir='results/realistic_systematic_tests'):
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_manager = ComponentConfigurationManager()
        self.test_results = {}
        
        # Realistic configuration parameters
        self.seq_len = 750
        self.pred_len = 20
        self.enc_in = 118
        self.c_out = 4
        self.d_model = 128
        self.n_heads = 8
        self.num_wave_features = 114
        
    def create_realistic_config(self, **component_overrides):
        """Create realistic configuration with component overrides"""
        class RealisticConfig:
            def __init__(self, seq_len, pred_len, enc_in, c_out, d_model, n_heads, num_wave_features, **overrides):
                # Core parameters
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.enc_in = enc_in
                self.c_out = c_out
                self.d_model = d_model
                self.n_heads = n_heads
                self.dropout = 0.1
                
                # Component defaults (all enabled for realistic testing)
                self.use_multi_scale_patching = True
                self.use_hierarchical_mapper = True
                self.use_stochastic_learner = True
                self.use_gated_graph_combiner = True
                self.use_mixture_decoder = True
                
                # Enhanced parameters for long sequences
                self.num_wave_features = num_wave_features
                self.mdn_components = 3
                self.mixture_multivariate_mode = 'independent'
                self.num_wave_patch_latents = 128  # More latents for longer sequences
                self.num_target_patch_latents = 64
                
                # Training parameters
                self.learning_rate = 0.0001
                self.batch_size = 2  # Smaller batch for memory efficiency with long sequences
                
                # Apply overrides
                for key, value in overrides.items():
                    setattr(self, key, value)
        
        return RealisticConfig(
            self.seq_len, self.pred_len, self.enc_in, self.c_out, 
            self.d_model, self.n_heads, self.num_wave_features, 
            **component_overrides
        )
    
    def create_realistic_dataset(self, config, num_samples=100):
        """Create realistic financial time series dataset"""
        print(f"📊 Creating realistic financial dataset...")
        print(f"   • Samples: {num_samples}")
        print(f"   • Sequence length: {config.seq_len} (~3 years daily data)")
        print(f"   • Prediction length: {config.pred_len} (~1 month ahead)")
        
        # Time vectors
        t_input = torch.linspace(0, config.seq_len/252, config.seq_len)  # ~3 years in trading days
        t_target = torch.linspace(config.seq_len/252, (config.seq_len + config.pred_len)/252, config.pred_len)
        
        wave_data = []
        target_data = []
        
        for i in range(num_samples):
            # Create realistic market patterns with multiple time scales (NORMALIZED)
            
            # Long-term trend (3-year) - normalized to [-0.1, 0.1]
            long_trend = 0.1 * (2 * t_input / t_input.max() - 1)
            
            # Seasonal cycles (quarterly earnings, annual patterns)
            seasonal = 0.05 * torch.sin(2 * torch.pi * t_input * 4)  # Quarterly
            seasonal += 0.03 * torch.sin(2 * torch.pi * t_input * 1)  # Annual
            
            # Medium-term cycles (monthly, bi-monthly)
            medium_cycle = 0.04 * torch.sin(2 * torch.pi * t_input * 12)  # Monthly
            medium_cycle += 0.02 * torch.cos(2 * torch.pi * t_input * 6)   # Bi-monthly
            
            # Short-term volatility (weekly, daily)
            short_volatility = 0.03 * torch.sin(2 * torch.pi * t_input * 52)  # Weekly
            short_volatility += 0.01 * torch.randn_like(t_input)  # Daily noise
            
            # Combine all patterns - NORMALIZED around 0
            base_price = long_trend + seasonal + medium_cycle + short_volatility
            
            # Generate OHLC with realistic relationships - NORMALIZED
            daily_volatility = 0.01 + 0.005 * torch.abs(torch.randn_like(base_price))
            
            open_price = base_price + 0.002 * torch.randn_like(base_price)
            high_price = torch.max(open_price, base_price) + daily_volatility * torch.rand_like(base_price)
            low_price = torch.min(open_price, base_price) - daily_volatility * torch.rand_like(base_price)
            close_price = base_price + 0.001 * torch.randn_like(base_price)
            
            ohlc = torch.stack([open_price, high_price, low_price, close_price], dim=-1)
            
            # Generate celestial features with realistic patterns - NORMALIZED
            # Simulate planetary cycles, lunar phases, etc.
            celestial_features = []
            
            for j in range(114):  # 114 celestial features
                # Different celestial bodies have different cycle lengths
                if j < 20:  # Fast-moving planets (daily to weekly cycles)
                    cycle_freq = 50 + 200 * torch.rand(1)  # 50-250 day cycles
                elif j < 50:  # Medium planets (monthly to quarterly)
                    cycle_freq = 10 + 40 * torch.rand(1)   # 10-50 day cycles
                else:  # Slow planets and aspects (seasonal to multi-year)
                    cycle_freq = 1 + 10 * torch.rand(1)    # 1-10 day cycles
                
                phase = 2 * torch.pi * torch.rand(1)
                amplitude = 0.3 + 0.2 * torch.rand(1)  # Reduced amplitude: 0.3-0.5
                
                celestial_feature = amplitude * torch.sin(2 * torch.pi * t_input * cycle_freq / 365 + phase)
                celestial_feature += 0.05 * torch.randn_like(t_input)  # Reduced noise
                celestial_features.append(celestial_feature)
            
            celestial_tensor = torch.stack(celestial_features, dim=-1)
            
            # Combine OHLC + celestial features
            wave_sample = torch.cat([ohlc, celestial_tensor], dim=-1)
            
            # Create target (future OHLC + celestial) - NORMALIZED
            # Future trend continuation with some noise
            future_trend = 0.1 * (2 * t_target / t_input.max() - 1)  # Continue normalized trend
            future_seasonal = 0.05 * torch.sin(2 * torch.pi * t_target * 4)
            future_seasonal += 0.03 * torch.sin(2 * torch.pi * t_target * 1)
            future_medium = 0.04 * torch.sin(2 * torch.pi * t_target * 12)
            future_short = 0.01 * torch.randn_like(t_target)
            
            future_base = future_trend + future_seasonal + future_medium + future_short
            
            future_volatility = 0.01 + 0.005 * torch.abs(torch.randn_like(future_base))
            future_open = future_base + 0.002 * torch.randn_like(future_base)
            future_high = torch.max(future_open, future_base) + future_volatility * torch.rand_like(future_base)
            future_low = torch.min(future_open, future_base) - future_volatility * torch.rand_like(future_base)
            future_close = future_base + 0.001 * torch.randn_like(future_base)
            
            future_ohlc = torch.stack([future_open, future_high, future_low, future_close], dim=-1)
            
            # Future celestial features (predictable)
            future_celestial_features = []
            for j in range(114):
                if j < 20:
                    cycle_freq = 50 + 200 * torch.rand(1)
                elif j < 50:
                    cycle_freq = 10 + 40 * torch.rand(1)
                else:
                    cycle_freq = 1 + 10 * torch.rand(1)
                
                phase = 2 * torch.pi * torch.rand(1)
                amplitude = 0.5 + 0.5 * torch.rand(1)
                
                future_celestial = amplitude * torch.sin(2 * torch.pi * t_target * cycle_freq / 365 + phase)
                future_celestial += 0.05 * torch.randn_like(t_target)  # Less noise for future
                future_celestial_features.append(future_celestial)
            
            future_celestial_tensor = torch.stack(future_celestial_features, dim=-1)
            target_sample = torch.cat([future_ohlc, future_celestial_tensor], dim=-1)
            
            wave_data.append(wave_sample)
            target_data.append(target_sample)
        
        wave_tensor = torch.stack(wave_data)
        target_tensor = torch.stack(target_data)
        
        # Create dataset and dataloader
        dataset = TensorDataset(wave_tensor, target_tensor)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        print(f"✅ Dataset created successfully")
        print(f"   • Wave tensor: {wave_tensor.shape}")
        print(f"   • Target tensor: {target_tensor.shape}")
        
        return dataloader, dataset
    
    def train_model_config(self, config, config_name, epochs=5):
        """Train model with specific configuration"""
        print(f"\n🚀 Training {config_name}")
        print("-" * 60)
        
        # Create model
        try:
            start_time = time.time()
            model = Enhanced_SOTA_PGAT(config).to(self.device)
            creation_time = time.time() - start_time
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"✅ Model created successfully")
            print(f"   • Creation time: {creation_time:.2f}s")
            print(f"   • Parameters: {total_params:,} total, {trainable_params:,} trainable")
            print(f"   • Model size: ~{total_params * 4 / 1024**2:.1f} MB")
            
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            return {'status': 'failed', 'error': f'Model creation: {e}'}
        
        # Create dataset
        try:
            train_loader, train_dataset = self.create_realistic_dataset(config, num_samples=50)
            val_loader, val_dataset = self.create_realistic_dataset(config, num_samples=20)
        except Exception as e:
            print(f"❌ Dataset creation failed: {e}")
            return {'status': 'failed', 'error': f'Dataset creation: {e}'}
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Training metrics
        train_losses = []
        val_losses = []
        training_times = []
        
        # Print component status
        try:
            config_info = model.get_enhanced_config_info()
            print(f"\n🔧 Component Status:")
            components = [
                ('use_multi_scale_patching', 'Multi-Scale Patching'),
                ('use_hierarchical_mapper', 'Hierarchical Mapper'),
                ('use_stochastic_learner', 'Stochastic Learner'),
                ('use_gated_graph_combiner', 'Graph Combiner'),
                ('use_mixture_decoder', 'Mixture Decoder')
            ]
            for key, name in components:
                status = "✅" if config_info.get(key, False) else "❌"
                print(f"   {status} {name}")
        except:
            pass
        
        # Training loop
        print(f"\n🏋️ Training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (wave_batch, target_batch) in enumerate(train_loader):
                wave_batch = wave_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                
                optimizer.zero_grad()
                
                try:
                    # Forward pass
                    output = model(wave_batch, target_batch)
                    
                    # Compute loss
                    targets = target_batch[:, :, :config.c_out]
                    if isinstance(output, tuple):
                        # MDN output
                        loss = model.loss(output, targets)
                    else:
                        # Regular output
                        loss = nn.MSELoss()(output, targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                except Exception as e:
                    print(f"⚠️  Training step failed: {e}")
                    continue
            
            avg_train_loss = train_loss / max(train_batches, 1)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for wave_batch, target_batch in val_loader:
                    wave_batch = wave_batch.to(self.device)
                    target_batch = target_batch.to(self.device)
                    
                    try:
                        output = model(wave_batch, target_batch)
                        targets = target_batch[:, :, :config.c_out]
                        
                        if isinstance(output, tuple):
                            loss = model.loss(output, targets)
                        else:
                            loss = nn.MSELoss()(output, targets)
                        
                        val_loss += loss.item()
                        val_batches += 1
                        
                    except Exception as e:
                        print(f"⚠️  Validation step failed: {e}")
                        continue
            
            avg_val_loss = val_loss / max(val_batches, 1)
            epoch_time = time.time() - epoch_start
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            training_times.append(epoch_time)
            
            print(f"   Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, time={epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        
        results = {
            'status': 'success',
            'config_name': config_name,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'training_times': training_times,
            'total_time': total_time,
            'final_train_loss': train_losses[-1] if train_losses else float('inf'),
            'final_val_loss': val_losses[-1] if val_losses else float('inf'),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'creation_time': creation_time,
            'epochs_completed': len(train_losses),
            'avg_epoch_time': total_time / max(len(train_losses), 1)
        }
        
        print(f"✅ Training completed")
        print(f"   • Final val loss: {results['final_val_loss']:.4f}")
        print(f"   • Total time: {total_time:.2f}s")
        print(f"   • Avg epoch time: {results['avg_epoch_time']:.2f}s")
        
        return results
    
    def run_realistic_progressive_tests(self):
        """Run progressive component tests with realistic sequences"""
        print("🚀 REALISTIC PROGRESSIVE COMPONENT TESTS")
        print("=" * 70)
        print(f"📊 Configuration: seq_len={self.seq_len}, pred_len={self.pred_len}")
        print(f"   This represents ~3 years → 1 month financial forecasting")
        
        # Progressive configurations
        progressive_configs = [
            ("Baseline (No Components)", {
                "use_multi_scale_patching": False,
                "use_hierarchical_mapper": False,
                "use_stochastic_learner": False,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": False
            }),
            ("+ Multi-Scale Patching", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": False,
                "use_stochastic_learner": False,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": False
            }),
            ("+ Hierarchical Mapper", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": False,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": False
            }),
            ("+ Stochastic Learner", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": False
            }),
            ("+ Graph Combiner", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": True,
                "use_mixture_decoder": False
            }),
            ("+ Mixture Decoder (Full)", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": True,
                "use_mixture_decoder": True
            })
        ]
        
        results = {}
        
        for name, component_overrides in progressive_configs:
            try:
                config = self.create_realistic_config(**component_overrides)
                result = self.train_model_config(config, name, epochs=3)
                results[f"progressive_{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}"] = result
                
                # Save intermediate results
                self.save_results(results, 'realistic_progressive_results.json')
                
            except Exception as e:
                print(f"❌ Failed to test {name}: {e}")
                results[f"progressive_{name.replace(' ', '_').lower()}"] = {
                    'status': 'failed', 
                    'error': str(e)
                }
        
        return results
    
    def run_realistic_ablation_tests(self):
        """Run ablation study with realistic sequences"""
        print("\n🚀 REALISTIC ABLATION STUDY TESTS")
        print("=" * 50)
        
        ablation_configs = [
            ("Without Multi-Scale Patching", {
                "use_multi_scale_patching": False,
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": True,
                "use_mixture_decoder": True
            }),
            ("Without Hierarchical Mapper", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": False,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": True,
                "use_mixture_decoder": True
            }),
            ("Without Stochastic Learner", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": False,
                "use_gated_graph_combiner": True,
                "use_mixture_decoder": True
            }),
            ("Without Graph Combiner", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": True
            }),
            ("Without Mixture Decoder", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": True,
                "use_mixture_decoder": False
            })
        ]
        
        results = {}
        
        for name, component_overrides in ablation_configs:
            try:
                config = self.create_realistic_config(**component_overrides)
                result = self.train_model_config(config, name, epochs=3)
                results[f"ablation_{name.replace(' ', '_').replace('Without_', '').lower()}"] = result
                
                # Save intermediate results
                self.save_results(results, 'realistic_ablation_results.json')
                
            except Exception as e:
                print(f"❌ Failed to test {name}: {e}")
                results[f"ablation_{name.replace(' ', '_').lower()}"] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def save_results(self, results, filename):
        """Save results to JSON file"""
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def create_performance_comparison(self, all_results):
        """Create performance comparison plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Enhanced SOTA PGAT - Realistic Sequence Performance Analysis', fontsize=16)
            
            # Extract successful results
            successful_results = {}
            for test_type, results in all_results.items():
                for name, result in results.items():
                    if result.get('status') == 'success':
                        successful_results[f"{test_type}_{name}"] = result
            
            if not successful_results:
                print("⚠️  No successful results to plot")
                return
            
            names = list(successful_results.keys())
            final_val_losses = [result['final_val_loss'] for result in successful_results.values()]
            total_params = [result['total_params'] for result in successful_results.values()]
            training_times = [result['total_time'] for result in successful_results.values()]
            
            # Plot 1: Final validation loss
            axes[0, 0].bar(range(len(names)), final_val_losses)
            axes[0, 0].set_title('Final Validation Loss')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_xticks(range(len(names)))
            axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
            
            # Plot 2: Model parameters
            axes[0, 1].bar(range(len(names)), [p/1e6 for p in total_params])
            axes[0, 1].set_title('Model Parameters (Millions)')
            axes[0, 1].set_ylabel('Parameters (M)')
            axes[0, 1].set_xticks(range(len(names)))
            axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
            
            # Plot 3: Training time
            axes[1, 0].bar(range(len(names)), training_times)
            axes[1, 0].set_title('Total Training Time')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].set_xticks(range(len(names)))
            axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
            
            # Plot 4: Performance vs Parameters
            axes[1, 1].scatter([p/1e6 for p in total_params], final_val_losses)
            axes[1, 1].set_xlabel('Parameters (Millions)')
            axes[1, 1].set_ylabel('Final Validation Loss')
            axes[1, 1].set_title('Performance vs Model Size')
            
            # Add labels to scatter plot
            for i, name in enumerate(names):
                axes[1, 1].annotate(name.split('_')[-1], 
                                  (total_params[i]/1e6, final_val_losses[i]),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.results_dir / 'realistic_performance_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance comparison saved to {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Failed to create plots: {e}")
    
    def generate_summary_report(self, all_results):
        """Generate comprehensive summary report"""
        report_lines = [
            "# Enhanced SOTA PGAT - Realistic Sequence Testing Report",
            f"**Configuration**: seq_len={self.seq_len}, pred_len={self.pred_len}",
            f"**Represents**: ~3 years → 1 month financial forecasting",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Find best performing configurations
        successful_results = {}
        for test_type, results in all_results.items():
            for name, result in results.items():
                if result.get('status') == 'success':
                    successful_results[f"{test_type}_{name}"] = result
        
        if successful_results:
            # Sort by validation loss
            sorted_results = sorted(successful_results.items(), 
                                  key=lambda x: x[1]['final_val_loss'])
            
            best_config = sorted_results[0]
            worst_config = sorted_results[-1]
            
            report_lines.extend([
                f"**🏆 Best Configuration**: {best_config[0]}",
                f"- Final validation loss: {best_config[1]['final_val_loss']:.4f}",
                f"- Parameters: {best_config[1]['total_params']:,}",
                f"- Training time: {best_config[1]['total_time']:.2f}s",
                "",
                f"**📊 Performance Range**: {best_config[1]['final_val_loss']:.4f} - {worst_config[1]['final_val_loss']:.4f}",
                f"**🎯 Improvement**: {((worst_config[1]['final_val_loss'] - best_config[1]['final_val_loss']) / worst_config[1]['final_val_loss'] * 100):.1f}% better than worst",
                ""
            ])
        
        # Progressive results analysis
        if 'progressive' in all_results:
            report_lines.extend([
                "## Progressive Component Analysis",
                ""
            ])
            
            progressive_results = all_results['progressive']
            for name, result in progressive_results.items():
                if result.get('status') == 'success':
                    report_lines.append(f"- **{result['config_name']}**: {result['final_val_loss']:.4f} loss, {result['total_params']:,} params")
        
        # Ablation study analysis
        if 'ablation' in all_results:
            report_lines.extend([
                "",
                "## Ablation Study Analysis",
                ""
            ])
            
            ablation_results = all_results['ablation']
            for name, result in ablation_results.items():
                if result.get('status') == 'success':
                    report_lines.append(f"- **{result['config_name']}**: {result['final_val_loss']:.4f} loss")
        
        # Component importance ranking
        if successful_results:
            report_lines.extend([
                "",
                "## Component Importance Ranking",
                "(Based on performance degradation when removed)",
                ""
            ])
            
            # This would require more sophisticated analysis
            # For now, just note the methodology
            report_lines.append("*Analysis requires comparison of ablation study results*")
        
        # Save report
        report_path = self.results_dir / 'realistic_testing_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📋 Summary report saved to {report_path}")
        
        return '\n'.join(report_lines)
    
    def run_all_tests(self):
        """Run all systematic tests"""
        print("🚀 STARTING COMPREHENSIVE REALISTIC TESTING SUITE")
        print("=" * 80)
        
        all_results = {}
        
        # Run progressive tests
        try:
            progressive_results = self.run_realistic_progressive_tests()
            all_results['progressive'] = progressive_results
        except Exception as e:
            print(f"❌ Progressive tests failed: {e}")
            all_results['progressive'] = {'error': str(e)}
        
        # Run ablation tests
        try:
            ablation_results = self.run_realistic_ablation_tests()
            all_results['ablation'] = ablation_results
        except Exception as e:
            print(f"❌ Ablation tests failed: {e}")
            all_results['ablation'] = {'error': str(e)}
        
        # Save all results
        self.save_results(all_results, 'realistic_all_results.json')
        
        # Generate visualizations
        self.create_performance_comparison(all_results)
        
        # Generate summary report
        summary = self.generate_summary_report(all_results)
        
        print("\n" + "=" * 80)
        print("🎉 REALISTIC TESTING SUITE COMPLETED")
        print("=" * 80)
        print(summary)
        
        return all_results


def main():
    """Main execution function"""
    print("🌟 Enhanced SOTA PGAT - Realistic Systematic Testing")
    print("=" * 60)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Create tester
    tester = RealisticSystematicTester(device=device)
    
    # Run all tests
    results = tester.run_all_tests()
    
    print(f"\n✅ All results saved to: {tester.results_dir}")
    
    return results


if __name__ == "__main__":
    results = main()