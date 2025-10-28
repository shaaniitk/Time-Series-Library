"""
Systematic Training Tests for Enhanced SOTA PGAT with REAL Financial Data
seq_len=750, pred_len=20 - Using actual 11,181 samples with 113 covariates + 4 targets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt

from Enhanced_SOTA_PGAT_Refactored import Enhanced_SOTA_PGAT
from component_configuration_manager import ComponentConfigurationManager


class RealDataSystematicTester:
    """Runs systematic training tests with real financial data"""
    
    def __init__(self, device='cpu', results_dir='results/real_data_systematic_tests'):
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_manager = ComponentConfigurationManager()
        self.test_results = {}
        
        # Real data configuration parameters
        self.seq_len = 750
        self.pred_len = 20
        self.enc_in = 117  # 4 targets + 113 covariates
        self.c_out = 4     # OHLC targets
        self.d_model = 128
        self.n_heads = 8
        self.num_wave_features = 113  # Celestial covariates
        
    def create_realistic_config(self, **component_overrides):
        """Create realistic configuration with component overrides"""
        class RealDataConfig:
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
        
        return RealDataConfig(
            self.seq_len, self.pred_len, self.enc_in, self.c_out, 
            self.d_model, self.n_heads, self.num_wave_features, 
            **component_overrides
        )
    
    def load_real_financial_data(self, config, train_split=0.8):
        """Load and prepare real financial dataset"""
        print(f"📊 Loading REAL financial dataset...")
        
        # Load the actual financial data
        try:
            df = pd.read_csv("data/prepared_financial_data.csv")
            print(f"   • Loaded data shape: {df.shape}")
            print(f"   • Total samples: {len(df)} (Expected: 11,181)")
        except FileNotFoundError:
            raise FileNotFoundError("Real financial data not found at data/prepared_financial_data.csv")
        
        # Prepare column ordering (targets first, then features)
        target_cols = ['log_Open', 'log_High', 'log_Low', 'log_Close']  # 4 targets
        feature_cols = [col for col in df.columns if col not in ['date'] + target_cols]
        
        print(f"   • Target columns (4): {target_cols}")
        print(f"   • Feature columns ({len(feature_cols)}): Expected 113, got {len(feature_cols)}")
        
        # Verify we have the expected dimensions
        if len(feature_cols) != 113:
            print(f"⚠️  Warning: Expected 113 features, got {len(feature_cols)}")
            print(f"   • First 10 features: {feature_cols[:10]}")
        
        # Combine all data (targets + features)
        all_cols = target_cols + feature_cols
        data = df[all_cols].values.astype(np.float32)
        
        print(f"   • Combined data shape: {data.shape}")
        print(f"   • Data range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"   • Data std: {data.std():.4f}")
        
        # Create sequences
        seq_len = config.seq_len
        pred_len = config.pred_len
        
        # Calculate how many sequences we can create
        max_sequences = len(data) - seq_len - pred_len + 1
        print(f"   • Sequence length: {seq_len}")
        print(f"   • Prediction length: {pred_len}")
        print(f"   • Max possible sequences: {max_sequences}")
        
        if max_sequences <= 0:
            raise ValueError(f"Not enough data for sequences. Need at least {seq_len + pred_len} samples, got {len(data)}")
        
        # Create sequences
        wave_sequences = []
        target_sequences = []
        
        # Use every 20th sequence to reduce memory usage while maintaining diversity
        step_size = max(1, max_sequences // 500)  # Limit to ~500 sequences max for testing
        sequence_indices = range(0, max_sequences, step_size)
        
        print(f"   • Using every {step_size}th sequence")
        print(f"   • Creating {len(sequence_indices)} sequences...")
        
        for i in sequence_indices:
            # Input sequence (all features: targets + covariates)
            wave_seq = data[i:i+seq_len]  # Shape: (seq_len, 117)
            
            # Target sequence (only first 4 columns = OHLC targets)
            target_seq = data[i+seq_len:i+seq_len+pred_len, :4]  # Shape: (pred_len, 4)
            
            wave_sequences.append(wave_seq)
            target_sequences.append(target_seq)
        
        # Convert to tensors
        wave_tensor = torch.tensor(np.array(wave_sequences), dtype=torch.float32)
        target_tensor = torch.tensor(np.array(target_sequences), dtype=torch.float32)
        
        print(f"   • Wave tensor shape: {wave_tensor.shape}")
        print(f"   • Target tensor shape: {target_tensor.shape}")
        print(f"   • Wave tensor range: [{wave_tensor.min():.4f}, {wave_tensor.max():.4f}]")
        print(f"   • Target tensor range: [{target_tensor.min():.4f}, {target_tensor.max():.4f}]")
        
        # Split into train/validation
        n_sequences = len(wave_sequences)
        n_train = int(n_sequences * train_split)
        
        # Training data
        train_wave = wave_tensor[:n_train]
        train_target = target_tensor[:n_train]
        train_dataset = TensorDataset(train_wave, train_target)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        # Validation data
        val_wave = wave_tensor[n_train:]
        val_target = target_tensor[n_train:]
        val_dataset = TensorDataset(val_wave, val_target)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        print(f"✅ REAL dataset prepared successfully")
        print(f"   • Training sequences: {len(train_dataset)}")
        print(f"   • Validation sequences: {len(val_dataset)}")
        print(f"   • Input features: {wave_tensor.shape[-1]} (4 targets + {wave_tensor.shape[-1]-4} covariates)")
        print(f"   • Target features: {target_tensor.shape[-1]} (OHLC)")
        
        return train_loader, val_loader, train_dataset, val_dataset
    
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
        
        # Load real dataset
        try:
            train_loader, val_loader, train_dataset, val_dataset = self.load_real_financial_data(config)
        except Exception as e:
            print(f"❌ Dataset loading failed: {e}")
            return {'status': 'failed', 'error': f'Dataset loading: {e}'}
        
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
                    targets = target_batch  # Already only OHLC
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
                        targets = target_batch  # Already only OHLC
                        
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
    
    def run_progressive_tests(self):
        """Run progressive component tests with real data"""
        print("🚀 REAL DATA PROGRESSIVE COMPONENT TESTS")
        print("=" * 70)
        print(f"📊 Configuration: seq_len={self.seq_len}, pred_len={self.pred_len}")
        print(f"   Using REAL financial data: 11,181 samples, 113 covariates + 4 targets")
        
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
                self.save_results(results, 'real_data_progressive_results.json')
                
            except Exception as e:
                print(f"❌ Failed to test {name}: {e}")
                results[f"progressive_{name.replace(' ', '_').lower()}"] = {
                    'status': 'failed', 
                    'error': str(e)
                }
        
        return results
    
    def run_ablation_tests(self):
        """Run ablation study with real data"""
        print("\n🚀 REAL DATA ABLATION STUDY TESTS")
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
                self.save_results(results, 'real_data_ablation_results.json')
                
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
    
    def generate_summary_report(self, all_results):
        """Generate comprehensive summary report"""
        report_lines = [
            "# Enhanced SOTA PGAT - REAL Financial Data Testing Report",
            f"**Configuration**: seq_len={self.seq_len}, pred_len={self.pred_len}",
            f"**Real Data**: 11,181 samples, 113 celestial covariates + 4 OHLC targets",
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
                "## Progressive Component Analysis (Real Data)",
                ""
            ])
            
            progressive_results = all_results['progressive']
            for name, result in progressive_results.items():
                if result.get('status') == 'success':
                    report_lines.append(f"- **{result['config_name']}**: {result['final_val_loss']:.4f} loss, {result['total_params']:,} params")
        
        # Save report
        report_path = self.results_dir / 'real_data_testing_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📋 Summary report saved to {report_path}")
        
        return '\n'.join(report_lines)
    
    def run_all_tests(self):
        """Run all systematic tests with real data"""
        print("🚀 STARTING COMPREHENSIVE REAL DATA TESTING SUITE")
        print("=" * 80)
        
        all_results = {}
        
        # Run progressive tests
        try:
            progressive_results = self.run_progressive_tests()
            all_results['progressive'] = progressive_results
        except Exception as e:
            print(f"❌ Progressive tests failed: {e}")
            all_results['progressive'] = {'error': str(e)}
        
        # Run ablation tests
        try:
            ablation_results = self.run_ablation_tests()
            all_results['ablation'] = ablation_results
        except Exception as e:
            print(f"❌ Ablation tests failed: {e}")
            all_results['ablation'] = {'error': str(e)}
        
        # Save all results
        self.save_results(all_results, 'real_data_all_results.json')
        
        # Generate summary report
        summary = self.generate_summary_report(all_results)
        
        print("\n" + "=" * 80)
        print("🎉 REAL DATA TESTING SUITE COMPLETED")
        print("=" * 80)
        print(summary)
        
        return all_results


def main():
    """Main execution function"""
    print("🌟 Enhanced SOTA PGAT - REAL Financial Data Systematic Testing")
    print("=" * 70)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Create tester
    tester = RealDataSystematicTester(device=device)
    
    # Run all tests
    results = tester.run_all_tests()
    
    print(f"\n✅ All results saved to: {tester.results_dir}")
    
    return results


if __name__ == "__main__":
    results = main()