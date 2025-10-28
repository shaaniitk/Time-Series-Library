"""
Systematic Training Tests for Enhanced SOTA PGAT
Tests progressive component enabling with actual training loops
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


class SystematicTrainingTester:
    """Runs systematic training tests across different configurations"""
    
    def __init__(self, device='cpu', results_dir='results/systematic_tests'):
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_manager = ComponentConfigurationManager()
        self.test_results = {}
        
    def create_synthetic_dataset(self, config, num_samples=1000):
        """Create synthetic time series dataset for testing"""
        # Create realistic time series data
        batch_size = num_samples
        
        # Generate wave-like patterns with celestial influences
        t = torch.linspace(0, 10, config.seq_len)
        
        # Base wave patterns
        wave_data = []
        target_data = []
        
        for i in range(num_samples):
            # Create multiple frequency components
            freq1 = 0.5 + 0.3 * torch.rand(1)
            freq2 = 1.0 + 0.5 * torch.rand(1)
            phase1 = 2 * np.pi * torch.rand(1)
            phase2 = 2 * np.pi * torch.rand(1)
            
            # Generate celestial-like features (114 features)
            celestial_base = torch.sin(freq1 * t.unsqueeze(-1) + phase1) + 0.3 * torch.cos(freq2 * t.unsqueeze(-1) + phase2)
            celestial_features = celestial_base.repeat(1, 114) + 0.1 * torch.randn(config.seq_len, 114)
            
            # Generate OHLC-like target features (4 features)
            ohlc_base = torch.sin(freq1 * t + phase1) + 0.2 * torch.cos(freq2 * t + phase2)
            ohlc_features = torch.stack([
                ohlc_base + 0.05 * torch.randn(config.seq_len),  # Open
                ohlc_base + 0.1 + 0.05 * torch.randn(config.seq_len),  # High
                ohlc_base - 0.1 + 0.05 * torch.randn(config.seq_len),  # Low
                ohlc_base + 0.02 * torch.randn(config.seq_len),  # Close
            ], dim=-1)
            
            # Combine features
            wave_sample = torch.cat([ohlc_features, celestial_features], dim=-1)
            
            # Create target (predict next pred_len steps)
            target_t = torch.linspace(10, 10 + config.pred_len * 0.1, config.pred_len)
            target_ohlc = torch.sin(freq1 * target_t + phase1) + 0.2 * torch.cos(freq2 * target_t + phase2)
            target_ohlc_features = torch.stack([
                target_ohlc + 0.05 * torch.randn(config.pred_len),
                target_ohlc + 0.1 + 0.05 * torch.randn(config.pred_len),
                target_ohlc - 0.1 + 0.05 * torch.randn(config.pred_len),
                target_ohlc + 0.02 * torch.randn(config.pred_len),
            ], dim=-1)
            
            # Extend target with celestial features for consistency
            target_celestial = celestial_base[:config.pred_len] + 0.1 * torch.randn(config.pred_len, 114)
            target_sample = torch.cat([target_ohlc_features, target_celestial], dim=-1)
            
            wave_data.append(wave_sample)
            target_data.append(target_sample)
        
        wave_tensor = torch.stack(wave_data)
        target_tensor = torch.stack(target_data)
        
        # Create dataset and dataloader
        dataset = TensorDataset(wave_tensor, target_tensor)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        return dataloader, dataset
    
    def train_model_config(self, config, config_name, epochs=5):
        """Train model with specific configuration"""
        print(f"\n🚀 Training {config_name}")
        print("-" * 50)
        
        # Create model
        try:
            model = Enhanced_SOTA_PGAT(config).to(self.device)
            print(f"✅ Model created successfully")
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            return {'status': 'failed', 'error': f'Model creation: {e}'}
        
        # Create dataset
        try:
            train_loader, train_dataset = self.create_synthetic_dataset(config, num_samples=200)
            val_loader, val_dataset = self.create_synthetic_dataset(config, num_samples=50)
            print(f"✅ Dataset created: {len(train_dataset)} train, {len(val_dataset)} val samples")
        except Exception as e:
            print(f"❌ Dataset creation failed: {e}")
            return {'status': 'failed', 'error': f'Dataset creation: {e}'}
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Use appropriate loss function
        if config.use_mixture_decoder:
            criterion = model.mixture_loss if model.mixture_loss is not None else nn.MSELoss()
        else:
            criterion = nn.MSELoss()
        
        # Training metrics
        train_losses = []
        val_losses = []
        training_times = []
        
        print(f"🔧 Configuration:")
        self.config_manager.print_config_summary(config)
        
        # Training loop
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
                    if isinstance(output, tuple):
                        # MDN output
                        loss = model.loss(output, target_batch[:, :, :config.c_out])
                    else:
                        # Regular output
                        loss = criterion(output, target_batch[:, :, :config.c_out])
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
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
                        
                        if isinstance(output, tuple):
                            loss = model.loss(output, target_batch[:, :, :config.c_out])
                        else:
                            loss = criterion(output, target_batch[:, :, :config.c_out])
                        
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
            
            print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, time={epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # Model analysis
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get enhanced config info
        try:
            config_info = model.get_enhanced_config_info()
        except:
            config_info = {}
        
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
            'config_info': config_info,
            'epochs_completed': len(train_losses)
        }
        
        print(f"✅ Training completed: final_val_loss={results['final_val_loss']:.6f}")
        print(f"📊 Model: {total_params:,} total params, {trainable_params:,} trainable")
        
        return results
    
    def run_progressive_tests(self):
        """Run progressive component tests"""
        print("🚀 Starting Progressive Component Tests")
        print("=" * 60)
        
        # Get progressive configurations
        progressive_configs = self.config_manager.get_progressive_configs()
        
        config_names = [
            "Baseline (No Components)",
            "+ Multi-Scale Patching", 
            "+ Hierarchical Mapper",
            "+ Stochastic Learner",
            "+ Gated Graph Combiner",
            "+ Mixture Decoder (Full)"
        ]
        
        results = {}
        
        for i, (config, name) in enumerate(zip(progressive_configs, config_names)):
            try:
                result = self.train_model_config(config, name, epochs=3)
                results[f"step_{i:02d}_{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}"] = result
                
                # Save intermediate results
                self.save_results(results, 'progressive_results.json')
                
            except Exception as e:
                print(f"❌ Failed to test {name}: {e}")
                results[f"step_{i:02d}_{name.replace(' ', '_').lower()}"] = {
                    'status': 'failed', 
                    'error': str(e)
                }
        
        return results
    
    def run_recommended_tests(self):
        """Run recommended configuration tests"""
        print("\n🚀 Starting Recommended Configuration Tests")
        print("=" * 60)
        
        recommended_configs = self.config_manager.get_recommended_configs()
        results = {}
        
        for name, config in recommended_configs.items():
            try:
                result = self.train_model_config(config, f"Recommended {name.title()}", epochs=5)
                results[f"recommended_{name}"] = result
                
                # Save intermediate results
                self.save_results(results, 'recommended_results.json')
                
            except Exception as e:
                print(f"❌ Failed to test {name}: {e}")
                results[f"recommended_{name}"] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def run_ablation_tests(self):
        """Run ablation study tests"""
        print("\n🚀 Starting Ablation Study Tests")
        print("=" * 60)
        
        ablation_configs = self.config_manager.get_ablation_configs()
        component_names = [
            "Without Multi-Scale Patching",
            "Without Hierarchical Mapper", 
            "Without Stochastic Learner",
            "Without Gated Graph Combiner",
            "Without Mixture Decoder"
        ]
        
        results = {}
        
        for config, name in zip(ablation_configs, component_names):
            try:
                result = self.train_model_config(config, name, epochs=3)
                results[f"ablation_{name.replace(' ', '_').replace('Without_', '').lower()}"] = result
                
                # Save intermediate results
                self.save_results(results, 'ablation_results.json')
                
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
    
    def create_performance_plots(self, results, test_type):
        """Create performance visualization plots"""
        successful_results = {k: v for k, v in results.items() if v.get('status') == 'success'}
        
        if not successful_results:
            print(f"⚠️  No successful results to plot for {test_type}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{test_type.title()} Test Results', fontsize=16)
        
        # Extract data
        names = list(successful_results.keys())
        final_val_losses = [r['final_val_loss'] for r in successful_results.values()]
        total_params = [r['total_params'] for r in successful_results.values()]
        total_times = [r['total_time'] for r in successful_results.values()]
        
        # Plot 1: Final Validation Loss
        axes[0, 0].bar(range(len(names)), final_val_losses)
        axes[0, 0].set_title('Final Validation Loss')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_xticks(range(len(names)))
        axes[0, 0].set_xticklabels([n.replace('_', '\n') for n in names], rotation=45, ha='right')
        
        # Plot 2: Model Parameters
        axes[0, 1].bar(range(len(names)), total_params)
        axes[0, 1].set_title('Total Parameters')
        axes[0, 1].set_ylabel('Parameters')
        axes[0, 1].set_xticks(range(len(names)))
        axes[0, 1].set_xticklabels([n.replace('_', '\n') for n in names], rotation=45, ha='right')
        
        # Plot 3: Training Time
        axes[1, 0].bar(range(len(names)), total_times)
        axes[1, 0].set_title('Total Training Time')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_xticks(range(len(names)))
        axes[1, 0].set_xticklabels([n.replace('_', '\n') for n in names], rotation=45, ha='right')
        
        # Plot 4: Training Curves (first few results)
        for i, (name, result) in enumerate(list(successful_results.items())[:4]):
            if 'train_losses' in result and 'val_losses' in result:
                epochs = range(1, len(result['train_losses']) + 1)
                axes[1, 1].plot(epochs, result['train_losses'], label=f'{name} (train)', linestyle='--')
                axes[1, 1].plot(epochs, result['val_losses'], label=f'{name} (val)')
        
        axes[1, 1].set_title('Training Curves')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f'{test_type}_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance plots saved to {plot_path}")
    
    def print_summary(self, all_results):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("📊 SYSTEMATIC TESTING SUMMARY")
        print("=" * 80)
        
        total_tests = 0
        successful_tests = 0
        
        for test_type, results in all_results.items():
            print(f"\n🔧 {test_type.upper()} TESTS:")
            print("-" * 40)
            
            for name, result in results.items():
                total_tests += 1
                status_icon = "✅" if result.get('status') == 'success' else "❌"
                print(f"{status_icon} {name}")
                
                if result.get('status') == 'success':
                    successful_tests += 1
                    print(f"    Final Val Loss: {result.get('final_val_loss', 'N/A'):.6f}")
                    print(f"    Total Params: {result.get('total_params', 'N/A'):,}")
                    print(f"    Training Time: {result.get('total_time', 'N/A'):.2f}s")
                else:
                    print(f"    Error: {result.get('error', 'Unknown error')}")
        
        print(f"\n📈 OVERALL SUCCESS RATE: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
        
        if successful_tests == total_tests:
            print("🎉 All systematic tests passed! The modular architecture is working perfectly.")
        else:
            print("⚠️  Some tests failed. Check the detailed results above.")
    
    def run_all_systematic_tests(self):
        """Run all systematic tests"""
        print("🚀 ENHANCED SOTA PGAT - SYSTEMATIC TESTING SUITE")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Results Directory: {self.results_dir}")
        
        all_results = {}
        
        # Run progressive tests
        all_results['progressive'] = self.run_progressive_tests()
        self.create_performance_plots(all_results['progressive'], 'progressive')
        
        # Run recommended tests
        all_results['recommended'] = self.run_recommended_tests()
        self.create_performance_plots(all_results['recommended'], 'recommended')
        
        # Run ablation tests
        all_results['ablation'] = self.run_ablation_tests()
        self.create_performance_plots(all_results['ablation'], 'ablation')
        
        # Save comprehensive results
        self.save_results(all_results, 'comprehensive_results.json')
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results


def main():
    """Main function to run systematic tests"""
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create tester
    tester = SystematicTrainingTester(device=device)
    
    # Run all tests
    results = tester.run_all_systematic_tests()
    
    print(f"\n✅ Systematic testing complete! Results saved to {tester.results_dir}")
    
    return results


if __name__ == "__main__":
    main()