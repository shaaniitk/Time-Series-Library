"""
Enhanced SOTA PGAT with MoE - Comprehensive Example

This example demonstrates the complete enhanced SOTA PGAT model with:
- Mixture of Experts for specialized pattern modeling
- Curriculum learning for progressive training
- Memory optimization techniques
- Hierarchical graph attention
- Advanced uncertainty quantification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import the enhanced model and components
from models.Enhanced_SOTA_Temporal_PGAT_MoE import Enhanced_SOTA_Temporal_PGAT_MoE
from layers.modular.training.curriculum_learning import create_adaptive_curriculum
from layers.modular.training.memory_optimization import MemoryOptimizedTrainer
from layers.modular.experts.registry import expert_registry


class EnhancedSOTAConfig:
    """Enhanced configuration for SOTA PGAT with MoE."""
    
    def __init__(self):
        # Basic model parameters
        self.d_model = 512
        self.n_heads = 8
        self.dropout = 0.1
        
        # Data dimensions
        self.seq_len = 96
        self.pred_len = 24
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.features = 'M'
        
        # MoE parameters
        self.temporal_top_k = 2
        self.spatial_top_k = 2
        self.uncertainty_top_k = 1
        
        # Enhanced components
        self.use_mixture_density = True
        self.use_autocorr_attention = True
        self.use_dynamic_edge_weights = True
        self.use_adaptive_temporal = True
        
        # MDN parameters
        self.mdn_components = 3
        
        # Graph parameters
        self.hierarchy_levels = 3
        self.sparsity_ratio = 0.1
        self.max_eigenvectors = 16
        
        # Curriculum learning parameters
        self.use_sequence_curriculum = True
        self.use_complexity_curriculum = True
        self.min_seq_len = 24
        self.max_seq_len = 96
        self.min_experts = 1
        self.max_experts = 4
        
        # Memory optimization parameters
        self.use_mixed_precision = True
        self.use_gradient_checkpointing = True
        self.batch_size = 16
        self.max_batch_size = 64
        self.memory_threshold = 0.8


def generate_complex_synthetic_data(batch_size: int = 32, seq_len: int = 96, 
                                  pred_len: int = 24, num_features: int = 7) -> tuple:
    """Generate complex synthetic time series data with multiple patterns."""
    
    # Generate wave data with multiple patterns
    t_wave = np.linspace(0, 4*np.pi, seq_len)
    wave_data = []
    
    for _ in range(batch_size):
        wave_sample = np.zeros((seq_len, num_features))
        
        for f in range(num_features):
            # Base pattern
            freq = 0.5 + f * 0.2
            phase = f * np.pi / 4
            
            # Seasonal component
            seasonal = np.sin(freq * t_wave + phase)
            
            # Trend component
            trend = 0.01 * t_wave * (f + 1)
            
            # Volatility clustering
            volatility = 0.1 + 0.05 * np.abs(np.sin(2 * freq * t_wave))
            noise = np.random.normal(0, volatility, seq_len)
            
            # Regime changes (random breakpoints)
            if np.random.random() > 0.7:  # 30% chance of regime change
                breakpoint = np.random.randint(seq_len // 3, 2 * seq_len // 3)
                regime_shift = np.random.normal(0, 0.5)
                trend[breakpoint:] += regime_shift
            
            wave_sample[:, f] = trend + seasonal + noise
        
        wave_data.append(wave_sample)
    
    # Generate target data continuing the patterns
    t_target = np.linspace(4*np.pi, 4*np.pi + np.pi, pred_len)
    target_data = []
    
    for i in range(batch_size):
        target_sample = np.zeros((pred_len, num_features))
        
        for f in range(num_features):
            freq = 0.5 + f * 0.2
            phase = f * np.pi / 4
            
            # Continue patterns from wave data
            last_trend = 0.01 * 4*np.pi * (f + 1)
            trend = last_trend + 0.01 * (t_target - 4*np.pi) * (f + 1)
            seasonal = np.sin(freq * t_target + phase)
            
            volatility = 0.1 + 0.05 * np.abs(np.sin(2 * freq * t_target))
            noise = np.random.normal(0, volatility, pred_len)
            
            target_sample[:, f] = trend + seasonal + noise
        
        target_data.append(target_sample)
    
    # Create graph adjacency matrix
    total_nodes = seq_len + pred_len
    adjacency = np.ones((total_nodes, total_nodes)) - np.eye(total_nodes)
    
    graph_data = []
    for _ in range(batch_size):
        graph_data.append(adjacency)
    
    return (
        torch.FloatTensor(wave_data),
        torch.FloatTensor(target_data),
        torch.FloatTensor(graph_data)
    )


def train_enhanced_model(model: nn.Module, train_loader: DataLoader, 
                        config: EnhancedSOTAConfig, num_epochs: int = 10) -> dict:
    """Train the enhanced SOTA PGAT model with all optimizations."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Create curriculum scheduler
    curriculum_config = {
        'use_sequence_curriculum': config.use_sequence_curriculum,
        'use_complexity_curriculum': config.use_complexity_curriculum,
        'min_seq_len': config.min_seq_len,
        'max_seq_len': config.max_seq_len,
        'min_experts': config.min_experts,
        'max_experts': config.max_experts
    }
    
    curriculum_scheduler = create_adaptive_curriculum(num_epochs, curriculum_config)
    
    # Create memory-optimized trainer
    trainer_config = {
        'use_mixed_precision': config.use_mixed_precision,
        'use_gradient_checkpointing': config.use_gradient_checkpointing,
        'batch_size': config.batch_size,
        'max_batch_size': config.max_batch_size,
        'memory_threshold': config.memory_threshold,
        'device': device
    }
    
    memory_trainer = MemoryOptimizedTrainer(model, trainer_config)
    
    # Training loop
    training_history = {
        'losses': [],
        'curriculum_params': [],
        'expert_utilization': [],
        'memory_usage': []
    }
    
    print(f"Training Enhanced SOTA PGAT with MoE for {num_epochs} epochs...")
    print(f"Available experts: {expert_registry.list_experts()}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        # Get curriculum parameters for this epoch
        curriculum_params = curriculum_scheduler.step(epoch)
        training_history['curriculum_params'].append(curriculum_params)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Curriculum params: {curriculum_params}")
        
        for batch_idx, (wave_window, target_window, graph) in enumerate(train_loader):
            wave_window = wave_window.to(device)
            target_window = target_window.to(device)
            graph = graph.to(device)
            
            # Prepare batch for trainer
            batch_data = {
                'wave_window': wave_window,
                'target_window': target_window,
                'graph': graph,
                'targets': target_window[:, :, 0:1]  # Use first feature as target
            }
            
            # Training step with memory optimization
            step_result = memory_trainer.train_step(batch_data, optimizer)
            
            if 'oom_occurred' in step_result:
                print(f"OOM occurred, adjusting batch size to {step_result['new_batch_size']}")
                continue
            
            epoch_losses.append(step_result['loss'])
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}, Loss: {step_result['loss']:.6f}")
        
        # Record epoch statistics
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            training_history['losses'].append(avg_loss)
            
            # Get expert utilization
            expert_util = model.get_expert_utilization()
            training_history['expert_utilization'].append(expert_util)
            
            # Get memory usage
            memory_summary = memory_trainer.get_memory_summary()
            training_history['memory_usage'].append(memory_summary)
            
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.6f}")
            print(f"Expert utilization: {expert_util}")
            print(f"Memory usage: {memory_summary['peak_memory']}")
    
    return training_history


def evaluate_enhanced_model(model: nn.Module, test_loader: DataLoader, 
                           device: str = 'cpu') -> dict:
    """Evaluate the enhanced model with uncertainty quantification."""
    
    model.eval()
    predictions = []
    targets = []
    uncertainties = []
    expert_outputs = []
    
    print("Evaluating enhanced model...")
    
    with torch.no_grad():
        for wave_window, target_window, graph in test_loader:
            wave_window = wave_window.to(device)
            target_window = target_window.to(device)
            graph = graph.to(device)
            
            try:
                # Forward pass
                result = model(wave_window, target_window, graph)
                
                if isinstance(result, tuple):
                    output, moe_info = result
                    expert_outputs.append(moe_info)
                else:
                    output = result
                
                # Handle different output types
                if isinstance(output, tuple) and len(output) == 3:
                    # Mixture density output
                    means, log_stds, log_weights = output
                    
                    # Get prediction summary
                    if hasattr(model.decoder, 'prediction_summary'):
                        pred_mean, pred_std = model.decoder.prediction_summary((means, log_stds, log_weights))
                    else:
                        # Fallback
                        import torch.nn.functional as F
                        weights = F.softmax(log_weights, dim=-1)
                        pred_mean = torch.sum(weights * means, dim=-1)
                        pred_std = torch.exp(log_stds).mean(dim=-1)
                    
                    predictions.append(pred_mean.cpu())
                    uncertainties.append(pred_std.cpu())
                else:
                    # Standard output
                    if output.dim() > 2:
                        output = output.squeeze(-1)
                    predictions.append(output.cpu())
                    uncertainties.append(torch.zeros_like(output).cpu())
                
                targets.append(target_window[:, :, 0].cpu())
                
            except Exception as e:
                print(f"Error in evaluation: {str(e)}")
                continue
    
    if predictions:
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        uncertainties = torch.cat(uncertainties, dim=0)
        
        # Calculate metrics
        mse = mean_squared_error(targets.numpy(), predictions.numpy())
        mae = mean_absolute_error(targets.numpy(), predictions.numpy())
        
        # Uncertainty metrics
        avg_uncertainty = uncertainties.mean().item()
        uncertainty_coverage = (torch.abs(predictions - targets) <= uncertainties).float().mean().item()
        
        print(f"\nEvaluation Results:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"Average Uncertainty: {avg_uncertainty:.6f}")
        print(f"Uncertainty Coverage: {uncertainty_coverage:.3f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'predictions': predictions,
            'targets': targets,
            'uncertainties': uncertainties,
            'uncertainty_coverage': uncertainty_coverage,
            'expert_outputs': expert_outputs
        }
    else:
        print("No valid predictions generated")
        return {}


def visualize_enhanced_results(results: dict, training_history: dict, num_samples: int = 3):
    """Visualize enhanced model results with expert analysis."""
    
    if not results:
        print("No results to visualize")
        return
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Prediction results
    for i in range(min(num_samples, len(results['predictions']))):
        ax = plt.subplot(4, 3, i + 1)
        
        time_steps = range(len(results['targets'][i]))
        
        ax.plot(time_steps, results['targets'][i].numpy(), 'b-', label='Ground Truth', linewidth=2)
        ax.plot(time_steps, results['predictions'][i].numpy(), 'r--', label='Prediction', linewidth=2)
        
        # Add uncertainty bands
        if results['uncertainties'][i].sum() > 0:
            pred_np = results['predictions'][i].numpy()
            unc_np = results['uncertainties'][i].numpy()
            ax.fill_between(time_steps, 
                          pred_np - unc_np, 
                          pred_np + unc_np, 
                          alpha=0.3, color='red', label='Uncertainty')
        
        ax.set_title(f'Sample {i+1}: Enhanced PGAT Prediction')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Training loss curve
    ax = plt.subplot(4, 3, 4)
    if training_history['losses']:
        ax.plot(training_history['losses'], 'g-', linewidth=2)
        ax.set_title('Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    
    # 3. Expert utilization over time
    ax = plt.subplot(4, 3, 5)
    if training_history['expert_utilization']:
        # Plot temporal expert utilization
        temporal_utils = []
        for util in training_history['expert_utilization']:
            if 'temporal' in util:
                temporal_utils.append(list(util['temporal'].values()))
        
        if temporal_utils:
            temporal_utils = np.array(temporal_utils)
            for i in range(temporal_utils.shape[1]):
                ax.plot(temporal_utils[:, i], label=f'Temporal Expert {i}')
        
        ax.set_title('Expert Utilization Over Time')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Utilization')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Memory usage over time
    ax = plt.subplot(4, 3, 6)
    if training_history['memory_usage']:
        memory_allocated = [mem['peak_memory']['peak_allocated_gb'] 
                          for mem in training_history['memory_usage'] 
                          if 'peak_memory' in mem]
        if memory_allocated:
            ax.plot(memory_allocated, 'purple', linewidth=2)
            ax.set_title('Peak Memory Usage')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Memory (GB)')
            ax.grid(True, alpha=0.3)
    
    # 5. Curriculum progression
    ax = plt.subplot(4, 3, 7)
    if training_history['curriculum_params']:
        progress_values = [params.get('progress', 0) 
                         for params in training_history['curriculum_params']]
        ax.plot(progress_values, 'orange', linewidth=2)
        ax.set_title('Curriculum Learning Progress')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Progress')
        ax.grid(True, alpha=0.3)
    
    # 6. Uncertainty distribution
    ax = plt.subplot(4, 3, 8)
    if results['uncertainties'].sum() > 0:
        uncertainties_flat = results['uncertainties'].flatten().numpy()
        ax.hist(uncertainties_flat, bins=50, alpha=0.7, color='red')
        ax.set_title('Uncertainty Distribution')
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # 7. Error vs Uncertainty scatter
    ax = plt.subplot(4, 3, 9)
    if results['uncertainties'].sum() > 0:
        errors = torch.abs(results['predictions'] - results['targets']).flatten().numpy()
        uncertainties_flat = results['uncertainties'].flatten().numpy()
        ax.scatter(uncertainties_flat, errors, alpha=0.5)
        ax.set_title('Prediction Error vs Uncertainty')
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('Absolute Error')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_sota_pgat_moe_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Enhanced results saved to 'enhanced_sota_pgat_moe_results.png'")


def main():
    """Main execution function for enhanced SOTA PGAT with MoE."""
    
    print("=" * 80)
    print("Enhanced SOTA Temporal PGAT with Mixture of Experts")
    print("=" * 80)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create enhanced configuration
    config = EnhancedSOTAConfig()
    print(f"Model configuration: d_model={config.d_model}, seq_len={config.seq_len}, pred_len={config.pred_len}")
    
    # Generate complex synthetic data
    print("\nGenerating complex synthetic data...")
    wave_data, target_data, graph_data = generate_complex_synthetic_data(
        batch_size=64,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        num_features=config.enc_in
    )
    
    # Create data loaders
    dataset = TensorDataset(wave_data, target_data, graph_data)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Initialize enhanced model
    print("\nInitializing Enhanced SOTA PGAT with MoE...")
    try:
        model = Enhanced_SOTA_Temporal_PGAT_MoE(config, mode='probabilistic')
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model initialized successfully with {total_params:,} parameters")
        
        # Print expert information
        print(f"\nAvailable expert types: {expert_registry.list_categories()}")
        for category in expert_registry.list_categories():
            experts = expert_registry.list_experts(category)
            print(f"  {category}: {experts}")
        
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Train model
    print("\n" + "="*60)
    print("ENHANCED TRAINING PHASE")
    print("="*60)
    
    try:
        training_history = train_enhanced_model(model, train_loader, config, num_epochs=5)
        print("Enhanced training completed successfully!")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate model
    print("\n" + "="*60)
    print("ENHANCED EVALUATION PHASE")
    print("="*60)
    
    try:
        eval_results = evaluate_enhanced_model(model, test_loader, device=device)
        
        if eval_results:
            print("\nEnhanced evaluation completed successfully!")
            
            # Visualize results
            print("\nGenerating comprehensive visualizations...")
            visualize_enhanced_results(eval_results, training_history, num_samples=3)
            
        else:
            print("Enhanced evaluation failed - no results generated")
            
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("ENHANCED SOTA PGAT WITH MOE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nKey Enhancements Demonstrated:")
    print("✓ Mixture of Experts with specialized temporal, spatial, and uncertainty experts")
    print("✓ Curriculum learning with progressive complexity increase")
    print("✓ Memory optimization with mixed precision and gradient checkpointing")
    print("✓ Hierarchical graph attention for multi-scale spatial modeling")
    print("✓ Advanced uncertainty quantification with multiple expert types")
    print("✓ Adaptive batch sizing and memory monitoring")
    print("✓ Comprehensive expert utilization tracking")
    print("✓ Enhanced visualization and analysis tools")


if __name__ == "__main__":
    main()