#!/usr/bin/env python3
"""
SOTA Temporal PGAT Example

This example demonstrates how to use the enhanced SOTA Temporal PGAT model
with all the new components including:
- Mixture Density Network (MDN) decoder
- AutoCorrelation temporal attention
- Structural positional encoding
- Enhanced temporal encoding
- Dynamic edge weights in PGAT layers

Author: SOTA PGAT Implementation Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# Import the enhanced SOTA PGAT model and components
from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
from layers.modular.decoder.mixture_density_decoder import MixtureNLLLoss

class SOTAConfig:
    """Configuration class for SOTA Temporal PGAT model."""
    
    def __init__(self):
        # Model dimensions
        self.d_model = 512
        self.n_heads = 8
        self.d_ff = 2048
        
        # Sequence parameters
        self.seq_len = 96
        self.pred_len = 96
        self.enc_in = 7  # Number of input features
        self.dec_in = 7
        self.c_out = 7
        
        # Enhanced components configuration
        self.use_mixture_decoder = True
        self.num_mixture_components = 5
        
        self.use_autocorr_attention = True
        self.autocorr_factor = 3
        
        self.use_structural_encoding = True
        self.num_eigenvectors = 16
        
        self.use_enhanced_temporal_encoding = True
        self.temporal_encoding_scales = 4
        
        self.use_dynamic_edge_weights = True
        
        # Training parameters
        self.dropout = 0.1
        self.activation = 'gelu'
        self.e_layers = 2
        self.d_layers = 1
        
        # Graph parameters
        self.max_nodes = 1000
        self.edge_weight_dropout = 0.1

def create_sample_data(batch_size: int = 32, seq_len: int = 96, pred_len: int = 96, 
                      num_features: int = 7) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sample time series data for demonstration.
    
    Returns:
        Tuple of (input_data, target_data)
    """
    # Generate synthetic time series with multiple patterns
    t = torch.linspace(0, 4*np.pi, seq_len + pred_len)
    
    data_list = []
    targets_list = []
    
    for i in range(batch_size):
        # Create multi-variate time series with different patterns
        series = torch.zeros(seq_len + pred_len, num_features)
        
        for j in range(num_features):
            # Different frequency and phase for each feature
            freq = 0.5 + j * 0.2
            phase = i * 0.1 + j * 0.3
            
            # Combine sine waves with trend and noise
            trend = 0.01 * t
            seasonal = torch.sin(freq * t + phase) + 0.3 * torch.sin(2 * freq * t + phase)
            noise = 0.1 * torch.randn_like(t)
            
            series[:, j] = trend + seasonal + noise
        
        # Split into input and target
        input_seq = series[:seq_len]
        target_seq = series[seq_len:seq_len + pred_len]
        
        data_list.append(input_seq)
        targets_list.append(target_seq)
    
    return torch.stack(data_list), torch.stack(targets_list)

def create_sample_graph_data(batch_size: int, seq_len: int, num_features: int) -> Dict:
    """
    Create sample heterogeneous graph data for PGAT.
    
    Returns:
        Dictionary containing graph structure information
    """
    # For demonstration, create a simple graph structure
    # In practice, this would come from your domain knowledge
    
    # Create node mappings
    num_wave_nodes = seq_len // 3
    num_transition_nodes = seq_len // 3
    num_target_nodes = seq_len - num_wave_nodes - num_transition_nodes
    
    # Create edge indices for heterogeneous graph
    # Wave nodes interact with transition nodes
    wave_to_trans_edges = []
    for i in range(min(num_wave_nodes, num_transition_nodes)):
        wave_to_trans_edges.append([i, i])
        if i > 0:
            wave_to_trans_edges.append([i-1, i])
    
    # Transition nodes influence target nodes
    trans_to_target_edges = []
    for i in range(min(num_transition_nodes, num_target_nodes)):
        trans_to_target_edges.append([i, i])
        if i > 0:
            trans_to_target_edges.append([i-1, i])
    
    edge_index_dict = {
        ('wave', 'interacts_with', 'transition'): torch.tensor(wave_to_trans_edges).T,
        ('transition', 'influences', 'target'): torch.tensor(trans_to_target_edges).T
    }
    
    return {
        'edge_index_dict': edge_index_dict,
        'num_nodes_dict': {
            'wave': num_wave_nodes,
            'transition': num_transition_nodes,
            'target': num_target_nodes
        }
    }

def train_model(model: nn.Module, train_loader: DataLoader, 
               config: SOTAConfig, num_epochs: int = 10) -> Dict:
    """
    Train the SOTA PGAT model.
    
    Returns:
        Dictionary containing training history
    """
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    if config.use_mixture_decoder:
        criterion = MixtureNLLLoss()
    else:
        criterion = nn.MSELoss()
    
    # Training history
    history = {'train_loss': [], 'lr': []}
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            if config.use_mixture_decoder:
                # Model returns mixture parameters
                mixture_params = model(data)
                loss = criterion(mixture_params, target)
            else:
                # Standard prediction
                pred = model(data)
                loss = criterion(pred, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        
        history['train_loss'].append(avg_loss)
        history['lr'].append(current_lr)
        
        print(f'Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.6f}, LR: {current_lr:.2e}')
    
    return history

def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                  config: SOTAConfig) -> Dict:
    """
    Evaluate the trained model and compute uncertainty metrics.
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    predictions = []
    targets = []
    uncertainties = []
    
    if config.use_mixture_decoder:
        criterion = MixtureNLLLoss()
    else:
        criterion = nn.MSELoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            if config.use_mixture_decoder:
                # Get mixture parameters
                mixture_params = model(data)
                loss = criterion(mixture_params, target)
                
                # Get prediction summary
                pred_mean, pred_std = model.decoder.prediction_summary(mixture_params)
                
                predictions.append(pred_mean)
                uncertainties.append(pred_std)
            else:
                pred = model(data)
                loss = criterion(pred, target)
                predictions.append(pred)
            
            targets.append(target)
            total_loss += loss.item()
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(predictions, dim=0)
    all_targets = torch.cat(targets, dim=0)
    
    # Compute metrics
    mse = torch.mean((all_predictions - all_targets) ** 2)
    mae = torch.mean(torch.abs(all_predictions - all_targets))
    
    metrics = {
        'test_loss': total_loss / len(test_loader),
        'mse': mse.item(),
        'mae': mae.item()
    }
    
    if config.use_mixture_decoder and uncertainties:
        all_uncertainties = torch.cat(uncertainties, dim=0)
        metrics['mean_uncertainty'] = torch.mean(all_uncertainties).item()
        metrics['uncertainty_std'] = torch.std(all_uncertainties).item()
    
    return metrics, all_predictions, all_targets

def visualize_predictions(predictions: torch.Tensor, targets: torch.Tensor, 
                         uncertainties: Optional[torch.Tensor] = None, 
                         num_samples: int = 3):
    """
    Visualize model predictions with uncertainty bands.
    """
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 8))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Plot target and prediction
        time_steps = range(predictions.shape[1])
        
        # Plot first feature for simplicity
        target_series = targets[i, :, 0].numpy()
        pred_series = predictions[i, :, 0].numpy()
        
        ax.plot(time_steps, target_series, 'b-', label='Ground Truth', linewidth=2)
        ax.plot(time_steps, pred_series, 'r--', label='Prediction', linewidth=2)
        
        # Add uncertainty bands if available
        if uncertainties is not None:
            uncertainty = uncertainties[i, :, 0].numpy()
            ax.fill_between(time_steps, 
                          pred_series - 2*uncertainty, 
                          pred_series + 2*uncertainty, 
                          alpha=0.3, color='red', label='95% Confidence')
        
        ax.set_title(f'Sample {i+1} - Time Series Prediction')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_model_components(model: SOTA_Temporal_PGAT, sample_input: torch.Tensor):
    """
    Analyze the behavior of different model components.
    """
    print("\n=== Model Component Analysis ===")
    
    model.eval()
    with torch.no_grad():
        # Forward pass to get intermediate outputs
        output = model(sample_input)
        
        print(f"Input shape: {sample_input.shape}")
        
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'prediction_summary'):
            pred_mean, pred_std = model.decoder.prediction_summary(output)
            print(f"Prediction mean shape: {pred_mean.shape}")
            print(f"Prediction std shape: {pred_std.shape}")
            print(f"Mean uncertainty: {torch.mean(pred_std):.4f}")
            print(f"Uncertainty range: [{torch.min(pred_std):.4f}, {torch.max(pred_std):.4f}]")
        
        # Analyze edge weights if available
        if hasattr(model, 'spatial_encoder') and hasattr(model.spatial_encoder, 'get_edge_weights'):
            print("\nDynamic Edge Weight Analysis:")
            # This would require proper graph data structure
            print("- Dynamic edge weights are enabled")
            print("- Edge weights adapt based on node features")
        
        print(f"\nModel has {sum(p.numel() for p in model.parameters())} total parameters")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

def main():
    """
    Main function demonstrating SOTA PGAT usage.
    """
    print("SOTA Temporal PGAT Example")
    print("=" * 50)
    
    # Configuration
    config = SOTAConfig()
    
    # Create sample data
    print("\n1. Creating sample data...")
    train_data, train_targets = create_sample_data(
        batch_size=128, seq_len=config.seq_len, 
        pred_len=config.pred_len, num_features=config.enc_in
    )
    
    test_data, test_targets = create_sample_data(
        batch_size=32, seq_len=config.seq_len, 
        pred_len=config.pred_len, num_features=config.enc_in
    )
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Create data loaders
    train_dataset = TensorDataset(train_data, train_targets)
    test_dataset = TensorDataset(test_data, test_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    print("\n2. Initializing SOTA PGAT model...")
    model = SOTA_Temporal_PGAT(config)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Analyze model components
    sample_batch = train_data[:4]  # Small batch for analysis
    analyze_model_components(model, sample_batch)
    
    # Train model
    print("\n3. Training model...")
    history = train_model(model, train_loader, config, num_epochs=5)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate model
    print("\n4. Evaluating model...")
    metrics, predictions, targets = evaluate_model(model, test_loader, config)
    
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Visualize predictions
    print("\n5. Visualizing predictions...")
    uncertainties = None
    if config.use_mixture_decoder:
        # Get uncertainties for visualization
        model.eval()
        with torch.no_grad():
            sample_data = test_data[:3]
            mixture_params = model(sample_data)
            _, uncertainties = model.decoder.prediction_summary(mixture_params)
    
    visualize_predictions(predictions[:3], targets[:3], uncertainties)
    
    print("\n=== Example completed successfully! ===")
    print("\nKey Features Demonstrated:")
    print("✓ Mixture Density Network for uncertainty quantification")
    print("✓ AutoCorrelation attention for temporal modeling")
    print("✓ Enhanced temporal and structural encodings")
    print("✓ Dynamic edge weights in graph attention")
    print("✓ Comprehensive evaluation with uncertainty metrics")

if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()