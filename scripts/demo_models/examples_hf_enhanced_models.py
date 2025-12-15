"""
HF Enhanced Models Usage Examples

This script demonstrates how to use the new HF Enhanced Models with:
1. Different configurations and features
2. Integration with existing loss functions
3. Bayesian uncertainty quantification
4. Hierarchical multi-scale processing
5. Quantile regression
6. Complete training workflow

Run this to see practical usage examples.
"""

import sys
import os
import torch
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.HFAdvancedFactory import (
    create_hf_bayesian_model,
    create_hf_hierarchical_model,
    create_hf_quantile_model,
    create_hf_full_model,
    create_hf_model_from_config
)
from models.HFEnhancedAutoformer import HFEnhancedAutoformer
from utils.logger import logger


def example_1_basic_usage():
    """Example 1: Basic HF Enhanced Model Usage"""
    print("Example 1: Basic HF Enhanced Model Usage")
    print("-" * 50)
    
    # Basic configuration
    config = Namespace(
        seq_len=96, pred_len=24, enc_in=7, dec_in=7, c_out=7,
        d_model=512, n_heads=8, e_layers=2, d_layers=1,
        dropout=0.1, activation='gelu'
    )
    
    # Create standard HF model
    model = HFEnhancedAutoformer(config)
    print(f"Created standard HF model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create sample data
    batch_size = 4
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    x_dec = torch.randn(batch_size, 48 + config.pred_len, config.dec_in)
    x_mark_dec = torch.randn(batch_size, 48 + config.pred_len, 4)
    
    # Forward pass
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"Model output shape: {output.shape}")
        print(f"Expected: [{batch_size}, {config.pred_len}, {config.c_out}]")
    
    print("✓ Basic usage successful\n")


def example_2_bayesian_uncertainty():
    """Example 2: Bayesian Uncertainty Quantification"""
    print("Example 2: Bayesian Uncertainty Quantification")
    print("-" * 50)
    
    # Configuration with Bayesian features
    config = Namespace(
        seq_len=96, pred_len=24, enc_in=7, dec_in=7, c_out=7,
        d_model=512, n_heads=8, e_layers=2, d_layers=1,
        dropout=0.1, activation='gelu',
        # Bayesian settings
        use_bayesian=True,
        uncertainty_method='bayesian',
        n_samples=10,
        kl_weight=1e-5,
        bayesian_layers=['output_projection'],
        loss_function='mse'
    )
    
    # Create Bayesian HF model
    model = create_hf_bayesian_model(config)
    info = model.get_model_info()
    print(f"Created Bayesian HF model:")
    print(f"  - Total parameters: {info['parameters']['total']:,}")
    print(f"  - Uncertainty method: {info['bayesian_info']['uncertainty_method']}")
    print(f"  - Bayesian layers: {info['bayesian_info']['bayesian_layers']}")
    
    # Create sample data
    batch_size = 2
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    x_dec = torch.randn(batch_size, 48 + config.pred_len, config.dec_in)
    x_mark_dec = torch.randn(batch_size, 48 + config.pred_len, 4)
    targets = torch.randn(batch_size, config.pred_len, config.c_out)
    
    # Forward pass with uncertainty
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if isinstance(output, dict):
            prediction = output['prediction']
            uncertainty = output['uncertainty']
            
            print(f"Prediction shape: {prediction.shape}")
            print(f"Uncertainty shape: {uncertainty.shape}")
            print(f"Mean uncertainty: {uncertainty.mean().item():.6f}")
            print(f"Confidence intervals: {list(output['confidence_intervals'].keys())}")
            
            # Show uncertainty statistics for first sample, first timestep
            sample_uncertainty = uncertainty[0, 0, :].numpy()
            print(f"Feature uncertainties (sample 0, timestep 0): {sample_uncertainty}")
    
    # Training mode with loss computation
    model.train()
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    total_loss, components = model.compute_loss(output, targets, x_enc)
    
    print(f"Training loss: {total_loss.item():.6f}")
    print(f"Loss components: {list(components.keys())}")
    for name, value in components.items():
        print(f"  - {name}: {value.item():.6f}")
    
    print("✓ Bayesian uncertainty example successful\n")


def example_3_hierarchical_processing():
    """Example 3: Hierarchical Multi-Scale Processing"""
    print("Example 3: Hierarchical Multi-Scale Processing")
    print("-" * 50)
    
    # Configuration with hierarchical features
    config = Namespace(
        seq_len=96, pred_len=24, enc_in=7, dec_in=7, c_out=7,
        d_model=512, n_heads=8, e_layers=2, d_layers=1,
        dropout=0.1, activation='gelu',
        # Hierarchical settings
        use_hierarchical=True,
        use_wavelet=True,
        hierarchy_levels=[1, 2, 4],
        aggregation_method='adaptive',
        wavelet_type='db4',
        decomposition_levels=3,
        reconstruction_method='learnable',
        loss_function='mse'
    )
    
    # Create hierarchical HF model
    model = create_hf_hierarchical_model(config)
    info = model.get_model_info()
    print(f"Created Hierarchical HF model:")
    print(f"  - Total parameters: {info['parameters']['total']:,}")
    print(f"  - Extensions: {info['extensions']}")
    print(f"  - Hierarchy levels: {info['hierarchical_info']['hierarchy_levels']}")
    print(f"  - Wavelet type: {info['wavelet_info']['wavelet_type']}")
    
    # Create sample data
    batch_size = 2
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    x_dec = torch.randn(batch_size, 48 + config.pred_len, config.dec_in)
    x_mark_dec = torch.randn(batch_size, 48 + config.pred_len, 4)
    targets = torch.randn(batch_size, config.pred_len, config.c_out)
    
    # Forward pass
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"Hierarchical output shape: {output.shape}")
        
    # Training with loss
    total_loss, components = model.compute_loss(output, targets, x_enc)
    print(f"Training loss: {total_loss.item():.6f}")
    print(f"Loss components: {list(components.keys())}")
    
    print("✓ Hierarchical processing example successful\n")


def example_4_quantile_regression():
    """Example 4: Quantile Regression"""
    print("Example 4: Quantile Regression")
    print("-" * 50)
    
    # Configuration with quantile features
    config = Namespace(
        seq_len=96, pred_len=24, enc_in=7, dec_in=7, c_out=7,
        d_model=512, n_heads=8, e_layers=2, d_layers=1,
        dropout=0.1, activation='gelu',
        # Quantile settings
        use_quantile=True,
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        loss_function='pinball'
    )
    
    # Create quantile HF model
    model = create_hf_quantile_model(config)
    info = model.get_model_info()
    print(f"Created Quantile HF model:")
    print(f"  - Total parameters: {info['parameters']['total']:,}")
    print(f"  - Quantile levels: {config.quantile_levels}")
    
    # Create sample data
    batch_size = 2
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    x_dec = torch.randn(batch_size, 48 + config.pred_len, config.dec_in)
    x_mark_dec = torch.randn(batch_size, 48 + config.pred_len, 4)
    
    # Targets for quantile regression (expanded)
    n_quantiles = len(config.quantile_levels)
    base_targets = torch.randn(batch_size, config.pred_len, config.c_out)
    quantile_targets = base_targets.repeat(1, 1, n_quantiles)
    
    # Forward pass
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        expected_features = config.c_out * n_quantiles
        print(f"Quantile output shape: {output.shape}")
        print(f"Expected features: {expected_features} ({config.c_out} * {n_quantiles} quantiles)")
        
        # Extract quantile predictions for first feature
        feature_0_quantiles = output[0, 0, :config.c_out].numpy()  # First sample, first timestep
        print(f"Feature 0 quantile predictions: {feature_0_quantiles}")
    
    # Training with pinball loss
    total_loss, components = model.compute_loss(output, quantile_targets, x_enc)
    print(f"Pinball loss: {total_loss.item():.6f}")
    
    print("✓ Quantile regression example successful\n")


def example_5_full_model():
    """Example 5: Full Model with All Features"""
    print("Example 5: Full Model with All Features")
    print("-" * 50)
    
    # Configuration with all features
    config = Namespace(
        seq_len=96, pred_len=24, enc_in=7, dec_in=7, c_out=7,
        d_model=512, n_heads=8, e_layers=2, d_layers=1,
        dropout=0.1, activation='gelu',
        # All features enabled
        use_bayesian=True,
        use_hierarchical=True,
        use_wavelet=True,
        use_quantile=True,
        # Bayesian settings
        uncertainty_method='bayesian',
        n_samples=8,
        kl_weight=1e-5,
        # Hierarchical settings
        hierarchy_levels=[1, 2, 4],
        aggregation_method='adaptive',
        wavelet_type='db4',
        decomposition_levels=3,
        # Quantile settings
        quantile_levels=[0.1, 0.5, 0.9],
        loss_function='pinball'
    )
    
    # Create full HF model
    model = create_hf_full_model(config)
    info = model.get_model_info()
    print(f"Created Full HF model:")
    print(f"  - Total parameters: {info['parameters']['total']:,}")
    print(f"  - Extensions: {info['extensions']}")
    print(f"  - Capabilities: {list(info['capabilities'].keys())}")
    
    # Create sample data
    batch_size = 2
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    x_dec = torch.randn(batch_size, 48 + config.pred_len, config.dec_in)
    x_mark_dec = torch.randn(batch_size, 48 + config.pred_len, 4)
    
    # Targets for quantile regression
    n_quantiles = len(config.quantile_levels)
    base_targets = torch.randn(batch_size, config.pred_len, config.c_out)
    quantile_targets = base_targets.repeat(1, 1, n_quantiles)
    
    # Forward pass with all features
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if isinstance(output, dict):
            prediction = output['prediction']
            uncertainty = output['uncertainty']
            
            print(f"Full model output:")
            print(f"  - Prediction shape: {prediction.shape}")
            print(f"  - Uncertainty shape: {uncertainty.shape}")
            print(f"  - Mean uncertainty: {uncertainty.mean().item():.6f}")
            print(f"  - Has confidence intervals: {'confidence_intervals' in output}")
            
            # Show combined quantile + uncertainty results
            print(f"Combined capabilities:")
            print(f"  - Quantile predictions with uncertainty bounds")
            print(f"  - Multi-scale hierarchical processing")
            print(f"  - Wavelet decomposition and reconstruction")
    
    # Training with full loss
    model.train()
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    total_loss, components = model.compute_loss(output, quantile_targets, x_enc)
    
    print(f"Full training loss: {total_loss.item():.6f}")
    print(f"All loss components: {list(components.keys())}")
    for name, value in components.items():
        print(f"  - {name}: {value.item():.6f}")
    
    print("✓ Full model example successful\n")


def example_6_existing_loss_integration():
    """Example 6: Integration with Existing Loss Functions"""
    print("Example 6: Integration with Existing Loss Functions")
    print("-" * 50)
    
    # Test different loss functions from existing infrastructure
    loss_configs = [
        {'name': 'MSE', 'loss_function': 'mse'},
        {'name': 'MAE', 'loss_function': 'mae'},
        {'name': 'Huber', 'loss_function': 'huber'},
        {'name': 'Pinball', 'loss_function': 'pinball', 'use_quantile': True},
        {'name': 'DTW', 'loss_function': 'dtw'},
        {'name': 'MASE', 'loss_function': 'mase'}
    ]
    
    base_config = Namespace(
        seq_len=96, pred_len=24, enc_in=7, dec_in=7, c_out=7,
        d_model=256, n_heads=4, e_layers=1, d_layers=1,
        dropout=0.1, activation='gelu',
        use_bayesian=True,  # Add Bayesian for richer testing
        uncertainty_method='dropout',  # Use dropout for speed
        n_samples=3
    )
    
    # Create sample data
    batch_size = 2
    x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
    x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
    x_dec = torch.randn(batch_size, 48 + base_config.pred_len, base_config.dec_in)
    x_mark_dec = torch.randn(batch_size, 48 + base_config.pred_len, 4)
    
    for loss_config in loss_configs:
        try:
            print(f"Testing {loss_config['name']} loss...")
            
            # Create config for this loss
            config = Namespace(**vars(base_config))
            config.loss_function = loss_config['loss_function']
            
            if loss_config.get('use_quantile', False):
                config.use_quantile = True
                config.quantile_levels = [0.1, 0.5, 0.9]
                targets = torch.randn(batch_size, base_config.pred_len, base_config.c_out * 3)
            else:
                targets = torch.randn(batch_size, base_config.pred_len, base_config.c_out)
            
            # Create model
            model = create_hf_bayesian_model(config)
            
            # Forward and loss
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            total_loss, components = model.compute_loss(output, targets, x_enc)
            
            print(f"  ✓ {loss_config['name']}: {total_loss.item():.6f}")
            
        except Exception as e:
            print(f"  ✗ {loss_config['name']}: {e}")
    
    print("✓ Loss integration example successful\n")


def example_7_training_workflow():
    """Example 7: Complete Training Workflow"""
    print("Example 7: Complete Training Workflow")
    print("-" * 50)
    
    # Configuration for training
    config = Namespace(
        seq_len=96, pred_len=24, enc_in=7, dec_in=7, c_out=7,
        d_model=256, n_heads=4, e_layers=1, d_layers=1,
        dropout=0.1, activation='gelu',
        # Use Bayesian + Quantile for comprehensive example
        use_bayesian=True,
        use_quantile=True,
        uncertainty_method='dropout',
        n_samples=5,
        quantile_levels=[0.1, 0.5, 0.9],
        loss_function='pinball',
        # Training settings
        learning_rate=0.001,
        weight_decay=1e-4
    )
    
    # Create model
    model = create_hf_full_model(config)
    print(f"Training model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Create sample dataset
    n_samples = 10
    dataset = []
    for _ in range(n_samples):
        x_enc = torch.randn(config.seq_len, config.enc_in)
        x_mark_enc = torch.randn(config.seq_len, 4)
        x_dec = torch.randn(48 + config.pred_len, config.dec_in)
        x_mark_dec = torch.randn(48 + config.pred_len, 4)
        targets = torch.randn(config.pred_len, config.c_out * len(config.quantile_levels))
        dataset.append((x_enc, x_mark_enc, x_dec, x_mark_dec, targets))
    
    # Training loop
    model.train()
    n_epochs = 3
    
    for epoch in range(n_epochs):
        epoch_losses = []
        
        for i, (x_enc, x_mark_enc, x_dec, x_mark_dec, targets) in enumerate(dataset):
            # Add batch dimension
            x_enc = x_enc.unsqueeze(0)
            x_mark_enc = x_mark_enc.unsqueeze(0)
            x_dec = x_dec.unsqueeze(0)
            x_mark_dec = x_mark_dec.unsqueeze(0)
            targets = targets.unsqueeze(0)
            
            # Forward pass
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Compute loss
            total_loss, components = model.compute_loss(output, targets, x_enc)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1}/{n_epochs}: Loss = {avg_loss:.6f}")
    
    print("✓ Training workflow example successful\n")


def main():
    """Run all examples"""
    print("=" * 70)
    print("HF Enhanced Models - Usage Examples")
    print("=" * 70)
    print()
    
    examples = [
        example_1_basic_usage,
        example_2_bayesian_uncertainty,
        example_3_hierarchical_processing,
        example_4_quantile_regression,
        example_5_full_model,
        example_6_existing_loss_integration,
        example_7_training_workflow
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ {example.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 70)
    print("🎉 All examples completed!")
    print("\nKey takeaways:")
    print("1. HF Enhanced Models leverage existing Hugging Face stability")
    print("2. Advanced features are added through external extensions")
    print("3. Full integration with existing loss function infrastructure")
    print("4. Bayesian uncertainty quantification with multiple methods")
    print("5. Hierarchical processing with wavelet decomposition")
    print("6. Quantile regression with pinball loss")
    print("7. All features can be combined in a single model")
    print("8. Complete compatibility with existing training workflows")
    print("=" * 70)


if __name__ == "__main__":
    main()
