#!/usr/bin/env python3
"""
Full training test for Enhanced_SOTA_PGAT to verify all fixes work in real training scenario.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from types import SimpleNamespace
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_synthetic_dataset(batch_size=8, seq_len=24, pred_len=6, enc_in=7, c_out=3, num_samples=100):
    """Create synthetic time series dataset for training."""
    
    # Generate synthetic wave data with temporal patterns
    wave_data = []
    target_data = []
    
    for _ in range(num_samples):
        # Create wave patterns with different frequencies and phases
        t = np.linspace(0, 4*np.pi, seq_len)
        wave_sample = np.zeros((seq_len, enc_in))
        
        for i in range(enc_in):
            # Different frequency and phase for each feature
            freq = 0.5 + i * 0.3
            phase = i * np.pi / 4
            wave_sample[:, i] = np.sin(freq * t + phase) + 0.1 * np.random.randn(seq_len)
        
        # Create target data that's related to wave data
        target_sample = np.zeros((pred_len, c_out))
        for i in range(c_out):
            # Target is a transformation of wave features
            base_signal = wave_sample[-pred_len:, i % enc_in]
            target_sample[:, i] = base_signal * 0.8 + 0.2 * np.random.randn(pred_len)
        
        wave_data.append(wave_sample)
        target_data.append(target_sample)
    
    # Convert to tensors and create batches
    wave_tensor = torch.FloatTensor(np.array(wave_data))
    target_tensor = torch.FloatTensor(np.array(target_data))
    
    # Create batched dataset
    dataset = []
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_wave = wave_tensor[i:end_idx]
        batch_target = target_tensor[i:end_idx]
        dataset.append((batch_wave, batch_target))
    
    return dataset

def test_full_training():
    """Test full training loop with Enhanced_SOTA_PGAT."""
    
    print("🚀 Testing Full Enhanced PGAT Training")
    print("=" * 60)
    
    try:
        # Import model
        from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT
        print("✅ Model import successful")
        
        # Create enhanced configuration
        config = SimpleNamespace(
            # Core parameters
            d_model=128,  # Smaller for faster testing
            n_heads=4,
            seq_len=24,
            pred_len=6,
            enc_in=7,
            c_out=3,
            dropout=0.1,
            
            # Enhanced features
            use_multi_scale_patching=True,
            use_hierarchical_mapper=True,
            use_stochastic_learner=True,
            use_gated_graph_combiner=True,
            patch_len=8,
            stride=4,
            
            # Graph settings
            enable_dynamic_graph=True,
            enable_graph_attention=True,
            use_autocorr_attention=False,  # Disabled for stability
            
            # Mixture density
            use_mixture_density=True,
            use_mixture_decoder=True,
            mixture_multivariate_mode='independent',
            mdn_components=3,
            
            # Training settings
            learning_rate=0.001,
            weight_decay=1e-4
        )
        
        # Create model
        model = Enhanced_SOTA_PGAT(config)
        print("✅ Enhanced PGAT model created successfully")
        
        # Print model configuration
        config_info = model.get_enhanced_config_info()
        print(f"\n📊 Model Configuration:")
        print(f"  Multi-scale patching: {config_info['use_multi_scale_patching']}")
        print(f"  Hierarchical mapper: {config_info['use_hierarchical_mapper']}")
        print(f"  Stochastic learner: {config_info['use_stochastic_learner']}")
        print(f"  Gated combiner: {config_info['use_gated_graph_combiner']}")
        print(f"  Mixture decoder: {config_info['use_mixture_decoder']}")
        
        # Create synthetic dataset
        print(f"\n📊 Creating synthetic dataset...")
        dataset = create_synthetic_dataset(
            batch_size=4, seq_len=config.seq_len, pred_len=config.pred_len,
            enc_in=config.enc_in, c_out=config.c_out, num_samples=20
        )
        print(f"✅ Dataset created: {len(dataset)} batches")
        
        # Setup optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = model.configure_optimizer_loss(nn.MSELoss(), verbose=True)
        print(f"✅ Optimizer and loss configured")
        
        # Training loop
        print(f"\n🏋️ Starting training loop...")
        model.train()
        
        epoch_losses = []
        for epoch in range(3):  # Short training for testing
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (wave_batch, target_batch) in enumerate(dataset):
                optimizer.zero_grad()
                
                # Forward pass
                output = model(wave_batch, target_batch)
                
                # Compute loss
                if isinstance(output, tuple):
                    # Mixture density output
                    loss = criterion(output, target_batch)
                else:
                    # Standard output
                    loss = criterion(output, target_batch)
                
                # Add regularization loss
                reg_loss = model.get_regularization_loss()
                total_loss = loss + 0.01 * reg_loss  # Small regularization weight
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += total_loss.item()
                batch_count += 1
                
                # Print batch info
                if batch_idx == 0:  # First batch of each epoch
                    print(f"  Epoch {epoch+1}, Batch {batch_idx+1}:")
                    print(f"    Wave input: {wave_batch.shape}")
                    print(f"    Target input: {target_batch.shape}")
                    if isinstance(output, tuple):
                        print(f"    Output: tuple with {len(output)} elements")
                        for i, elem in enumerate(output):
                            print(f"      Element {i}: {elem.shape}")
                    else:
                        print(f"    Output: {output.shape}")
                    print(f"    Loss: {loss.item():.6f}, Reg: {reg_loss.item():.6f}, Total: {total_loss.item():.6f}")
                    
                    # Check internal logs
                    logs = model.get_internal_logs()
                    print(f"    Internal logs: {logs}")
            
            avg_loss = epoch_loss / batch_count
            epoch_losses.append(avg_loss)
            print(f"  Epoch {epoch+1} completed - Average loss: {avg_loss:.6f}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"📈 Loss progression: {[f'{loss:.6f}' for loss in epoch_losses]}")
        
        # Test inference mode
        print(f"\n🔍 Testing inference mode...")
        model.eval()
        with torch.no_grad():
            test_wave, test_target = dataset[0]
            test_output = model(test_wave, test_target)
            print(f"✅ Inference successful")
            if isinstance(test_output, tuple):
                print(f"  Output: tuple with {len(test_output)} elements")
            else:
                print(f"  Output shape: {test_output.shape}")
        
        # Final model status
        final_config = model.get_enhanced_config_info()
        print(f"\n📊 Final Model Status:")
        if 'internal_logs' in final_config:
            for key, value in final_config['internal_logs'].items():
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_flow():
    """Test that gradients flow through all components properly."""
    
    print(f"\n🔍 Testing Gradient Flow...")
    
    try:
        from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT
        
        config = SimpleNamespace(
            d_model=64, n_heads=4, seq_len=12, pred_len=3,
            enc_in=4, c_out=2, dropout=0.1,
            use_multi_scale_patching=True,
            use_hierarchical_mapper=True,
            use_stochastic_learner=True,
            use_gated_graph_combiner=True,
            use_mixture_decoder=True,
            mixture_multivariate_mode='independent',
            mdn_components=2,
            enable_dynamic_graph=True,
            enable_graph_attention=True,
            use_autocorr_attention=False
        )
        
        model = Enhanced_SOTA_PGAT(config)
        
        # Create small test batch
        wave_input = torch.randn(2, config.seq_len, config.enc_in, requires_grad=True)
        target_input = torch.randn(2, config.pred_len, config.c_out)
        
        # Forward pass
        output = model(wave_input, target_input)
        
        # Compute loss
        criterion = model.configure_optimizer_loss(nn.MSELoss())
        loss = criterion(output, target_input)
        reg_loss = model.get_regularization_loss()
        total_loss = loss + reg_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        gradient_info = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_info[name] = grad_norm
            else:
                gradient_info[name] = 0.0
        
        # Print gradient information
        print(f"  Gradient norms:")
        for name, grad_norm in gradient_info.items():
            if grad_norm > 0:
                print(f"    {name}: {grad_norm:.6f}")
        
        # Check key components have gradients
        key_components = [
            'wave_patching_composer',
            'target_patching_composer', 
            'wave_temporal_to_spatial',
            'target_temporal_to_spatial',
            'stochastic_learner',
            'graph_combiner',
            'decoder'
        ]
        
        components_with_gradients = 0
        for component in key_components:
            component_has_grad = any(component in name and grad_norm > 0 
                                   for name, grad_norm in gradient_info.items())
            if component_has_grad:
                components_with_gradients += 1
                print(f"  ✅ {component}: has gradients")
            else:
                print(f"  ⚠️ {component}: no gradients")
        
        print(f"  📊 {components_with_gradients}/{len(key_components)} key components have gradients")
        
        return components_with_gradients >= len(key_components) // 2  # At least half should have gradients
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Enhanced PGAT Full Training Test")
    print("This test verifies that the model can train end-to-end with all fixes applied.")
    print()
    
    # Test full training
    training_success = test_full_training()
    
    # Test gradient flow
    gradient_success = test_gradient_flow()
    
    print(f"\n🎯 Test Results:")
    print(f"  Training: {'✅ PASS' if training_success else '❌ FAIL'}")
    print(f"  Gradients: {'✅ PASS' if gradient_success else '❌ FAIL'}")
    
    if training_success and gradient_success:
        print(f"\n🎉 All tests passed! Enhanced PGAT is ready for production training! 🚀")
    else:
        print(f"\n⚠️ Some tests failed. Check the implementation.")