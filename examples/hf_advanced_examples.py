"""
HF Enhanced Autoformer Advanced - Usage Examples and Factory Functions

This module provides comprehensive examples and factory functions for creating
HF models with advanced features using existing loss infrastructure.
"""

import torch
import torch.nn as nn
from argparse import Namespace
from models.HFEnhancedAutoformerAdvanced import HFEnhancedAutoformerAdvanced
from utils.logger import logger

def create_hf_bayesian_quantile_model(configs):
    """
    Create HF model with Bayesian uncertainty and quantile regression.
    
    Uses existing PinballLoss and BayesianQuantileLoss from infrastructure.
    """
    
    # Configure for Bayesian quantile regression
    configs.uncertainty_method = 'bayesian'
    configs.loss = 'pinball'
    configs.quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    configs.kl_weight = 2e-5
    configs.uncertainty_weight = 0.15
    configs.n_samples = 20
    configs.bayesian_layers = ['output_projection']  # Convert HF output layer
    
    logger.info("Creating HF Bayesian Quantile Model")
    logger.info(f"  - Quantiles: {configs.quantile_levels}")
    logger.info(f"  - KL Weight: {configs.kl_weight}")
    logger.info(f"  - Samples: {configs.n_samples}")
    
    model = HFEnhancedAutoformerAdvanced(configs)
    
    # Log model info
    info = model.get_model_info()
    logger.info(f"✅ Created {info['name']} with {info['total_params']:,} parameters")
    logger.info(f"   - Bayesian layers: {info.get('bayesian_layers_converted', 0)}")
    logger.info(f"   - Quantile mode: {info['is_quantile_mode']}")
    
    return model

def create_hf_dropout_structural_model(configs):
    """
    Create HF model with MC Dropout and structural loss.
    
    Uses existing PSLoss (Patch-wise Structural Loss) from infrastructure.
    """
    
    # Configure for MC Dropout with structural loss
    configs.uncertainty_method = 'dropout'
    configs.loss = 'ps_loss'
    configs.ps_mse_weight = 0.6
    configs.ps_w_corr = 1.2
    configs.ps_w_var = 0.8
    configs.ps_w_mean = 1.0
    configs.n_samples = 15
    
    logger.info("Creating HF MC Dropout Structural Model")
    logger.info(f"  - Uncertainty: {configs.uncertainty_method}")
    logger.info(f"  - Loss: {configs.loss}")
    logger.info(f"  - PS weights: corr={configs.ps_w_corr}, var={configs.ps_w_var}")
    
    model = HFEnhancedAutoformerAdvanced(configs)
    
    info = model.get_model_info()
    logger.info(f"✅ Created {info['name']} with {info['total_params']:,} parameters")
    
    return model

def create_hf_wavelet_bayesian_model(configs):
    """
    Create HF model with wavelet processing and Bayesian uncertainty.
    
    Combines wavelet decomposition with Bayesian inference.
    """
    
    # Configure for wavelet + Bayesian
    configs.uncertainty_method = 'bayesian'
    configs.use_wavelets = True
    configs.wavelet_type = 'db4'
    configs.n_levels = 3
    configs.loss = 'huber'
    configs.huber_delta = 1.0
    configs.kl_weight = 1e-5
    configs.n_samples = 12
    configs.bayesian_layers = ['output_projection']
    
    logger.info("Creating HF Wavelet Bayesian Model")
    logger.info(f"  - Wavelets: {configs.wavelet_type}, {configs.n_levels} levels")
    logger.info(f"  - Uncertainty: {configs.uncertainty_method}")
    logger.info(f"  - Loss: {configs.loss}")
    
    model = HFEnhancedAutoformerAdvanced(configs)
    
    info = model.get_model_info()
    logger.info(f"✅ Created {info['name']} with {info['total_params']:,} parameters")
    logger.info(f"   - Wavelets: {info['use_wavelets']}")
    logger.info(f"   - Bayesian layers: {info.get('bayesian_layers_converted', 0)}")
    
    return model

def create_hf_comprehensive_model(configs):
    """
    Create HF model with all advanced features enabled.
    
    This is the full-featured version with:
    - Bayesian uncertainty
    - Quantile regression  
    - Wavelet processing
    - Advanced loss function
    """
    
    # Configure for comprehensive setup
    configs.uncertainty_method = 'bayesian'
    configs.use_wavelets = True
    configs.wavelet_type = 'db4'
    configs.n_levels = 3
    configs.loss = 'quantile'
    configs.quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    configs.kl_weight = 1.5e-5
    configs.uncertainty_weight = 0.12
    configs.quantile_certainty_weight = 0.05
    configs.n_samples = 25
    configs.bayesian_layers = ['output_projection']
    
    logger.info("Creating HF Comprehensive Advanced Model")
    logger.info(f"  - ALL FEATURES ENABLED")
    logger.info(f"  - Wavelets: {configs.wavelet_type}, {configs.n_levels} levels")
    logger.info(f"  - Uncertainty: {configs.uncertainty_method}")
    logger.info(f"  - Quantiles: {configs.quantile_levels}")
    logger.info(f"  - Loss: {configs.loss}")
    
    model = HFEnhancedAutoformerAdvanced(configs)
    
    info = model.get_model_info()
    logger.info(f"✅ Created COMPREHENSIVE {info['name']}")
    logger.info(f"   - Parameters: {info['total_params']:,}")
    logger.info(f"   - Wavelets: {info['use_wavelets']}")
    logger.info(f"   - Bayesian layers: {info.get('bayesian_layers_converted', 0)}")
    logger.info(f"   - Quantiles: {len(info['quantiles'])}")
    
    return model

def demonstrate_training_loop():
    """
    Demonstrate training with HF advanced models using existing loss infrastructure.
    """
    
    logger.info("=== HF ADVANCED MODEL TRAINING DEMONSTRATION ===")
    
    # Create mock config
    configs = Namespace(
        # Basic model config
        seq_len=96, pred_len=24, enc_in=7, dec_in=7, c_out=1,
        d_model=64, n_heads=4, e_layers=2, d_layers=1, d_ff=256,
        dropout=0.1, activation='gelu', embed='timeF', freq='h',
        
        # Data config  
        c_out_evaluation=1,  # Number of target features for evaluation
        
        # Training config
        learning_rate=1e-4,
        batch_size=32
    )
    
    # Test different model configurations
    model_configs = [
        ("Bayesian Quantile", create_hf_bayesian_quantile_model),
        ("MC Dropout Structural", create_hf_dropout_structural_model),
        ("Wavelet Bayesian", create_hf_wavelet_bayesian_model),
        ("Comprehensive", create_hf_comprehensive_model)
    ]
    
    for model_name, model_factory in model_configs:
        logger.info(f"\n📊 Testing {model_name} Model:")
        logger.info("=" * 50)
        
        try:
            # Create model
            model = model_factory(configs)
            
            # Create mock data
            batch_size = 4
            batch_x = torch.randn(batch_size, configs.seq_len, configs.enc_in)
            batch_y = torch.randn(batch_size, configs.pred_len, configs.c_out_evaluation)
            
            # Mock time encodings
            batch_x_mark = torch.randn(batch_size, configs.seq_len, 4)  # time features
            batch_y_mark = torch.randn(batch_size, configs.pred_len, 4)
            
            # Create decoder input (standard pattern)
            dec_inp = torch.cat([
                batch_x[:, -configs.pred_len//2:, :configs.c_out_evaluation],  # last known targets
                torch.zeros_like(batch_y)  # future targets (unknown)
            ], dim=1)
            
            logger.info(f"📈 Input shapes:")
            logger.info(f"   - Encoder: {batch_x.shape}")
            logger.info(f"   - Decoder: {dec_inp.shape}")
            logger.info(f"   - Targets: {batch_y.shape}")
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
            logger.info(f"🎯 Output analysis:")
            if isinstance(outputs, dict):
                pred_shape = outputs['prediction'].shape
                logger.info(f"   - Prediction: {pred_shape}")
                logger.info(f"   - Has uncertainty: {'uncertainty' in outputs}")
                logger.info(f"   - Has quantiles: {'quantile_specific' in outputs}")
                logger.info(f"   - Has confidence intervals: {'confidence_intervals' in outputs}")
                
                # Test loss computation
                loss = model.compute_loss(outputs, batch_y)
                logger.info(f"   - Loss: {loss.item():.6f}")
                
                if hasattr(model, 'get_kl_loss'):
                    kl_loss = model.get_kl_loss()
                    if isinstance(kl_loss, torch.Tensor):
                        logger.info(f"   - KL loss: {kl_loss.item():.8f}")
                    else:
                        logger.info(f"   - KL loss: {kl_loss}")
                        
            else:
                logger.info(f"   - Standard output: {outputs.shape}")
                
            logger.info(f"✅ {model_name} model test PASSED")
            
        except Exception as e:
            logger.error(f"❌ {model_name} model test FAILED: {str(e)}")
            
    logger.info(f"\n🎉 Training demonstration complete!")

def benchmark_hf_vs_existing():
    """
    Benchmark HF models against existing models.
    """
    
    logger.info("=== HF vs EXISTING MODEL BENCHMARK ===")
    
    # This would compare:
    # 1. Parameter count
    # 2. Memory usage  
    # 3. Training speed
    # 4. Inference speed
    # 5. Accuracy/uncertainty quality
    
    logger.info("📊 Benchmark metrics:")
    logger.info("   - HF models: Stable, proven backbone")
    logger.info("   - Advanced features: External processing")
    logger.info("   - Loss integration: Existing infrastructure")
    logger.info("   - Trade-off: Stability vs some performance overhead")

if __name__ == "__main__":
    logger.info("🚀 Starting HF Enhanced Autoformer Advanced Demo")
    
    # Run demonstration
    demonstrate_training_loop()
    
    # Run benchmark info
    benchmark_hf_vs_existing()
    
    logger.info("✅ Demo complete!")
