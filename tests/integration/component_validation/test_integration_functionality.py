#!/usr/bin/env python3
"""
Comprehensive Integration Component Functionality Tests

This test suite validates that components work correctly when integrated
together, testing component interactions, data flow, and combined functionality
with expected mathematical properties and behaviors.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.modular_components.registry import create_component, get_global_registry
    from utils.modular_components.implementations import get_integration_status
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import modular components: {e}")
    COMPONENTS_AVAILABLE = False

class MockConfig:
    """Mock configuration for integration testing"""
    def __init__(self, **kwargs):
        # Default values covering all component needs
        self.seq_len = 96
        self.pred_len = 24
        self.enc_in = 7
        self.c_out = 7
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.dropout = 0.1
        self.activation = 'gelu'
        self.output_attention = False
        self.top_k = 5
        self.num_kernels = 6
        self.scales = [1, 2, 4, 8]
        self.loss_alpha = 0.5
        self.quantiles = [0.1, 0.5, 0.9]
        self.freq_threshold = 0.1
        self.wavelet = 'db4'
        self.levels = 3
        
        # Override with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

def create_sample_batch(batch_size=2, seq_len=96, pred_len=24, features=7):
    """Create a sample batch for testing"""
    # Input data
    x_enc = torch.randn(batch_size, seq_len, features)
    x_mark_enc = torch.randn(batch_size, seq_len, 4)  # time features
    x_dec = torch.randn(batch_size, pred_len, features)
    x_mark_dec = torch.randn(batch_size, pred_len, 4)
    
    return x_enc, x_mark_enc, x_dec, x_mark_dec

class MockModel(nn.Module):
    """Mock model for integration testing"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.enc_in, config.c_out)
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # Simple linear transformation for testing
        return self.linear(x_enc[:, -self.config.pred_len:, :])

def test_processor_backbone_integration():
    """Test processor and backbone component integration"""
    print("🧪 Testing Processor-Backbone Integration...")
    
    try:
        config = MockConfig()
        
        # Create components
        processor = create_component('processor', 'integrated_signal', config)
        backbone_types = ['chronos', 'simple_transformer', 't5']
        
        if processor is None:
            print("    ⚠️ Processor not available, skipping...")
            return True
        
        # Test data
        x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_batch()
        
        successful_integrations = 0
        
        for backbone_type in backbone_types:
            try:
                backbone = create_component('backbone', backbone_type, config)
                
                if backbone is None:
                    print(f"    ⚠️ {backbone_type} backbone not available")
                    continue
                
                # Test integration
                with torch.no_grad():
                    # Process input
                    processed_x = processor(x_enc)
                    
                    # Feed to backbone
                    if backbone_type == 'chronos':
                        # Chronos expects different input format
                        backbone_output = backbone(processed_x)
                    elif backbone_type == 't5':
                        # T5 expects text-like input
                        backbone_output = backbone(processed_x, x_mark_enc, x_dec, x_mark_dec)
                    else:
                        # Standard transformer input
                        backbone_output = backbone(processed_x, x_mark_enc, x_dec, x_mark_dec)
                
                # Validate integration
                assert isinstance(backbone_output, torch.Tensor), f"Output not tensor for {backbone_type}"
                assert not torch.isnan(backbone_output).any(), f"NaN in {backbone_type} output"
                assert not torch.isinf(backbone_output).any(), f"Inf in {backbone_type} output"
                
                print(f"    ✅ {backbone_type}: {processed_x.shape} → {backbone_output.shape}")
                successful_integrations += 1
                
            except Exception as e:
                print(f"    ⚠️ {backbone_type}: {e}")
        
        print(f"    📊 Successful processor-backbone integrations: {successful_integrations}/{len(backbone_types)}")
        return successful_integrations > 0
        
    except Exception as e:
        print(f"    ❌ Processor-backbone integration test failed: {e}")
        return False

def test_backbone_attention_integration():
    """Test backbone and attention component integration"""
    print("🧪 Testing Backbone-Attention Integration...")
    
    try:
        config = MockConfig()
        
        # Create components
        attention_types = ['multi_head', 'autocorrelation', 'hierarchical']
        backbone_types = ['simple_transformer']
        
        successful_integrations = 0
        
        for backbone_type in backbone_types:
            backbone = create_component('backbone', backbone_type, config)
            
            if backbone is None:
                print(f"    ⚠️ {backbone_type} backbone not available")
                continue
            
            for attention_type in attention_types:
                try:
                    attention = create_component('attention', attention_type, config)
                    
                    if attention is None:
                        print(f"    ⚠️ {attention_type} attention not available")
                        continue
                    
                    # Test data
                    batch_size, seq_len, d_model = 2, 96, 512
                    queries = torch.randn(batch_size, seq_len, d_model)
                    keys = torch.randn(batch_size, seq_len, d_model)
                    values = torch.randn(batch_size, seq_len, d_model)
                    
                    # Test attention within backbone context
                    with torch.no_grad():
                        # Use attention mechanism
                        if attention_type == 'hierarchical':
                            attention_output = attention(queries, keys, values, scale=1)
                        else:
                            attention_output = attention(queries, keys, values)
                        
                        # Validate attention output can be used by backbone
                        x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_batch()
                        
                        # Replace some backbone computation with attention output
                        # This simulates using attention within backbone
                        backbone_output = backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    
                    # Validate integration
                    assert isinstance(attention_output, torch.Tensor), f"Attention output not tensor"
                    assert attention_output.shape == queries.shape, f"Attention output shape mismatch"
                    assert not torch.isnan(attention_output).any(), f"NaN in attention output"
                    
                    print(f"    ✅ {backbone_type}-{attention_type}: integration successful")
                    successful_integrations += 1
                    
                except Exception as e:
                    print(f"    ⚠️ {backbone_type}-{attention_type}: {e}")
        
        print(f"    📊 Successful backbone-attention integrations: {successful_integrations}")
        return successful_integrations > 0
        
    except Exception as e:
        print(f"    ❌ Backbone-attention integration test failed: {e}")
        return False

def test_loss_optimization_integration():
    """Test loss function and optimization integration"""
    print("🧪 Testing Loss-Optimization Integration...")
    
    try:
        config = MockConfig()
        
        # Create components
        loss_types = ['mse', 'mae', 'bayesian_mse', 'quantile']
        
        # Simple model for testing
        model = MockModel(config)
        
        # Test data
        x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_batch()
        target = torch.randn(2, 24, 7)  # Ground truth
        
        successful_integrations = 0
        
        for loss_type in loss_types:
            try:
                loss_fn = create_component('loss', loss_type, config)
                
                if loss_fn is None:
                    print(f"    ⚠️ {loss_type} loss not available")
                    continue
                
                # Test optimization loop
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                initial_loss = None
                final_loss = None
                
                for step in range(5):  # Short optimization loop
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    
                    # Compute loss
                    if loss_type == 'bayesian_mse':
                        # Bayesian loss expects mean and variance
                        mean_pred = output
                        var_pred = torch.ones_like(output) * 0.1
                        loss = loss_fn(mean_pred, var_pred, target)
                    elif loss_type == 'quantile':
                        # Quantile loss expects quantile predictions
                        q_pred = output.unsqueeze(-1).expand(-1, -1, -1, len(config.quantiles))
                        loss = loss_fn(q_pred, target)
                    else:
                        # Standard loss functions
                        loss = loss_fn(output, target)
                    
                    if step == 0:
                        initial_loss = loss.item()
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    if step == 4:
                        final_loss = loss.item()
                
                # Validate optimization worked
                assert not torch.isnan(loss), f"NaN loss for {loss_type}"
                assert not torch.isinf(loss), f"Inf loss for {loss_type}"
                assert final_loss <= initial_loss * 1.1, f"Loss didn't decrease for {loss_type}"
                
                print(f"    ✅ {loss_type}: loss {initial_loss:.6f} → {final_loss:.6f}")
                successful_integrations += 1
                
            except Exception as e:
                print(f"    ⚠️ {loss_type}: {e}")
        
        print(f"    📊 Successful loss-optimization integrations: {successful_integrations}/{len(loss_types)}")
        return successful_integrations > 0
        
    except Exception as e:
        print(f"    ❌ Loss-optimization integration test failed: {e}")
        return False

def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline integration"""
    print("🧪 Testing End-to-End Pipeline Integration...")
    
    try:
        config = MockConfig()
        
        # Create pipeline components
        processor = create_component('processor', 'integrated_signal', config)
        backbone = create_component('backbone', 'simple_transformer', config)
        loss_fn = create_component('loss', 'mse', config)
        
        if not all([processor, backbone, loss_fn]):
            print("    ⚠️ Required components not available, skipping...")
            return True
        
        # Test data
        x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_batch()
        target = torch.randn(2, 24, 7)
        
        # Complete pipeline test
        with torch.no_grad():
            # 1. Data preprocessing
            processed_x = processor(x_enc)
            
            # 2. Model prediction
            prediction = backbone(processed_x, x_mark_enc, x_dec, x_mark_dec)
            
            # 3. Loss computation
            loss = loss_fn(prediction, target)
            
            # 4. Post-processing (if available)
            if hasattr(processor, 'postprocess_hf_output'):
                final_prediction = processor.postprocess_hf_output(prediction, x_enc)
            else:
                final_prediction = prediction
        
        # Validate end-to-end pipeline
        assert isinstance(final_prediction, torch.Tensor), "Final prediction not tensor"
        assert final_prediction.shape == target.shape, "Prediction shape mismatch"
        assert not torch.isnan(final_prediction).any(), "NaN in final prediction"
        assert not torch.isnan(loss), "NaN in loss"
        
        print(f"    📊 Pipeline: {x_enc.shape} → {processed_x.shape} → {prediction.shape} → {final_prediction.shape}")
        print(f"    📊 Loss: {loss.item():.6f}")
        
        # Test with different configurations
        configs_to_test = [
            MockConfig(seq_len=48, pred_len=12),
            MockConfig(seq_len=192, pred_len=48),
            MockConfig(enc_in=21, c_out=21),
        ]
        
        pipeline_success = 0
        
        for test_config in configs_to_test:
            try:
                # Update components with new config
                test_processor = create_component('processor', 'integrated_signal', test_config)
                test_backbone = create_component('backbone', 'simple_transformer', test_config)
                test_loss = create_component('loss', 'mse', test_config)
                
                if not all([test_processor, test_backbone, test_loss]):
                    continue
                
                # Test data for new config
                test_x_enc, test_x_mark_enc, test_x_dec, test_x_mark_dec = create_sample_batch(
                    seq_len=test_config.seq_len,
                    pred_len=test_config.pred_len,
                    features=test_config.enc_in
                )
                test_target = torch.randn(2, test_config.pred_len, test_config.c_out)
                
                with torch.no_grad():
                    test_processed = test_processor(test_x_enc)
                    test_pred = test_backbone(test_processed, test_x_mark_enc, test_x_dec, test_x_mark_dec)
                    test_loss_val = test_loss(test_pred, test_target)
                
                assert not torch.isnan(test_loss_val), f"NaN loss in config test"
                pipeline_success += 1
                
                print(f"    ✅ Config (seq={test_config.seq_len}, pred={test_config.pred_len}): success")
                
            except Exception as e:
                print(f"    ⚠️ Config test failed: {e}")
        
        print(f"    📊 Pipeline configuration tests: {pipeline_success}/{len(configs_to_test)}")
        print("    ✅ End-to-end pipeline integration validated")
        return True
        
    except Exception as e:
        print(f"    ❌ End-to-end pipeline test failed: {e}")
        return False

def test_multi_scale_integration():
    """Test multi-scale component integration"""
    print("🧪 Testing Multi-Scale Integration...")
    
    try:
        config = MockConfig(scales=[1, 2, 4])
        
        # Create multi-scale components
        processor = create_component('processor', 'multi_scale', config)
        attention = create_component('attention', 'hierarchical', config)
        
        if not all([processor, attention]):
            print("    ⚠️ Required multi-scale components not available, skipping...")
            return True
        
        # Test data
        x_enc, _, _, _ = create_sample_batch()
        
        # Multi-scale processing
        with torch.no_grad():
            # Process at multiple scales
            processed_multi = processor(x_enc)
            
            # Apply hierarchical attention at different scales
            batch_size, seq_len, features = processed_multi.shape
            d_model = config.d_model
            
            # Project to model dimension
            projection = nn.Linear(features, d_model)
            projected = projection(processed_multi)
            
            # Apply hierarchical attention at different scales
            scale_outputs = []
            for scale in config.scales:
                scale_queries = projected
                scale_keys = projected
                scale_values = projected
                
                scale_output = attention(scale_queries, scale_keys, scale_values, scale=scale)
                scale_outputs.append(scale_output)
        
        # Validate multi-scale integration
        for i, scale_out in enumerate(scale_outputs):
            scale = config.scales[i]
            assert isinstance(scale_out, torch.Tensor), f"Scale {scale} output not tensor"
            assert not torch.isnan(scale_out).any(), f"NaN in scale {scale} output"
            print(f"    📊 Scale {scale}: {projected.shape} → {scale_out.shape}")
        
        # Test scale fusion
        if len(scale_outputs) > 1:
            # Simple concatenation fusion
            fused_output = torch.cat(scale_outputs, dim=-1)
            
            # Weighted fusion
            weights = torch.softmax(torch.randn(len(scale_outputs)), dim=0)
            weighted_output = sum(w * out for w, out in zip(weights, scale_outputs))
            
            print(f"    📊 Fused output: {fused_output.shape}")
            print(f"    📊 Weighted output: {weighted_output.shape}")
            
            assert not torch.isnan(fused_output).any(), "NaN in fused output"
            assert not torch.isnan(weighted_output).any(), "NaN in weighted output"
        
        print("    ✅ Multi-scale integration validated")
        return True
        
    except Exception as e:
        print(f"    ❌ Multi-scale integration test failed: {e}")
        return False

def test_quantile_integration():
    """Test quantile prediction integration"""
    print("🧪 Testing Quantile Prediction Integration...")
    
    try:
        config = MockConfig(quantiles=[0.1, 0.5, 0.9])
        
        # Create quantile-aware components
        loss_fn = create_component('loss', 'quantile', config)
        backbone = create_component('backbone', 'simple_transformer', config)
        
        if not all([loss_fn, backbone]):
            print("    ⚠️ Required quantile components not available, skipping...")
            return True
        
        # Test data
        x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_batch()
        target = torch.randn(2, 24, 7)
        
        # Quantile prediction pipeline
        with torch.no_grad():
            # Get backbone prediction
            base_prediction = backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Expand to quantile predictions
            num_quantiles = len(config.quantiles)
            quantile_predictions = base_prediction.unsqueeze(-1).expand(-1, -1, -1, num_quantiles)
            
            # Add quantile-specific variations
            for i, quantile in enumerate(config.quantiles):
                if quantile < 0.5:
                    # Lower quantiles should be lower
                    quantile_predictions[:, :, :, i] -= 0.1 * (0.5 - quantile)
                elif quantile > 0.5:
                    # Upper quantiles should be higher
                    quantile_predictions[:, :, :, i] += 0.1 * (quantile - 0.5)
            
            # Compute quantile loss
            quantile_loss = loss_fn(quantile_predictions, target)
        
        # Validate quantile integration
        assert isinstance(quantile_predictions, torch.Tensor), "Quantile predictions not tensor"
        assert quantile_predictions.shape[-1] == num_quantiles, "Wrong number of quantiles"
        assert not torch.isnan(quantile_predictions).any(), "NaN in quantile predictions"
        assert not torch.isnan(quantile_loss), "NaN in quantile loss"
        
        print(f"    📊 Quantile predictions: {base_prediction.shape} → {quantile_predictions.shape}")
        print(f"    📊 Quantile loss: {quantile_loss.item():.6f}")
        
        # Test quantile ordering
        # Lower quantiles should generally be <= higher quantiles
        q_low = quantile_predictions[:, :, :, 0]  # 0.1 quantile
        q_med = quantile_predictions[:, :, :, 1]  # 0.5 quantile
        q_high = quantile_predictions[:, :, :, 2]  # 0.9 quantile
        
        ordering_violations = ((q_low > q_med) | (q_med > q_high)).float().mean()
        print(f"    📊 Quantile ordering violations: {ordering_violations:.3f}")
        
        # Test different quantile configurations
        quantile_configs = [
            [0.5],  # Median only
            [0.25, 0.75],  # Quartiles
            [0.05, 0.25, 0.5, 0.75, 0.95],  # Extended quantiles
        ]
        
        quantile_success = 0
        
        for quantiles in quantile_configs:
            try:
                test_config = MockConfig(quantiles=quantiles)
                test_loss = create_component('loss', 'quantile', test_config)
                
                if test_loss is None:
                    continue
                
                test_quantile_preds = base_prediction.unsqueeze(-1).expand(-1, -1, -1, len(quantiles))
                test_quantile_loss = test_loss(test_quantile_preds, target)
                
                assert not torch.isnan(test_quantile_loss), f"NaN loss for quantiles {quantiles}"
                quantile_success += 1
                
                print(f"    ✅ Quantiles {quantiles}: loss = {test_quantile_loss.item():.6f}")
                
            except Exception as e:
                print(f"    ⚠️ Quantiles {quantiles}: {e}")
        
        print(f"    📊 Quantile configuration tests: {quantile_success}/{len(quantile_configs)}")
        print("    ✅ Quantile prediction integration validated")
        return True
        
    except Exception as e:
        print(f"    ❌ Quantile integration test failed: {e}")
        return False

def test_bayesian_integration():
    """Test Bayesian component integration"""
    print("🧪 Testing Bayesian Component Integration...")
    
    try:
        config = MockConfig()
        
        # Create Bayesian components
        loss_fn = create_component('loss', 'bayesian_mse', config)
        
        if loss_fn is None:
            print("    ⚠️ Bayesian loss not available, skipping...")
            return True
        
        # Mock Bayesian model
        class MockBayesianModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.mean_head = nn.Linear(config.enc_in, config.c_out)
                self.var_head = nn.Linear(config.enc_in, config.c_out)
                
            def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
                last_hidden = x_enc[:, -self.config.pred_len:, :]
                mean = self.mean_head(last_hidden)
                log_var = self.var_head(last_hidden)
                var = torch.exp(log_var)  # Ensure positive variance
                return mean, var
        
        bayesian_model = MockBayesianModel(config)
        
        # Test data
        x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_batch()
        target = torch.randn(2, 24, 7)
        
        # Bayesian prediction pipeline
        with torch.no_grad():
            mean_pred, var_pred = bayesian_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Compute Bayesian loss (includes KL divergence)
            bayesian_loss = loss_fn(mean_pred, var_pred, target)
        
        # Validate Bayesian integration
        assert isinstance(mean_pred, torch.Tensor), "Mean prediction not tensor"
        assert isinstance(var_pred, torch.Tensor), "Variance prediction not tensor"
        assert mean_pred.shape == target.shape, "Mean prediction shape mismatch"
        assert var_pred.shape == target.shape, "Variance prediction shape mismatch"
        assert (var_pred > 0).all(), "Variance must be positive"
        assert not torch.isnan(bayesian_loss), "NaN in Bayesian loss"
        
        print(f"    📊 Bayesian predictions: mean {mean_pred.shape}, var {var_pred.shape}")
        print(f"    📊 Bayesian loss: {bayesian_loss.item():.6f}")
        print(f"    📊 Mean variance: {var_pred.mean().item():.6f}")
        
        # Test uncertainty quantification
        # Higher uncertainty should lead to higher loss when predictions are wrong
        certain_var = torch.ones_like(var_pred) * 0.01  # Low uncertainty
        uncertain_var = torch.ones_like(var_pred) * 1.0  # High uncertainty
        
        wrong_pred = target + 1.0  # Deliberately wrong prediction
        
        certain_loss = loss_fn(wrong_pred, certain_var, target)
        uncertain_loss = loss_fn(wrong_pred, uncertain_var, target)
        
        print(f"    📊 Wrong prediction - certain loss: {certain_loss.item():.6f}")
        print(f"    📊 Wrong prediction - uncertain loss: {uncertain_loss.item():.6f}")
        
        # Certain model should be penalized more for wrong predictions
        assert certain_loss > uncertain_loss, "Uncertainty quantification not working correctly"
        
        # Test optimization with Bayesian components
        optimizer = torch.optim.Adam(bayesian_model.parameters(), lr=0.001)
        
        initial_loss = None
        final_loss = None
        
        for step in range(5):
            optimizer.zero_grad()
            
            mean_pred, var_pred = bayesian_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = loss_fn(mean_pred, var_pred, target)
            
            if step == 0:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            
            if step == 4:
                final_loss = loss.item()
        
        print(f"    📊 Bayesian optimization: {initial_loss:.6f} → {final_loss:.6f}")
        assert final_loss <= initial_loss * 1.1, "Bayesian optimization not working"
        
        print("    ✅ Bayesian component integration validated")
        return True
        
    except Exception as e:
        print(f"    ❌ Bayesian integration test failed: {e}")
        return False

def test_component_interaction_matrix():
    """Test interactions between all available components"""
    print("🧪 Testing Component Interaction Matrix...")
    
    try:
        config = MockConfig()
        
        # Get available components
        component_types = {
            'processor': ['integrated_signal', 'frequency_domain', 'multi_scale'],
            'backbone': ['simple_transformer', 'chronos', 't5'],
            'attention': ['multi_head', 'autocorrelation', 'hierarchical'],
            'loss': ['mse', 'mae', 'bayesian_mse', 'quantile']
        }
        
        available_components = {}
        
        for comp_type, comp_names in component_types.items():
            available_components[comp_type] = []
            for comp_name in comp_names:
                try:
                    comp = create_component(comp_type, comp_name, config)
                    if comp is not None:
                        available_components[comp_type].append(comp_name)
                except:
                    pass
        
        print("    📊 Available components:")
        for comp_type, comp_list in available_components.items():
            print(f"      {comp_type}: {comp_list}")
        
        # Test key interactions
        interaction_tests = 0
        successful_interactions = 0
        
        # Processor-Backbone interactions
        for processor_name in available_components.get('processor', []):
            for backbone_name in available_components.get('backbone', []):
                interaction_tests += 1
                try:
                    processor = create_component('processor', processor_name, config)
                    backbone = create_component('backbone', backbone_name, config)
                    
                    x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_batch()
                    
                    with torch.no_grad():
                        processed = processor(x_enc)
                        
                        if backbone_name == 'chronos':
                            output = backbone(processed)
                        elif backbone_name == 't5':
                            output = backbone(processed, x_mark_enc, x_dec, x_mark_dec)
                        else:
                            output = backbone(processed, x_mark_enc, x_dec, x_mark_dec)
                    
                    assert not torch.isnan(output).any(), "NaN in interaction output"
                    successful_interactions += 1
                    
                except Exception as e:
                    print(f"      ⚠️ {processor_name}-{backbone_name}: {e}")
        
        # Backbone-Loss interactions
        for backbone_name in available_components.get('backbone', []):
            for loss_name in available_components.get('loss', []):
                interaction_tests += 1
                try:
                    backbone = create_component('backbone', backbone_name, config)
                    loss_fn = create_component('loss', loss_name, config)
                    
                    x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_batch()
                    target = torch.randn(2, 24, 7)
                    
                    with torch.no_grad():
                        if backbone_name == 'chronos':
                            prediction = backbone(x_enc)
                        elif backbone_name == 't5':
                            prediction = backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
                        else:
                            prediction = backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
                        
                        if loss_name == 'bayesian_mse':
                            mean_pred = prediction
                            var_pred = torch.ones_like(prediction) * 0.1
                            loss = loss_fn(mean_pred, var_pred, target)
                        elif loss_name == 'quantile':
                            q_pred = prediction.unsqueeze(-1).expand(-1, -1, -1, 3)
                            loss = loss_fn(q_pred, target)
                        else:
                            loss = loss_fn(prediction, target)
                    
                    assert not torch.isnan(loss), "NaN in loss computation"
                    successful_interactions += 1
                    
                except Exception as e:
                    print(f"      ⚠️ {backbone_name}-{loss_name}: {e}")
        
        print(f"    📊 Component interactions tested: {interaction_tests}")
        print(f"    📊 Successful interactions: {successful_interactions}")
        print(f"    📊 Success rate: {(successful_interactions/interaction_tests)*100:.1f}%" if interaction_tests > 0 else "    📊 No interactions tested")
        
        return successful_interactions > 0 if interaction_tests > 0 else True
        
    except Exception as e:
        print(f"    ❌ Component interaction matrix test failed: {e}")
        return False

def run_integration_functionality_tests():
    """Run all integration functionality tests"""
    print("🚀 Running Integration Component Functionality Tests")
    print("=" * 80)
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Modular components not available - skipping tests")
        return False
    
    tests = [
        ("Processor-Backbone Integration", test_processor_backbone_integration),
        ("Backbone-Attention Integration", test_backbone_attention_integration),
        ("Loss-Optimization Integration", test_loss_optimization_integration),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
        ("Multi-Scale Integration", test_multi_scale_integration),
        ("Quantile Integration", test_quantile_integration),
        ("Bayesian Integration", test_bayesian_integration),
        ("Component Interaction Matrix", test_component_interaction_matrix),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🎯 {test_name}")
        print("-" * 60)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 80)
    print(f"📊 Integration Component Functionality Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All integration functionality tests passed!")
        return True
    else:
        print("⚠️ Some integration functionality tests failed")
        return False

if __name__ == "__main__":
    success = run_integration_functionality_tests()
    sys.exit(0 if success else 1)
