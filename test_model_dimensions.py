#!/usr/bin/env python3
"""
Unit Test for Model Input/Output Dimensions and Covariate Behavior

This script tests how different TimesNet models handle:
1. Input dimensions (targets + dynamic covariates + static covariates)
2. Output dimensions based on features mode ('M', 'MS', 'S')
3. Future covariate behavior (should model predict covariates or ignore them?)
4. Target vs covariate separation in outputs

Test Case:
- 2 target variables (e.g., Open, Close prices)
- 2 dynamic covariates (e.g., Volume, Market_Cap) 
- 1 static covariate (encoded as time features)
- Total input features: 5 (2 targets + 2 dynamic + 1 static)
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'layers'))

from models.TimesNet import Model as TimesNet


class ModelDimensionTester:
    def __init__(self):
        """Initialize the tester with a simple synthetic dataset"""
        self.seq_len = 96    # Input sequence length
        self.pred_len = 24   # Prediction length
        self.label_len = 48  # Label length
        self.batch_size = 4  # Small batch for testing
        
        # Feature composition
        self.n_targets = 2          # Target variables (Open, Close)
        self.n_dynamic_covariates = 2   # Dynamic covariates (Volume, Market_Cap)
        self.n_static_features = 1      # Static/time features
        self.total_features = self.n_targets + self.n_dynamic_covariates + self.n_static_features
        
        print(f"🧪 MODEL DIMENSION TESTER")
        print(f"=" * 60)
        print(f"📊 Test Dataset Composition:")
        print(f"   - Target variables: {self.n_targets} (Open, Close)")
        print(f"   - Dynamic covariates: {self.n_dynamic_covariates} (Volume, Market_Cap)")
        print(f"   - Static features: {self.n_static_features} (Time encoding)")
        print(f"   - Total input features: {self.total_features}")
        print(f"📏 Sequence Configuration:")
        print(f"   - Input sequence length: {self.seq_len}")
        print(f"   - Prediction length: {self.pred_len}")
        print(f"   - Label length: {self.label_len}")
        print(f"   - Batch size: {self.batch_size}")
        print(f"=" * 60)
    
    def generate_test_data(self):
        """Generate synthetic test data with known structure"""
        # Input data: [batch_size, seq_len, features]
        batch_x = torch.randn(self.batch_size, self.seq_len, self.total_features)
        
        # Target data: [batch_size, label_len + pred_len, features]
        total_target_len = self.label_len + self.pred_len
        batch_y = torch.randn(self.batch_size, total_target_len, self.total_features)
        
        # Time markings (for business day frequency: DayOfWeek, DayOfMonth, DayOfYear = 3 features)
        batch_x_mark = torch.randn(self.batch_size, self.seq_len, 3)  # 3 time features for business day
        batch_y_mark = torch.randn(self.batch_size, total_target_len, 3)
        
        print(f"📦 Generated Test Data:")
        print(f"   - batch_x shape: {batch_x.shape} (input sequences)")
        print(f"   - batch_y shape: {batch_y.shape} (target sequences)")
        print(f"   - batch_x_mark shape: {batch_x_mark.shape} (input time features)")
        print(f"   - batch_y_mark shape: {batch_y_mark.shape} (target time features)")
        
        return batch_x, batch_y, batch_x_mark, batch_y_mark
    
    def create_model(self, features_mode='M', c_out=None):
        """Create TimesNet model with specified configuration"""
        
        # Automatically determine c_out based on features mode if not specified
        if c_out is None:
            if features_mode == 'M':
                c_out = self.total_features  # Predict all features
            elif features_mode == 'MS':
                c_out = 1  # Multivariate to Univariate
            elif features_mode == 'S':
                c_out = 1  # Univariate
          # Model configuration
        args = type('Args', (), {
            'task_name': 'long_term_forecast',
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'enc_in': self.total_features,
            'dec_in': self.total_features,
            'c_out': c_out,
            'd_model': 64,
            'd_ff': 128,
            'n_heads': 4,
            'e_layers': 2,
            'd_layers': 1,
            'factor': 1,
            'distil': False,
            'dropout': 0.1,
            'embed': 'timeF',
            'activation': 'gelu',
            'output_attention': False,
            'top_k': 5,
            'num_kernels': 5,
            'moving_avg': 25,
            'features': features_mode,
            'freq': 'b'  # Business day frequency
        })()
        
        model = TimesNet(args)
        
        print(f"🔧 Created TimesNet Model:")
        print(f"   - Features mode: '{features_mode}'")
        print(f"   - Input features (enc_in): {args.enc_in}")
        print(f"   - Output features (c_out): {args.c_out}")
        print(f"   - Model dimension: {args.d_model}")
        print(f"   - Encoder layers: {args.e_layers}")
        
        return model, args
    
    def test_model_forward(self, features_mode='M', c_out=None):
        """Test model forward pass and analyze dimensions"""
        print(f"\n🚀 TESTING FEATURES MODE: '{features_mode}'")
        print(f"=" * 40)
        
        # Create model and data
        model, args = self.create_model(features_mode, c_out)
        batch_x, batch_y, batch_x_mark, batch_y_mark = self.generate_test_data()
        
        # Prepare decoder input (standard for forecasting models)
        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float()
        
        print(f"🔄 Decoder input shape: {dec_inp.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            try:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                print(f"✅ Forward pass successful!")
                print(f"📤 Model output shape: {outputs.shape}")
                print(f"📊 Expected target shape: {batch_y[:, -self.pred_len:, :].shape}")
                
                # Analyze output dimensions
                batch_size_out, seq_len_out, features_out = outputs.shape
                
                print(f"\n📋 DIMENSION ANALYSIS:")
                print(f"   - Output batch size: {batch_size_out} (expected: {self.batch_size})")
                print(f"   - Output sequence length: {seq_len_out} (expected: {self.pred_len})")
                print(f"   - Output features: {features_out}")
                
                # Analyze what the model is predicting
                if features_mode == 'M':
                    print(f"   ➜ Mode 'M': Model predicts ALL {features_out} features")
                    print(f"     - Targets (first {self.n_targets}): {outputs[0, 0, :self.n_targets].detach().numpy()}")
                    print(f"     - Dynamic covariates (next {self.n_dynamic_covariates}): {outputs[0, 0, self.n_targets:self.n_targets+self.n_dynamic_covariates].detach().numpy()}")
                    if features_out > self.n_targets + self.n_dynamic_covariates:
                        print(f"     - Static features (remaining): {outputs[0, 0, self.n_targets+self.n_dynamic_covariates:].detach().numpy()}")
                
                elif features_mode in ['MS', 'S']:
                    print(f"   ➜ Mode '{features_mode}': Model predicts {features_out} feature(s)")
                    print(f"     - Single output: {outputs[0, 0, 0].detach().numpy()}")
                    if features_out > 1:
                        print(f"     - Additional outputs: {outputs[0, 0, 1:].detach().numpy()}")
                
                # Check if output dimensions match expectations
                dimension_check = {
                    'batch_size_correct': batch_size_out == self.batch_size,
                    'sequence_length_correct': seq_len_out == self.pred_len,
                    'features_expected': features_out == args.c_out
                }
                
                print(f"\n✅ DIMENSION VALIDATION:")
                for check, result in dimension_check.items():
                    status = "✅ PASS" if result else "❌ FAIL"
                    print(f"   - {check.replace('_', ' ').title()}: {status}")
                
                return outputs, dimension_check
                
            except Exception as e:
                print(f"❌ Forward pass failed!")
                print(f"🚨 Error: {e}")
                return None, {'error': str(e)}
    
    def test_covariate_behavior(self):
        """Test how the model handles future covariates vs targets"""
        print(f"\n🔍 TESTING COVARIATE BEHAVIOR")
        print(f"=" * 50)
        
        # Test with different scenarios
        scenarios = [
            ('M', None, "Predict all features (targets + covariates)"),
            ('MS', 1, "Multivariate input → Single target output"),
            ('MS', 2, "Multivariate input → Multiple target output"),
            ('S', 1, "Univariate mode")
        ]
        
        results = {}
        
        for features_mode, c_out, description in scenarios:
            print(f"\n📊 Scenario: {description}")
            print(f"   Features mode: '{features_mode}', c_out: {c_out}")
            
            outputs, check = self.test_model_forward(features_mode, c_out)
            results[f"{features_mode}_{c_out}"] = {
                'outputs': outputs,
                'validation': check,
                'description': description
            }
        
        return results
    
    def analyze_covariate_predictions(self, results):
        """Analyze whether the model predicts future covariates correctly"""
        print(f"\n🎯 COVARIATE PREDICTION ANALYSIS")
        print(f"=" * 50)
        
        for scenario_key, result in results.items():
            if 'error' in result['validation']:
                continue
                
            outputs = result['outputs']
            if outputs is None:
                continue
                
            print(f"\n📈 {result['description']}:")
            
            # Extract features from output
            if outputs.shape[-1] >= self.total_features:  # Full multivariate output
                target_predictions = outputs[0, 0, :self.n_targets]
                covariate_predictions = outputs[0, 0, self.n_targets:self.n_targets+self.n_dynamic_covariates]
                
                print(f"   - Target predictions: {target_predictions.detach().numpy()}")
                print(f"   - Covariate predictions: {covariate_predictions.detach().numpy()}")
                print(f"   ⚠️  Model is predicting future covariates (this may not be desired!)")
                
            elif outputs.shape[-1] == 1:  # Single output
                print(f"   - Single prediction: {outputs[0, 0, 0].detach().numpy()}")
                print(f"   ✅ Model outputs single value (targets only)")
                
            else:  # Partial multivariate
                print(f"   - Predictions: {outputs[0, 0, :].detach().numpy()}")
                print(f"   🤔 Model outputs {outputs.shape[-1]} features")
    
    def test_covariate_passthrough(self, features_mode='M'):
        """Test if model passes through future covariate values unchanged"""
        print(f"\n🔍 TESTING COVARIATE PASSTHROUGH")
        print(f"=" * 50)
          # Create model
        model, args = self.create_model(features_mode)
        model.eval()  # Set to evaluation mode
        
        # Generate test data with SPECIFIC recognizable values for future covariates
        batch_x = torch.randn(self.batch_size, self.seq_len, self.total_features)
        
        # Create future data with recognizable covariate patterns
        total_target_len = self.label_len + self.pred_len
        batch_y = torch.zeros(self.batch_size, total_target_len, self.total_features)
        
        # Fill future period with recognizable covariate values
        for t in range(self.pred_len):
            # Future targets: random (these will be predicted)
            batch_y[:, self.label_len + t, :self.n_targets] = torch.randn(self.batch_size, self.n_targets)
            
            # Future covariates: specific recognizable pattern
            # Covariate 1: timestep value (1.0, 2.0, 3.0, ...)
            batch_y[:, self.label_len + t, self.n_targets] = float(t + 1.0)
            # Covariate 2: timestep * 10 (10.0, 20.0, 30.0, ...)  
            batch_y[:, self.label_len + t, self.n_targets + 1] = float((t + 1.0) * 10.0)
            
            # Static feature: constant value
            if self.n_static_features > 0:
                batch_y[:, self.label_len + t, -1] = 99.0
        
        # Label period (overlap): copy some historical data
        batch_y[:, :self.label_len, :] = batch_x[:, -self.label_len:, :]
        
        # Time features
        batch_x_mark = torch.randn(self.batch_size, self.seq_len, 3)
        batch_y_mark = torch.randn(self.batch_size, total_target_len, 3)
        
        # Extract the input future covariate values for comparison
        input_future_covariates = batch_y[:, -self.pred_len:, self.n_targets:self.n_targets + self.n_dynamic_covariates].clone()
        
        print(f"📊 Test Setup:")
        print(f"   - Prediction length: {self.pred_len}")
        print(f"   - Future covariate pattern for Covariate 1: {input_future_covariates[0, :, 0].tolist()}")
        print(f"   - Future covariate pattern for Covariate 2: {input_future_covariates[0, :, 1].tolist()}")
        
        try:
            # Forward pass
            with torch.no_grad():
                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float()
                
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Extract output covariate values for the prediction period
                output_future_covariates = outputs[:, -self.pred_len:, self.n_targets:self.n_targets + self.n_dynamic_covariates]
                
                print(f"\n📤 Model Output:")
                print(f"   - Output covariate 1: {output_future_covariates[0, :, 0].tolist()}")
                print(f"   - Output covariate 2: {output_future_covariates[0, :, 1].tolist()}")
                
                # Compare input vs output covariates
                covariate_diff = torch.abs(input_future_covariates - output_future_covariates)
                max_diff = torch.max(covariate_diff).item()
                mean_diff = torch.mean(covariate_diff).item()
                
                print(f"\n🔍 COVARIATE PASSTHROUGH ANALYSIS:")
                print(f"   - Maximum difference: {max_diff:.6f}")
                print(f"   - Mean difference: {mean_diff:.6f}")
                
                # Determine if it's passthrough (very small differences due to floating point)
                is_passthrough = max_diff < 1e-4
                
                if is_passthrough:
                    print(f"   ✅ PASSTHROUGH: Model copies future covariate inputs to outputs")
                    print(f"   ➜ This is GOOD for known future covariates!")
                else:
                    print(f"   ❌ TRANSFORMATION: Model modifies future covariate values")
                    print(f"   ➜ This may be problematic if covariates are known!")
                
                return {
                    'is_passthrough': is_passthrough,
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'input_covariates': input_future_covariates.numpy(),
                    'output_covariates': output_future_covariates.numpy()
                }
                
        except Exception as e:
            print(f"❌ Covariate passthrough test failed!")
            print(f"🚨 Error: {str(e)}")
            return None

    def test_scaling_behavior(self):
        """Test if model is applying scaling/normalization to covariate values"""
        print(f"\n🔍 TESTING SCALING/NORMALIZATION BEHAVIOR")
        print(f"=" * 60)
        
        # Create model
        model, args = self.create_model('M')
        model.eval()
        
        # Test different scaling scenarios
        scenarios = [
            {"name": "Small positive values", "cov1": [1.0, 2.0, 3.0, 4.0], "cov2": [5.0, 6.0, 7.0, 8.0]},
            {"name": "Large positive values", "cov1": [100.0, 200.0, 300.0, 400.0], "cov2": [500.0, 600.0, 700.0, 800.0]},
            {"name": "Zero centered", "cov1": [-2.0, -1.0, 0.0, 1.0], "cov2": [-4.0, -2.0, 0.0, 2.0]},
            {"name": "Constant values", "cov1": [10.0, 10.0, 10.0, 10.0], "cov2": [20.0, 20.0, 20.0, 20.0]}
        ]
        
        for scenario in scenarios:
            print(f"\n📊 Scenario: {scenario['name']}")
            print(f"   Input cov1: {scenario['cov1']}")
            print(f"   Input cov2: {scenario['cov2']}")
            
            # Create test data with specific covariate values
            batch_x = torch.randn(1, self.seq_len, self.total_features)
            total_target_len = self.label_len + 4  # Use pred_len=4 for this test
            batch_y = torch.zeros(1, total_target_len, self.total_features)
            
            # Set the specific covariate values for future period
            for t in range(4):
                # Random targets
                batch_y[:, self.label_len + t, :self.n_targets] = torch.randn(1, self.n_targets)
                # Specific covariate values
                batch_y[:, self.label_len + t, self.n_targets] = scenario['cov1'][t]
                batch_y[:, self.label_len + t, self.n_targets + 1] = scenario['cov2'][t]
                # Static feature
                batch_y[:, self.label_len + t, -1] = 99.0
            
            # Label period
            batch_y[:, :self.label_len, :] = batch_x[:, -self.label_len:, :]
            
            # Time features
            batch_x_mark = torch.randn(1, self.seq_len, 3)
            batch_y_mark = torch.randn(1, total_target_len, 3)
            
            try:
                with torch.no_grad():
                    # Prepare decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -4:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float()
                    
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    # Extract output covariates
                    output_cov1 = outputs[0, -4:, self.n_targets].tolist()
                    output_cov2 = outputs[0, -4:, self.n_targets + 1].tolist()
                    
                    print(f"   Output cov1: {[f'{x:.3f}' for x in output_cov1]}")
                    print(f"   Output cov2: {[f'{x:.3f}' for x in output_cov2]}")
                    
                    # Calculate some scaling metrics
                    input_cov1_mean = sum(scenario['cov1']) / len(scenario['cov1'])
                    input_cov1_std = (sum([(x - input_cov1_mean)**2 for x in scenario['cov1']]) / len(scenario['cov1']))**0.5
                    
                    output_cov1_mean = sum(output_cov1) / len(output_cov1)
                    output_cov1_std = (sum([(x - output_cov1_mean)**2 for x in output_cov1]) / len(output_cov1))**0.5
                    
                    print(f"   Input stats - Mean: {input_cov1_mean:.3f}, Std: {input_cov1_std:.3f}")
                    print(f"   Output stats - Mean: {output_cov1_mean:.3f}, Std: {output_cov1_std:.3f}")
                    
                    # Check if output values are in typical normalized range
                    output_range = max(max(output_cov1), max(output_cov2)) - min(min(output_cov1), min(output_cov2))
                    print(f"   Output range: {output_range:.3f}")
                    
                    if output_range < 5.0:  # Typical for normalized data
                        print(f"   🔍 Likely normalized/scaled output")
                    else:
                        print(f"   🔍 Raw/unscaled output")
                        
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        # Test scaling hypotheses
        print(f"\n🧮 SCALING HYPOTHESIS TESTING")
        print(f"=" * 40)
        print(f"Common scaling methods:")
        print(f"1. Z-score: (x - mean) / std")
        print(f"2. Min-Max: (x - min) / (max - min)")
        print(f"3. Tanh normalization: tanh(x/scale)")
        print(f"4. Layer normalization")
        print(f"5. Model learns its own scaling")
    
    def test_target_order_and_scaling(self):
        """Test the exact order and scaling of target outputs"""
        print(f"\n🎯 TESTING TARGET ORDER AND SCALING")
        print(f"=" * 60)
        
        # Create model
        model, args = self.create_model('M')
        model.eval()
        
        # Create test data with VERY specific target values to track them
        batch_x = torch.randn(1, self.seq_len, self.total_features)
        total_target_len = self.label_len + 4  # Use pred_len=4 for clarity
        batch_y = torch.zeros(1, total_target_len, self.total_features)
        
        # Set VERY recognizable target values for future period
        target_patterns = {
            'Target_0 (Open)': [100.0, 200.0, 300.0, 400.0],
            'Target_1 (Close)': [1000.0, 2000.0, 3000.0, 4000.0],
        }
        
        covariate_patterns = {
            'Covariate_0 (Volume)': [10.0, 20.0, 30.0, 40.0],
            'Covariate_1 (Market_Cap)': [50.0, 60.0, 70.0, 80.0],
        }
        
        print(f"📊 INPUT TEST PATTERNS:")
        for name, pattern in target_patterns.items():
            print(f"   {name}: {pattern}")
        for name, pattern in covariate_patterns.items():
            print(f"   {name}: {pattern}")
        print(f"   Static feature: [99.0, 99.0, 99.0, 99.0]")
        
        # Fill future period with specific patterns
        for t in range(4):
            # Targets: position 0 and 1
            batch_y[:, self.label_len + t, 0] = target_patterns['Target_0 (Open)'][t]
            batch_y[:, self.label_len + t, 1] = target_patterns['Target_1 (Close)'][t]
            
            # Covariates: position 2 and 3  
            batch_y[:, self.label_len + t, 2] = covariate_patterns['Covariate_0 (Volume)'][t]
            batch_y[:, self.label_len + t, 3] = covariate_patterns['Covariate_1 (Market_Cap)'][t]
            
            # Static: position 4
            batch_y[:, self.label_len + t, 4] = 99.0
        
        # Label period: copy some historical data
        batch_y[:, :self.label_len, :] = batch_x[:, -self.label_len:, :]
        
        # Time features
        batch_x_mark = torch.randn(1, self.seq_len, 3)
        batch_y_mark = torch.randn(1, total_target_len, 3)
        
        try:
            with torch.no_grad():
                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -4:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float()
                
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                print(f"\n📤 MODEL OUTPUT ANALYSIS:")
                print(f"Output shape: {outputs.shape}")
                
                # Extract outputs for each position
                for pos in range(5):
                    output_values = outputs[0, -4:, pos].tolist()
                    print(f"   Position {pos}: {[f'{x:.3f}' for x in output_values]}")
                
                # Compare with input patterns to identify which position corresponds to what
                print(f"\n🔍 PATTERN MATCHING ANALYSIS:")
                
                # Check if any output position correlates with input targets
                for target_name, input_pattern in target_patterns.items():
                    print(f"\n   Looking for {target_name} pattern {input_pattern}:")
                    for pos in range(5):
                        output_values = outputs[0, -4:, pos].tolist()
                        
                        # Check correlation (not exact match due to normalization)
                        input_trend = [input_pattern[i+1] - input_pattern[i] for i in range(3)]
                        output_trend = [output_values[i+1] - output_values[i] for i in range(3)]
                        
                        # Simple correlation check
                        correlation = sum(a*b for a,b in zip(input_trend, output_trend))
                        print(f"     Position {pos}: trend correlation = {correlation:.3f}")
                
                # Test target scaling by checking if outputs preserve relative ratios
                print(f"\n📏 SCALING ANALYSIS:")
                target_0_out = outputs[0, -4:, 0].tolist()
                target_1_out = outputs[0, -4:, 1].tolist()
                
                # Check if relative scaling is preserved
                input_ratio_01 = target_patterns['Target_1 (Close)'][0] / target_patterns['Target_0 (Open)'][0]  # 1000/100 = 10
                output_ratio_01 = target_1_out[0] / target_0_out[0] if target_0_out[0] != 0 else 0
                
                print(f"   Input ratio (Target_1/Target_0) for t=0: {input_ratio_01:.3f}")
                print(f"   Output ratio (pos_1/pos_0) for t=0: {output_ratio_01:.3f}")
                
                if abs(input_ratio_01 - output_ratio_01) < 1.0:
                    print(f"   ✅ Ratios preserved - minimal scaling applied to targets")
                else:
                    print(f"   ⚠️  Ratios not preserved - targets are normalized/scaled")
                
                # Check absolute scaling
                input_target_0_range = max(target_patterns['Target_0 (Open)']) - min(target_patterns['Target_0 (Open)'])
                output_pos_0_range = max(target_0_out) - min(target_0_out)
                
                print(f"   Input Target_0 range: {input_target_0_range}")
                print(f"   Output position_0 range: {output_pos_0_range:.3f}")
                
                scaling_factor = input_target_0_range / output_pos_0_range if output_pos_0_range != 0 else 0
                print(f"   Approximate scaling factor: {scaling_factor:.1f}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print(f"\n🎯 CONCLUSIONS:")
        print(f"1. Verify which output positions correspond to your OHLC targets")
        print(f"2. Check if targets are scaled (normalize back if needed)")
        print(f"3. Ensure loss calculation uses correct positions and scaling")
    
    def test_real_data_structure(self):
        """Test with actual financial data structure to verify feature ordering and scaling"""
        print(f"\n🔍 TESTING REAL DATA STRUCTURE AND SCALING")
        print(f"=" * 60)
        
        # Check if prepared data exists
        data_path = './data/prepared_financial_data.csv'
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            print(f"   Please run data preparation first")
            return None
            
        try:
            import pandas as pd
            
            # Load the actual prepared data
            print(f"📊 Loading actual financial data...")
            df = pd.read_csv(data_path)
            
            print(f"   - Data shape: {df.shape}")
            print(f"   - Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
            
            # Identify target columns (should be first 4: OHLC)
            expected_targets = ['log_Open', 'log_High', 'log_Low', 'log_Close']
            actual_targets = [col for col in expected_targets if col in df.columns]
            
            print(f"   - Expected target columns: {expected_targets}")
            print(f"   - Found target columns: {actual_targets}")
            
            if len(actual_targets) != 4:
                print(f"⚠️  Warning: Found {len(actual_targets)} targets, expected 4")
                return None
                
            # Check if targets are first 4 columns
            first_4_cols = list(df.columns[:4])
            targets_are_first = first_4_cols == actual_targets
            
            print(f"   - First 4 columns: {first_4_cols}")
            print(f"   - Targets are first 4: {'✅ YES' if targets_are_first else '❌ NO'}")
            
            # Analyze scaling (check if data is already scaled)
            target_stats = df[actual_targets].describe()
            print(f"\n📏 Target Statistics (should be scaled if data prep was applied):")
            for col in actual_targets:
                mean_val = target_stats.loc['mean', col]
                std_val = target_stats.loc['std', col]
                min_val = target_stats.loc['min', col] 
                max_val = target_stats.loc['max', col]
                print(f"   - {col}: mean={mean_val:.3f}, std={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")
                
                # Check if it looks scaled (near 0 mean, ~1 std)
                is_scaled = abs(mean_val) < 0.1 and 0.8 < std_val < 1.2
                print(f"     {'✅ Appears scaled' if is_scaled else '❌ Not scaled'}")
            
            # Test with small sample of real data
            print(f"\n🧪 Testing model with real data sample...")
            
            # Create model
            model, args = self.create_model('M')
            model.eval()
            
            # Use actual data dimensions
            n_features = len(df.columns)
            args.enc_in = n_features
            args.dec_in = n_features
            args.c_out = n_features
            
            # Take a small sample for testing
            sample_size = min(200, len(df))
            sample_data = df.iloc[:sample_size].values  # Convert to numpy array
            
            # Create test tensors with real data structure
            batch_size = 2
            seq_len = 96
            pred_len = 24
            
            if sample_size < seq_len + pred_len:
                print(f"❌ Not enough data for test (need {seq_len + pred_len}, have {sample_size})")
                return None
                
            # Extract sequences from real data
            batch_x = torch.tensor(sample_data[:seq_len], dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
            
            # Create target sequence (label + prediction period)
            target_start = seq_len - 48  # label_len = 48
            batch_y = torch.tensor(sample_data[target_start:target_start + 72], dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
            
            # Time features (business day frequency = 3 features)
            batch_x_mark = torch.randn(batch_size, seq_len, 3)
            batch_y_mark = torch.randn(batch_size, 72, 3)
            
            print(f"   - Real data batch_x shape: {batch_x.shape}")
            print(f"   - Real data batch_y shape: {batch_y.shape}")
            print(f"   - Features: {n_features}")
            
            # Test forward pass
            try:
                with torch.no_grad():
                    # Prepare decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :48, :], dec_inp], dim=1).float()
                    
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    print(f"   ✅ Forward pass successful!")
                    print(f"   - Output shape: {outputs.shape}")
                    
                    # Check target consistency
                    target_outputs = outputs[:, -pred_len:, :4]  # First 4 features
                    target_y = batch_y[:, -pred_len:, :4]        # First 4 features
                    
                    print(f"   - Target output shape: {target_outputs.shape}")
                    print(f"   - Target ground truth shape: {target_y.shape}")
                    
                    # Check scaling consistency
                    output_stats = {
                        'mean': torch.mean(target_outputs).item(),
                        'std': torch.std(target_outputs).item(),
                        'min': torch.min(target_outputs).item(),
                        'max': torch.max(target_outputs).item()
                    }
                    
                    target_stats = {
                        'mean': torch.mean(target_y).item(),
                        'std': torch.std(target_y).item(),
                        'min': torch.min(target_y).item(),
                        'max': torch.max(target_y).item()
                    }
                    
                    print(f"\n📊 SCALING CONSISTENCY CHECK:")
                    print(f"   Target outputs - mean: {output_stats['mean']:.3f}, std: {output_stats['std']:.3f}")
                    print(f"   Target ground truth - mean: {target_stats['mean']:.3f}, std: {target_stats['std']:.3f}")
                    
                    # Check if distributions are similar (good for loss calculation)
                    mean_diff = abs(output_stats['mean'] - target_stats['mean'])
                    std_diff = abs(output_stats['std'] - target_stats['std'])
                    
                    scaling_consistent = mean_diff < 0.5 and std_diff < 0.5
                    print(f"   Scaling consistency: {'✅ GOOD' if scaling_consistent else '❌ POOR'}")
                    
                    if scaling_consistent:
                        print(f"   ➜ Loss calculation should work correctly")
                    else:
                        print(f"   ➜ May need scaling adjustment in loss calculation")
                        
                    return {
                        'targets_are_first': targets_are_first,
                        'scaling_consistent': scaling_consistent,
                        'n_features': n_features,
                        'output_stats': output_stats,
                        'target_stats': target_stats
                    }
                    
            except Exception as e:
                print(f"   ❌ Forward pass failed: {str(e)}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading real data: {str(e)}")
            return None

    # ...existing code...
def main():
    """Run the comprehensive model dimension test"""
    print("🧪 TIMESNET MODEL DIMENSION AND COVARIATE BEHAVIOR TEST")
    print("=" * 80)
    
    tester = ModelDimensionTester()
    
    # Test different scenarios
    results = tester.test_covariate_behavior()
    
    # Analyze covariate behavior
    tester.analyze_covariate_predictions(results)
      # Test covariate passthrough behavior
    print(f"\n🔍 COVARIATE PASSTHROUGH TEST")
    print(f"=" * 80)
    print("Testing if model passes through future covariate inputs unchanged...")
    
    passthrough_result = tester.test_covariate_passthrough('M')
    
    if passthrough_result:
        if passthrough_result['is_passthrough']:
            print(f"\n✅ CONCLUSION: Model correctly passes through future covariates")
            print(f"   ➜ This is ideal behavior for known future covariates!")
        else:
            print(f"\n⚠️  CONCLUSION: Model transforms future covariates")
            print(f"   ➜ This may be problematic if covariates are known/planned values")
            
            # Run scaling test to understand the transformation
            tester.test_scaling_behavior()
    
    # Test with real financial data structure
    print(f"\n🏦 REAL FINANCIAL DATA TEST")
    print(f"=" * 80)
    print("Testing with actual prepared financial data...")
    
    real_data_result = tester.test_real_data_structure()
    
    if real_data_result:
        print(f"\n✅ REAL DATA ANALYSIS COMPLETE")
        if real_data_result['targets_are_first'] and real_data_result['scaling_consistent']:
            print(f"   ➜ Your current training setup should work correctly!")
        else:
            print(f"   ⚠️  Issues found that may need fixing in training script")
    else:
        print(f"\n❌ Could not test with real data - check data preparation")
    
    print(f"\n🎯 SUMMARY AND RECOMMENDATIONS")
    print(f"=" * 50)
    print(f"1. Use 'M' mode if you want to predict ALL features (including covariates)")
    print(f"2. Use 'MS' mode if you want multivariate input but target-only output")
    print(f"3. Use 'S' mode for pure univariate forecasting")
    print(f"4. For financial forecasting:")
    print(f"   - If predicting OHLC only: Use 'MS' mode with c_out=4")
    print(f"   - If using covariates: Ensure they're NOT predicted in future")
    print(f"   - Dynamic covariates should be provided externally for future periods")
    
    if passthrough_result and not passthrough_result['is_passthrough']:
        print(f"5. ⚠️  IMPORTANT: Model modifies future covariates - consider this in your pipeline")


if __name__ == "__main__":
    main()
