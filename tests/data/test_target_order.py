"""
Quick test to verify target order and scaling in TimesNet outputs
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.TimesNet import TimesNet
import argparse

def test_target_order_and_scaling():
    """Test the exact order and scaling of target outputs"""
    print("TARGET TESTING TARGET ORDER AND SCALING")
    print("=" * 60)
    
    # Model configuration
    args = type('Args', (), {
        'task_name': 'long_term_forecast',
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 4,  # Short for clarity
        'enc_in': 5,    # 2 targets + 2 covariates + 1 static
        'dec_in': 5,
        'c_out': 5,     # Output all features
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
        'features': 'M',
        'freq': 'b'
    })()
    
    # Create model
    model = TimesNet(args)
    model.eval()
    
    # Create test data with VERY specific target values
    batch_x = torch.randn(1, 96, 5)
    batch_y = torch.zeros(1, 52, 5)  # 48 label + 4 pred
    
    # Set VERY recognizable target values for future period
    target_patterns = {
        'Position_0 (Expected: Open)': [100.0, 200.0, 300.0, 400.0],
        'Position_1 (Expected: Close)': [1000.0, 2000.0, 3000.0, 4000.0],
    }
    
    covariate_patterns = {
        'Position_2 (Expected: Volume)': [10.0, 20.0, 30.0, 40.0],
        'Position_3 (Expected: Market_Cap)': [50.0, 60.0, 70.0, 80.0],
    }
    
    print("CHART INPUT TEST PATTERNS:")
    for name, pattern in target_patterns.items():
        print(f"   {name}: {pattern}")
    for name, pattern in covariate_patterns.items():
        print(f"   {name}: {pattern}")
    print(f"   Position_4 (Static): [99.0, 99.0, 99.0, 99.0]")
    
    # Fill future period (last 4 timesteps) with specific patterns
    for t in range(4):
        # Targets: position 0 and 1
        batch_y[:, 48 + t, 0] = target_patterns['Position_0 (Expected: Open)'][t]
        batch_y[:, 48 + t, 1] = target_patterns['Position_1 (Expected: Close)'][t]
        
        # Covariates: position 2 and 3  
        batch_y[:, 48 + t, 2] = covariate_patterns['Position_2 (Expected: Volume)'][t]
        batch_y[:, 48 + t, 3] = covariate_patterns['Position_3 (Expected: Market_Cap)'][t]
        
        # Static: position 4
        batch_y[:, 48 + t, 4] = 99.0
    
    # Label period: copy some historical data
    batch_y[:, :48, :] = batch_x[:, -48:, :]
    
    # Time features
    batch_x_mark = torch.randn(1, 96, 3)
    batch_y_mark = torch.randn(1, 52, 3)
    
    try:
        with torch.no_grad():
            # Prepare decoder input (zero out future predictions)
            dec_inp = torch.zeros_like(batch_y[:, -4:, :]).float()
            dec_inp = torch.cat([batch_y[:, :48, :], dec_inp], dim=1).float()
            
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            print(f"\n MODEL OUTPUT ANALYSIS:")
            print(f"Output shape: {outputs.shape}")
            
            # Extract outputs for each position in the future period
            print(f"\nOUTPUTS for future period (last 4 timesteps):")
            for pos in range(5):
                output_values = outputs[0, -4:, pos].tolist()
                print(f"   Position {pos}: {[f'{x:.3f}' for x in output_values]}")
            
            # Check scaling preservation
            print(f"\n SCALING ANALYSIS:")
            
            # Check if position 0 and 1 preserve the 10x ratio from inputs
            out_pos_0 = outputs[0, -4:, 0].tolist()
            out_pos_1 = outputs[0, -4:, 1].tolist()
            
            # Input ratio analysis
            input_ratios = []
            output_ratios = []
            for t in range(4):
                input_ratio = target_patterns['Position_1 (Expected: Close)'][t] / target_patterns['Position_0 (Expected: Open)'][t]
                output_ratio = out_pos_1[t] / out_pos_0[t] if out_pos_0[t] != 0 else 0
                input_ratios.append(input_ratio)
                output_ratios.append(output_ratio)
                print(f"   Time {t}: Input ratio (1000+t*1000)/(100+t*300) = {input_ratio:.2f}, Output ratio = {output_ratio:.2f}")
            
            # Check if trends are preserved
            print(f"\nGRAPH TREND ANALYSIS:")
            input_trend_0 = [target_patterns['Position_0 (Expected: Open)'][i+1] - target_patterns['Position_0 (Expected: Open)'][i] for i in range(3)]
            output_trend_0 = [out_pos_0[i+1] - out_pos_0[i] for i in range(3)]
            
            input_trend_1 = [target_patterns['Position_1 (Expected: Close)'][i+1] - target_patterns['Position_1 (Expected: Close)'][i] for i in range(3)]
            output_trend_1 = [out_pos_1[i+1] - out_pos_1[i] for i in range(3)]
            
            print(f"   Position 0 - Input trends: {input_trend_0}")
            print(f"   Position 0 - Output trends: {[f'{x:.3f}' for x in output_trend_0]}")
            print(f"   Position 1 - Input trends: {input_trend_1}")
            print(f"   Position 1 - Output trends: {[f'{x:.3f}' for x in output_trend_1]}")
            
            # Check correlation
            correlation_0 = sum(a*b for a,b in zip(input_trend_0, output_trend_0))
            correlation_1 = sum(a*b for a,b in zip(input_trend_1, output_trend_1))
            print(f"   Position 0 trend correlation: {correlation_0:.3f}")
            print(f"   Position 1 trend correlation: {correlation_1:.3f}")
            
            print(f"\nTARGET CONCLUSIONS:")
            if correlation_0 > 1000 and correlation_1 > 1000000:  # Strong positive correlation
                print(f"PASS Position 0 and 1 appear to be targets (preserve trends)")
            else:
                print(f"WARN  Position 0 and 1 may not be targets (trends not preserved)")
                
            # Check scaling factor
            if abs(output_ratios[0] - input_ratios[0]) < 1.0:
                print(f"PASS Ratios preserved - targets maintain relative scale")
            else:
                print(f"WARN  Ratios not preserved - targets are normalized")
                
            # Check if outputs are in normalized range
            all_outputs = [val for pos in range(5) for val in outputs[0, -4:, pos].tolist()]
            output_range = max(all_outputs) - min(all_outputs)
            print(f"CHART Output range across all positions: {output_range:.3f}")
            
            if output_range < 5.0:
                print(f"WARN  All outputs are normalized (range < 5)")
            else:
                print(f"PASS Outputs preserve original scale (range > 5)")
                
    except Exception as e:
        print(f"FAIL Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_target_order_and_scaling()
