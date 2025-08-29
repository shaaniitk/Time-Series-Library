# TimesNet Training Speed Diagnostics - Quick Test Script
# Run this to quickly verify if your setup has major issues

import time
import torch
import sys
import os

def quick_system_check():
    """Quick check of system capabilities"""
    print("SEARCH QUICK SYSTEM CHECK")
    print("="*30)
    
    # Python version
    print(f" Python: {sys.version.split()[0]}")
    
    # PyTorch version and CUDA
    print(f"FIRE PyTorch: {torch.__version__}")
    
    # CUDA availability
    if torch.cuda.is_available():
        print(f"ROCKET CUDA: PASS Available (GPU: {torch.cuda.get_device_name(0)})")
        print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("ROCKET CUDA: FAIL Not available (CPU only)")
    
    # Memory test
    print("\n Memory Test:")
    try:
        # Create a small tensor
        test_tensor = torch.randn(1000, 1000)
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()
        print("   PASS Basic tensor operations work")
        del test_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"   FAIL Memory issue: {e}")
    
    # Data file check
    print("\nCHART Data File Check:")
    data_path = "./data/prepared_financial_data.csv"
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path) / 1024**2  # MB
        print(f"   PASS Data file found: {file_size:.1f} MB")
    else:
        print(f"   FAIL Data file not found: {data_path}")
        print(f"   IDEA Run the data preparation notebook first")
    
    return torch.cuda.is_available()

def quick_model_test():
    """Quick test of TimesNet model creation"""
    print("\nBRAIN QUICK MODEL TEST")
    print("="*30)
    
    try:
        # Add current directory to path for imports
        sys.path.append('.')
        
        # Import TimesNet
        from models.TimesNet import Model
          # Create minimal config
        class TestConfig:
            # === SEQUENCE PARAMETERS ===
            seq_len = 12
            pred_len = 1
            label_len = 2
            
            # === MODEL DIMENSIONS ===
            enc_in = 118
            dec_in = 118
            c_out = 118
            d_model = 8
            d_ff = 16
            
            # === ATTENTION ===
            n_heads = 1
            e_layers = 1
            d_layers = 1
            
            # === TIMESNET SPECIFIC ===
            top_k = 1
            num_kernels = 1
            
            # === REGULARIZATION ===
            dropout = 0.0
            
            # === EMBEDDINGS ===
            embed = 'timeF'
            activation = 'gelu'
            factor = 1
            distil = False
            moving_avg = 3
            output_attention = False
            
            # === TASK CONFIGURATION ===
            task_name = 'short_term_forecast'
            
            # === TRAINING (not used for model creation but required) ===
            use_amp = False
            
            # === DATA (not used for model creation but sometimes referenced) ===
            features = 'M'
            target = 'log_Close'
            freq = 'b'
        
        config = TestConfig()
        
        # Create model
        model_start = time.time()
        model = Model(config)
        model_time = time.time() - model_start
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"   PASS Model created in {model_time:.3f}s")
        print(f"    Parameters: {total_params:,}")        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dummy inputs with dimensions that match the actual data
        batch_size = 4  # Smaller batch size for testing
        
        # Main data tensor
        batch_x = torch.randn(batch_size, config.seq_len, config.enc_in).to(device)
        
        # Time features - try different dimensions to find what works
        try:
            # First try with 4 time features (most common)
            time_features_dim = 4
            batch_x_mark = torch.randn(batch_size, config.seq_len, time_features_dim).to(device)
            batch_y = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in).to(device)
            batch_y_mark = torch.randn(batch_size, config.label_len + config.pred_len, time_features_dim).to(device)
            
            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).to(device)
            dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1).to(device)
            
            # Test forward pass
            forward_start = time.time()
            with torch.no_grad():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            forward_time = time.time() - forward_start
            
        except Exception as e:
            if "cannot be multiplied" in str(e):
                print(f"   TOOL Trying with different time feature dimensions...")
                # Try with 5 time features (business day specific)
                time_features_dim = 5
                batch_x_mark = torch.randn(batch_size, config.seq_len, time_features_dim).to(device)
                batch_y_mark = torch.randn(batch_size, config.label_len + config.pred_len, time_features_dim).to(device)
                
                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).to(device)
                dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1).to(device)
                
                # Test forward pass again
                forward_start = time.time()
                with torch.no_grad():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                forward_time = time.time() - forward_start
            else:
                raise e
        
        print(f"   PASS Forward pass in {forward_time:.3f}s")
        print(f"   CHART Output shape: {outputs.shape}")
        
        if forward_time < 1.0:
            print("   ROCKET Model performance looks good!")
            return True
        else:
            print("   WARN Model is slower than expected")
            return False
        
    except Exception as e:
        print(f"   FAIL Model test failed: {e}")
        return False

def main():
    """Run complete quick test"""
    print("TEST TIMESNET QUICK DIAGNOSTIC TEST")
    print("TARGET This will quickly identify major issues")
    print("="*50)
    
    # System check
    cuda_available = quick_system_check()
    
    # Model test
    model_works = quick_model_test()
    
    # Summary
    print(f"\nCLIPBOARD QUICK TEST SUMMARY")
    print("="*30)
    
    if model_works:
        print("PASS SYSTEM LOOKS GOOD!")
        print("IDEA If training is still slow, use the diagnostic cells in the notebook")
        if cuda_available:
            print("ROCKET GPU is available - training should be fast")
        else:
            print("WARN CPU only - training will be slower but should work")
    else:
        print("FAIL ISSUES DETECTED!")
        print("IDEA Check the error messages above")
        print("TOOL Try running the emergency debug mode in the notebook")
    
    print(f"\nTARGET NEXT STEPS:")
    print(f"1. Open TimesNet_Light_Config.ipynb")
    if not model_works:
        print(f"2. Run: emergency_debug_mode()")
    elif not cuda_available:
        print(f"2. Use CPU-friendly settings (smaller batch size)")
    else:
        print(f"2. Try: switch_to_ultra_fast()")
    print(f"3. If issues persist, run the diagnostic cells")

if __name__ == "__main__":
    main()
