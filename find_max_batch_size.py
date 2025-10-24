"""
Find Maximum Safe Batch Size
Tests different batch sizes to find the largest one that doesn't OOM

Usage:
    python find_max_batch_size.py [--config path/to/config.yaml]
"""

import torch
import torch.nn as nn
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project root
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def test_batch_size(
    model: nn.Module,
    batch_size: int,
    enc_seq_len: int,
    dec_seq_len: int,
    input_features: int,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[bool, float, str]:
    """
    Test if a specific batch size causes OOM.
    
    Returns:
        (success, peak_memory_gb, error_msg)
    """
    
    try:
        # Clear cache before test
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        
        # Create dummy inputs matching production shapes
        # Encoder inputs
        x_enc = torch.randn(batch_size, enc_seq_len, input_features, device=device)
        x_mark_enc = torch.zeros(batch_size, enc_seq_len, input_features, device=device)

        # Decoder inputs (label_len + pred_len)
        x_dec = torch.randn(batch_size, dec_seq_len, input_features, device=device)
        x_mark_dec = torch.zeros(batch_size, dec_seq_len, input_features, device=device)
        
        # Forward pass
        model.train()
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Extract predictions
        if isinstance(output, tuple):
            predictions = output[0]
        else:
            predictions = output
        
        # Compute loss - predictions are for pred_len only, not full dec_seq_len
        # Target should match prediction shape (batch, pred_len, features)
        pred_len = predictions.size(1)  # Get actual prediction length from output
        dummy_target = torch.randn(batch_size, pred_len, 4, device=device)  # Match pred shape
        loss = nn.functional.mse_loss(predictions[:, :, :4], dummy_target)
        
        # Backward pass (the critical part)
        loss.backward()
        
        # Track peak memory
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)  # GB
        else:
            peak_memory = 0.0
        
        # Cleanup
        del x_enc, x_mark_enc, x_dec, x_mark_dec, dummy_target, output, predictions, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        return True, peak_memory, ""
        
    except RuntimeError as e:
        error_msg = str(e)
        
        # Get peak memory before OOM
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
        else:
            peak_memory = 0.0
        
        # Cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        if "out of memory" in error_msg.lower():
            return False, peak_memory, "OOM"
        else:
            # Some other error
            return False, peak_memory, error_msg
    
    except Exception as e:
        return False, 0.0, str(e)


def binary_search_max_batch_size(
    model: nn.Module,
    min_batch: int,
    max_batch: int,
    enc_seq_len: int,
    dec_seq_len: int,
    input_features: int,
    device: torch.device,
    use_amp: bool = False,
) -> int:
    """Binary search to find maximum safe batch size."""
    
    print(f"\n🔍 Binary search for maximum batch size...")
    print(f"   Range: {min_batch} to {max_batch}")
    print(f"   Encoder sequence length: {enc_seq_len}")
    print(f"   Decoder sequence length: {dec_seq_len}")
    print(f"   Input features: {input_features}")
    print(f"   Mixed precision: {use_amp}")
    print()
    
    left, right = min_batch, max_batch
    best_working = min_batch
    
    iteration = 0
    max_iterations = 20
    
    while left <= right and iteration < max_iterations:
        iteration += 1
        mid = (left + right) // 2
        
        print(f"Iteration {iteration}: Testing batch_size={mid}...", end=" ")
        
        success, peak_mem, error = test_batch_size(
            model, mid, enc_seq_len, dec_seq_len, input_features, device, use_amp
        )
        
        if success:
            print(f"✅ Success! Peak memory: {peak_mem:.2f}GB")
            best_working = mid
            left = mid + 1  # Try larger
        else:
            if error == "OOM":
                print(f"❌ OOM at {peak_mem:.2f}GB")
            else:
                print(f"❌ Error: {error[:50]}...")
            right = mid - 1  # Try smaller
    
    return best_working


def load_model_from_config(config_path: Path, device: torch.device) -> Tuple[Optional[nn.Module], Optional[Dict[str, Any]]]:
    """Load model based on config."""
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"📁 Loaded config from {config_path}")
        print(f"   Model: {config.get('model', config.get('model_name', 'Unknown'))}")
        print(f"   d_model: {config.get('d_model')}")
        print(f"   n_heads: {config.get('n_heads')}")
        print(f"   e_layers: {config.get('e_layers')}")
        
        # Create SimpleConfig
        from scripts.train.train_celestial_production import SimpleConfig
        args = SimpleConfig(config)
        
        # Add required attributes
        args.task_name = 'long_term_forecast'
        args.data_name = 'custom'
        
        # Import and create model
        from models.Celestial_Enhanced_PGAT import Model
        model = Model(args).to(device)
        
        return model, config
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None


def create_dummy_model(
    d_model: int,
    n_heads: int,
    e_layers: int,
    device: torch.device
) -> nn.Module:
    """Create a dummy model for testing."""
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4)
                for _ in range(e_layers)
            ])
            self.output = nn.Linear(d_model, 4)
        
        def forward(self, x, *args):
            for layer in self.layers:
                x = layer(x)
            return self.output(x)
    
    return DummyModel().to(device)


def main():
    parser = argparse.ArgumentParser(description='Find maximum safe batch size')
    parser.add_argument('--config', type=str, 
                       default='configs/celestial_enhanced_pgat_production.yaml',
                       help='Path to config file')
    parser.add_argument('--min-batch', type=int, default=1,
                       help='Minimum batch size to test')
    parser.add_argument('--max-batch', type=int, default=128,
                       help='Maximum batch size to test')
    parser.add_argument('--seq-len', type=int, default=512,
                       help='Sequence length')
    parser.add_argument('--input-features', type=int, default=118,
                       help='Number of input features')
    parser.add_argument('--use-amp', action='store_true',
                       help='Use mixed precision (AMP)')
    parser.add_argument('--dummy-model', action='store_true',
                       help='Use dummy model instead of loading from config')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MAXIMUM BATCH SIZE FINDER")
    print("="*80)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(device)
        print(f"\n🖥️  GPU: {props.name}")
        print(f"   Total memory: {props.total_memory / (1024**3):.2f}GB")
    else:
        device = torch.device('cpu')
        print(f"\n🖥️  Device: CPU")
    
    # Load or create model
    config: Optional[Dict[str, Any]] = None
    if args.dummy_model:
        print(f"\n🤖 Creating dummy model...")
        model = create_dummy_model(
            d_model=512,
            n_heads=8,
            e_layers=3,
            device=device
        )
        enc_seq_len = int(args.seq_len)
        dec_seq_len = int(args.seq_len)  # fallback for dummy
        input_features = int(args.input_features)
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"\n❌ Config not found: {config_path}")
            print("   Using dummy model instead...")
            model = create_dummy_model(512, 8, 3, device)
            enc_seq_len = int(args.seq_len)
            dec_seq_len = int(args.seq_len)
            input_features = int(args.input_features)
        else:
            model, config = load_model_from_config(config_path, device)
            if model is None:
                print("   Using dummy model instead...")
                model = create_dummy_model(512, 8, 3, device)
                enc_seq_len = int(args.seq_len)
                dec_seq_len = int(args.seq_len)
                input_features = int(args.input_features)
            else:
                # Pull shapes from config when available
                enc_seq_len = int(config.get('seq_len', args.seq_len))
                label_len = int(config.get('label_len', 0))
                pred_len = int(config.get('pred_len', 0))
                dec_seq_len = max(1, label_len + pred_len) if (label_len or pred_len) else int(args.seq_len)
                input_features = int(config.get('enc_in', config.get('dec_in', args.input_features)))
                print(f"\n📐 Using shapes from config:")
                print(f"   enc_seq_len={enc_seq_len}, dec_seq_len={dec_seq_len}, input_features={input_features}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Estimated model size: {total_params * 4 / (1024**2):.1f}MB")
    
    # Find maximum batch size
    max_safe_batch = binary_search_max_batch_size(
        model=model,
        min_batch=args.min_batch,
        max_batch=args.max_batch,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        input_features=input_features,
        device=device,
        use_amp=args.use_amp,
    )
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n✅ Maximum safe batch size: {max_safe_batch}")
    
    # Calculate effective batch sizes with gradient accumulation
    print(f"\n📈 Recommended configurations:")
    print(f"{'Batch Size':<12} {'Grad Accum':<12} {'Effective Batch':<15} {'Safety Margin'}")
    print("-" * 80)
    
    # Conservative (75% of max)
    conservative = max(1, int(max_safe_batch * 0.75))
    print(f"{conservative:<12} {1:<12} {conservative:<15} High (75%)")
    
    # With gradient accumulation
    for accum in [2, 4, 8]:
        batch_per_step = max(1, conservative // accum)
        effective = batch_per_step * accum
        print(f"{batch_per_step:<12} {accum:<12} {effective:<15} High (75%)")
    
    # Moderate (85% of max)
    moderate = max(1, int(max_safe_batch * 0.85))
    print(f"{moderate:<12} {1:<12} {moderate:<15} Moderate (85%)")
    
    # Aggressive (95% of max)
    aggressive = max(1, int(max_safe_batch * 0.95))
    print(f"{aggressive:<12} {1:<12} {aggressive:<15} Low (95%)")
    
    print("\n💡 Recommendation:")
    print(f"   Use batch_size={conservative} for stable training")
    print(f"   Or batch_size={conservative//2} with gradient_accumulation_steps=2")
    
    if args.use_amp:
        print(f"\n   Mixed precision is enabled (saves ~40% memory)")
        print(f"   Without AMP, reduce batch size by ~40%")
    else:
        no_amp_estimate = max(1, int(max_safe_batch * 0.6))
        print(f"\n   Mixed precision is disabled")
        print(f"   With AMP enabled, you could use batch_size~{int(max_safe_batch / 0.6)}")
    
    print("\n" + "="*80)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
