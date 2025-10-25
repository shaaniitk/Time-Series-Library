#!/usr/bin/env python3
"""
Debug script to investigate training vs validation loss discrepancy.

This script will:
1. Load the same model and data as production training
2. Run a few training batches and validation batches
3. Print detailed statistics about losses, gradients, and data
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main diagnostic function."""
    logger.info("=" * 80)
    logger.info("TRAINING LOSS DIAGNOSTIC")
    logger.info("=" * 80)
    
    # Import after path setup
    import yaml
    from data_provider.data_factory import data_provider
    from models import Celestial_Enhanced_PGAT
    
    # Simple config class
    class SimpleConfig:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                setattr(self, key, value)
    
    # Load config
    config_path = Path("configs/celestial_enhanced_pgat_production.yaml")
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    args = SimpleConfig(config_dict)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    
    train_data, train_loader = data_provider(args, flag='train')
    val_data, val_loader = data_provider(args, flag='val')
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Get scalers
    target_scaler = getattr(train_data, 'target_scaler', None) or getattr(train_data, 'scaler', None)
    if target_scaler is None:
        logger.error("No scaler found!")
        return
    
    logger.info(f"Target scaler: {type(target_scaler).__name__}")
    logger.info(f"Target scaler mean: {target_scaler.mean_[:4]}")
    logger.info(f"Target scaler std: {target_scaler.scale_[:4]}")
    
    # Initialize model
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING MODEL")
    logger.info("=" * 80)
    
    model = Celestial_Enhanced_PGAT.Model(args).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup loss
    criterion = nn.MSELoss()
    
    # Helper function to scale targets
    def scale_targets(targets_unscaled, indices=[0,1,2,3]):
        """Scale targets for loss computation."""
        targets_only = targets_unscaled[:, :, indices]
        targets_np = targets_only.cpu().numpy()
        batch_size, seq_len, n_targets = targets_np.shape
        targets_reshaped = targets_np.reshape(-1, n_targets)
        targets_scaled_reshaped = target_scaler.transform(targets_reshaped)
        targets_scaled_np = targets_scaled_reshaped.reshape(batch_size, seq_len, n_targets)
        return torch.from_numpy(targets_scaled_np).float().to(device)
    
    # Helper function to create decoder input
    def create_dec_inp(batch_y):
        """Create decoder input with future celestial data."""
        target_indices = [0, 1, 2, 3]
        celestial_indices = list(range(4, batch_y.shape[-1]))
        
        future_celestial = batch_y[:, -args.pred_len:, celestial_indices]
        future_targets = torch.zeros_like(batch_y[:, -args.pred_len:, target_indices])
        future_combined = torch.cat([future_targets, future_celestial], dim=-1)
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], future_combined], dim=1)
        return dec_inp
    
    # Test training mode
    logger.info("\n" + "=" * 80)
    logger.info("TESTING TRAINING MODE")
    logger.info("=" * 80)
    
    model.train()
    train_losses = []
    
    for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        if batch_idx >= 5:  # Only test first 5 batches
            break
        
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        
        dec_inp = create_dec_inp(batch_y).float().to(device)
        
        # Forward pass
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        # Extract predictions
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        
        # Compute loss
        y_pred = outputs[:, -args.pred_len:, :4]
        y_true = scale_targets(batch_y[:, -args.pred_len:, :])
        
        loss = criterion(y_pred, y_true)
        train_losses.append(loss.item())
        
        logger.info(f"  Train Batch {batch_idx}: loss={loss.item():.6f}")
        
        if batch_idx == 0:
            logger.info(f"    Batch shape: {batch_x.shape}")
            logger.info(f"    Pred shape: {y_pred.shape}")
            logger.info(f"    True shape: {y_true.shape}")
            logger.info(f"    Pred mean: {y_pred.mean().item():.6f}, std: {y_pred.std().item():.6f}")
            logger.info(f"    True mean: {y_true.mean().item():.6f}, std: {y_true.std().item():.6f}")
            logger.info(f"    Pred min/max: [{y_pred.min().item():.6f}, {y_pred.max().item():.6f}]")
            logger.info(f"    True min/max: [{y_true.min().item():.6f}, {y_true.max().item():.6f}]")
    
    avg_train_loss = sum(train_losses) / len(train_losses)
    logger.info(f"\nAverage Train Loss (first 5 batches): {avg_train_loss:.6f}")
    
    # Test validation mode
    logger.info("\n" + "=" * 80)
    logger.info("TESTING VALIDATION MODE")
    logger.info("=" * 80)
    
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
            if batch_idx >= 5:  # Only test first 5 batches
                break
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            dec_inp = create_dec_inp(batch_y).float().to(device)
            
            # Forward pass
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Extract predictions
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            
            # Compute loss
            y_pred = outputs[:, -args.pred_len:, :4]
            y_true = scale_targets(batch_y[:, -args.pred_len:, :])
            
            loss = criterion(y_pred, y_true)
            val_losses.append(loss.item())
            
            logger.info(f"  Val Batch {batch_idx}: loss={loss.item():.6f}")
            
            if batch_idx == 0:
                logger.info(f"    Batch shape: {batch_x.shape}")
                logger.info(f"    Pred shape: {y_pred.shape}")
                logger.info(f"    True shape: {y_true.shape}")
                logger.info(f"    Pred mean: {y_pred.mean().item():.6f}, std: {y_pred.std().item():.6f}")
                logger.info(f"    True mean: {y_true.mean().item():.6f}, std: {y_true.std().item():.6f}")
                logger.info(f"    Pred min/max: [{y_pred.min().item():.6f}, {y_pred.max().item():.6f}]")
                logger.info(f"    True min/max: [{y_true.min().item():.6f}, {y_true.max().item():.6f}]")
    
    avg_val_loss = sum(val_losses) / len(val_losses)
    logger.info(f"\nAverage Val Loss (first 5 batches): {avg_val_loss:.6f}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Train Loss: {avg_train_loss:.6f}")
    logger.info(f"Val Loss: {avg_val_loss:.6f}")
    logger.info(f"Ratio (train/val): {avg_train_loss / avg_val_loss:.2f}x")
    
    if avg_train_loss > avg_val_loss * 2:
        logger.warning("⚠️  SUSPICIOUS: Train loss is >2x validation loss!")
        logger.warning("This suggests a potential bug in training setup.")
    elif avg_train_loss > avg_val_loss * 1.5:
        logger.info("✓ NORMAL: Train loss slightly higher due to dropout/regularization.")
    else:
        logger.info("✓ EXPECTED: Train and val losses are similar for untrained model.")

if __name__ == "__main__":
    main()
