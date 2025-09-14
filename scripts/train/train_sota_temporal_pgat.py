#!/usr/bin/env python3
"""
Training script for SOTA Temporal PGAT model
Supports YAML configuration and modular components
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
from utils.graph_utils import get_pyg_graph
# from utils.losses import GaussianNLL_Loss

# Simple Gaussian NLL Loss implementation
class GaussianNLL_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, targets, mean, std):
        """Gaussian Negative Log Likelihood Loss"""
        var = std ** 2
        return 0.5 * (torch.log(2 * torch.pi * var) + (targets - mean) ** 2 / var).mean()
# from utils.metrics import calculate_metrics

# Simple metrics calculation function
def calculate_metrics(predictions, targets):
    """Calculate basic metrics"""
    mse = torch.mean((predictions - targets) ** 2)
    mae = torch.mean(torch.abs(predictions - targets))
    return {'mse': mse.item(), 'mae': mae.item()}

class ConfigLoader:
    """Load and validate YAML configuration"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration parameters"""
        required_keys = ['model', 'training', 'data']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration section: {key}")

class ModelConfig:
    """Model configuration wrapper"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        
        # Model parameters
        model_config = config_dict['model']['architecture']
        self.d_model = model_config['d_model']
        self.n_heads = model_config['n_heads']
        self.d_ff = model_config['d_ff']
        self.dropout = model_config['dropout']
        self.num_layers = model_config['num_layers']
        
        # Input/Output parameters
        input_config = config_dict['model']['input']
        self.seq_len = input_config['seq_len']
        self.label_len = input_config['label_len']
        self.pred_len = input_config['pred_len']
        self.features = input_config['features']
        self.target = input_config['target']
        
        # Training parameters
        training_config = config_dict['training']
        self.batch_size = training_config['batch_size']
        self.learning_rate = training_config['learning_rate']
        self.num_epochs = training_config['num_epochs']
        self.patience = training_config['patience']
        
        # Model mode
        self.mode = config_dict['model']['mode']

class DataGenerator:
    """Generate synthetic data for training and testing"""
    
    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
        self.device = device
    
    def generate_batch(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of synthetic data"""
        if batch_size is None:
            batch_size = self.config.batch_size
            
        # Generate wave window data
        wave_window = torch.randn(
            batch_size, 
            self.config.seq_len, 
            self.config.d_model
        ).to(self.device)
        
        # Generate target window data
        target_window = torch.randn(
            batch_size, 
            self.config.label_len, 
            self.config.d_model
        ).to(self.device)
        
        # Generate graph adjacency matrix
        graph = torch.randn(
            batch_size, 
            self.config.seq_len, 
            self.config.seq_len
        ).to(self.device)
        
        # Generate ground truth targets
        targets = torch.randn(
            batch_size, 
            self.config.pred_len
        ).to(self.device)
        
        return wave_window, target_window, graph, targets
    
    def create_dataloader(self, num_samples: int = 1000) -> DataLoader:
        """Create a DataLoader with synthetic data"""
        wave_windows = []
        target_windows = []
        graphs = []
        targets = []
        
        for _ in range(num_samples // self.config.batch_size):
            wave_w, target_w, graph, target = self.generate_batch()
            wave_windows.append(wave_w)
            target_windows.append(target_w)
            graphs.append(graph)
            targets.append(target)
        
        # Concatenate all batches
        wave_windows = torch.cat(wave_windows, dim=0)
        target_windows = torch.cat(target_windows, dim=0)
        graphs = torch.cat(graphs, dim=0)
        targets = torch.cat(targets, dim=0)
        
        dataset = TensorDataset(wave_windows, target_windows, graphs, targets)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

class Trainer:
    """Training manager for SOTA Temporal PGAT"""
    
    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
        self.device = device
        self.setup_logging()
        
        # Initialize model
        self.model = SOTA_Temporal_PGAT(config, mode=config.mode).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=0.0001
        )
        
        # Initialize loss function
        if config.mode == 'standard':
            self.loss_fn = nn.MSELoss()
        else:  # probabilistic
            self.loss_fn = GaussianNLL_Loss()
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10
        )
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (wave_window, target_window, graph, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.config.mode == 'standard':
                predictions = self.model(wave_window, target_window, graph)
                loss = self.loss_fn(predictions, targets)
            else:  # probabilistic
                mean, std = self.model(wave_window, target_window, graph)
                # Only use the target portion of targets (last 48 timesteps)
                target_only = targets[:, -mean.shape[1]:]
                loss = self.loss_fn(target_only, mean, std)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                self.logger.info(f"Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for wave_window, target_window, graph, targets in dataloader:
                if self.config.mode == 'standard':
                    predictions = self.model(wave_window, target_window, graph)
                    # Only use the target portion of targets
                    target_only = targets[:, -predictions.shape[1]:]
                    loss = self.loss_fn(target_only, predictions)
                else:  # probabilistic
                    mean, std = self.model(wave_window, target_window, graph)
                    # Only use the target portion of targets
                    target_only = targets[:, -mean.shape[1]:]
                    loss = self.loss_fn(target_only, mean, std)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Model mode: {self.config.mode}")
        self.logger.info(f"Device: {self.device}")
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self.train_epoch(train_dataloader)
            
            # Validation
            val_loss = self.validate(val_dataloader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.patience_counter += 1
            
            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.8f}"
            )
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_loss:.6f}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train SOTA Temporal PGAT model')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/sota_temporal_pgat.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        help='Device to use (auto, cpu, cuda)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_dict = ConfigLoader.load_config(args.config)
    ConfigLoader.validate_config(config_dict)
    config = ModelConfig(config_dict)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Model mode: {config.mode}")
    
    # Create data generators
    data_generator = DataGenerator(config, device)
    train_dataloader = data_generator.create_dataloader(num_samples=2000)
    val_dataloader = data_generator.create_dataloader(num_samples=500)
    
    # Initialize trainer
    trainer = Trainer(config, device)
    
    # Start training
    trainer.train(train_dataloader, val_dataloader)
    
    print("Training script completed successfully!")

if __name__ == "__main__":
    main()