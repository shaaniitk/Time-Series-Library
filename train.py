# File: 4_train.py
import torch
import torch.optim as optim
from dataclasses import dataclass
from utils.graph_utils import get_pyg_graph
from model import SOTA_PGAT_Model
from loss import GaussianNLL_Loss, nn

@dataclass
class Config:
    # Data params
    num_waves: int = 10
    num_targets: int = 4
    wave_feature_dim: int = 6
    target_feature_dim: int = 1
    topo_feature_dim: int = 2 # degree, clustering coeff
    # Model params
    num_transitions: int = 8
    d_model: int = 64
    num_encoder_layers: int = 3
    # Training params
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 16
    mode: str = 'bayesian'

def get_dummy_data(config, device):
    waves_x = torch.randn(config.batch_size, config.num_waves, config.wave_feature_dim).to(device)
    targets_x_in = torch.randn(config.batch_size, config.num_targets, config.target_feature_dim).to(device)
    targets_y_true = torch.randn(config.batch_size, config.num_targets).to(device)
    return waves_x, targets_x_in, targets_y_true

if __name__ == "__main__":
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    graph = get_pyg_graph(config, device)
    model = SOTA_PGAT_Model(config, mode=config.mode).to(device)
    
    if config.mode == 'standard': loss_fn = nn.MSELoss()
    else: loss_fn = GaussianNLL_Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    print(f"\n--- Starting Training of SOTA PGAT Model ---")
    
    for epoch in range(config.epochs):
        model.train()
        wave_x, target_x, target_y = get_dummy_data(config, device)
        
        optimizer.zero_grad()
        
        if config.mode == 'standard':
            preds = model(wave_x, target_x, graph)
            loss = loss_fn(preds, target_y)
        else:
            mean, std = model(wave_x, target_x, graph)
            loss = loss_fn(target_y, mean, std)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {loss.item():.4f}")
    print("--- Training Finished ---")