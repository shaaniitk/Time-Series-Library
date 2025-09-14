import torch.nn as nn
import torch.nn.functional as F

class StandardDecoder(nn.Module):
    """Predicts a single value for each target."""
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))
    def forward(self, x): return self.mlp(x).squeeze(-1)

class ProbabilisticDecoder(nn.Module):
    """Predicts a mean and standard deviation for each target to quantify uncertainty."""
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU())
        self.mean_head = nn.Linear(d_model // 2, 1)
        self.std_dev_head = nn.Linear(d_model // 2, 1)
    def forward(self, x):
        processed = self.mlp(x)
        mean = self.mean_head(processed).squeeze(-1)
        std_dev = F.softplus(self.std_dev_head(processed).squeeze(-1)) + 1e-6
        return mean, std_dev