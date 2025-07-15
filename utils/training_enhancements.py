import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np


class AdaptiveLossWeighting(nn.Module):
    """Adaptive loss weighting for multi-task learning"""
    
    def __init__(self, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses):
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
        
        return sum(weighted_losses)


class FocalLoss(nn.Module):
    """Focal loss for handling imbalanced data"""
    
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class WarmupCosineScheduler:
    """Warmup + Cosine annealing scheduler"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup phase
            lr_scale = epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
            lr_scale = max(lr_scale, self.min_lr / max(self.base_lrs))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale


class EarlyStopping:
    """Early stopping with patience"""
    
    def __init__(self, patience=10, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class GradientAccumulator:
    """Gradient accumulation for large batch training"""
    
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        
    def step(self, loss):
        loss = loss / self.accumulation_steps
        loss.backward()
        
        self.step_count += 1
        if self.step_count % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()


class ModelEMA:
    """Exponential Moving Average of model parameters"""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def get_enhanced_optimizer(model, lr=1e-3, weight_decay=1e-4):
    """Get enhanced optimizer with different learning rates for different components"""
    
    # Separate parameters by type
    embedding_params = []
    attention_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'embedding' in name:
            embedding_params.append(param)
        elif 'attention' in name or 'attn' in name:
            attention_params.append(param)
        else:
            other_params.append(param)
    
    # Different learning rates for different components
    optimizer = optim.AdamW([
        {'params': embedding_params, 'lr': lr * 0.5, 'weight_decay': weight_decay * 0.1},
        {'params': attention_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': other_params, 'lr': lr * 1.5, 'weight_decay': weight_decay}
    ])
    
    return optimizer