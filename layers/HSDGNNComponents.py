"""
HSDGNN Components adapted for Wave-Stock Prediction Architecture
Integrates hierarchical spatiotemporal dependency learning from HSDGNN paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple, List
import math
from layers.DynamicGraphAttention import DynamicGraphConstructor, DynamicGraphAttention


class IntraDependencyLearning(nn.Module):
    """
    HSDGNN's intra-dependency learning adapted for wave attributes
    Models time-varying correlations among wave variables [r, cos(θ), sin(θ), dθ/dt]
    """
    
    def __init__(self, n_attributes: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_attributes = n_attributes  # 4 for wave variables
        self.d_model = d_model
        
        # Attribute embedding (similar to HSDGNN's x_embedding)
        self.attribute_embedding = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1, 16)),
            ('sigmoid1', nn.Sigmoid()),
            ('fc2', nn.Linear(16, 8)),
            ('sigmoid2', nn.Sigmoid()),
            ('fc3', nn.Linear(8, d_model))
        ]))
        
        # Intra-dependency computation
        self.intra_fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(n_attributes, 16)),
            ('sigmoid1', nn.Sigmoid()),
            ('fc2', nn.Linear(16, 8)),
            ('sigmoid2', nn.Sigmoid()),
            ('fc3', nn.Linear(8, d_model))
        ]))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, wave_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wave_data: [B, L, 4] - Single wave's 4 variables
        Returns:
            wave_features: [B, L, d_model] - Processed wave features
        """
        B, L, A = wave_data.shape
        
        # Attribute-level embedding for each variable
        wave_embed = self.attribute_embedding(wave_data.unsqueeze(-1))  # [B, L, 4, d_model]
        
        # Compute intra-dependencies (HSDGNN's key innovation)
        # Create adjacency matrix based on attribute correlations
        supports_identity = torch.eye(A).to(wave_data.device)  # [4, 4]
        supports_learned = F.relu(torch.matmul(wave_embed, wave_embed.transpose(-2, -1)))  # [B, L, 4, 4]
        
        # Apply graph convolution on attributes
        x1 = torch.einsum("ij,blj->bli", supports_identity, wave_data)  # [B, L, 4]
        x2 = torch.einsum("blij,blj->bli", supports_learned, wave_data)  # [B, L, 4]
        
        # Combine and process
        combined_input = x1 + x2
        wave_features = self.intra_fc(combined_input)  # [B, L, d_model]
        
        return self.dropout(wave_features)


class HierarchicalSpatiotemporalBlock(nn.Module):
    """
    HSDGNN block adapted for Wave-Stock architecture
    Combines intra-wave dependencies with inter-wave dynamic topology
    """
    
    def __init__(self, n_waves: int, wave_features: int, d_model: int, rnn_units: int, seq_len: int, window_size: int, threshold: float):
        super().__init__()
        self.n_waves = n_waves
        self.wave_features = wave_features
        self.d_model = d_model
        self.rnn_units = rnn_units
        
        # Intra-dependency learning for each wave
        self.intra_dependency = IntraDependencyLearning(wave_features, d_model)
        
        # Dynamic topology generator with weighted edges and self-loops
        self.dynamic_graph_constructor = DynamicGraphConstructor(
            window_size, threshold, use_weighted=True, include_self_loops=True
        )
        
        # Dynamic graph attention
        self.dynamic_graph_attention = DynamicGraphAttention(d_model, rnn_units)

        # Two-level GRU (HSDGNN's key architecture)
        self.gru1 = nn.GRU(d_model, rnn_units, batch_first=True)  # Temporal modeling
        self.gru2 = nn.GRU(rnn_units, rnn_units, batch_first=True)  # Graph evolution modeling
        
        # Node adaptive parameters (from HSDGNN)
        self.weights_pool = nn.Parameter(torch.FloatTensor(d_model, rnn_units, rnn_units))
        self.bias_pool = nn.Parameter(torch.FloatTensor(d_model, rnn_units))
        
        # Diffusion convolution combiner
        self.diff_conv = nn.Conv2d(rnn_units * 2, rnn_units, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)
        
    def forward(self, wave_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wave_data: [B, L, N_waves, 4] - All waves data
        Returns:
            output: [B, L, N_waves, rnn_units] - Processed features
        """
        B, L, N, A = wave_data.shape
        
        # Step 1: Intra-dependency learning for each wave
        wave_features = []
        for i in range(N):
            wave_feat = self.intra_dependency(wave_data[:, :, i, :])  # [B, L, d_model]
            wave_features.append(wave_feat)
        
        wave_features = torch.stack(wave_features, dim=2)  # [B, L, N_waves, d_model]
        
        # Step 2: Temporal modeling with GRU1
        wave_features_flat = wave_features.reshape(B * N, L, self.d_model)
        gru1_output, _ = self.gru1(wave_features_flat)  # [B*N, L, rnn_units]
        gru1_output = gru1_output.reshape(B, N, L, self.rnn_units).transpose(1, 2)  # [B, L, N, rnn_units]
        
        # Step 3: Dynamic topology generation and attention
        adj_matrix = self.dynamic_graph_constructor(wave_features.mean(dim=-1)) # [B, L, N, N]
        dynamic_output = self.dynamic_graph_attention(gru1_output, adj_matrix)
        
        # Step 4: Combine outputs for GRU2 input
        combined_input = torch.cat([gru1_output, dynamic_output], dim=-1)  # [B, L, N, 2*rnn_units]
        combined_input = self.diff_conv(combined_input.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        combined_input = self.dropout(combined_input)
        
        # Step 5: Graph evolution modeling with GRU2
        combined_flat = combined_input.reshape(B * N, L, self.rnn_units)
        gru2_output, _ = self.gru2(combined_flat)
        final_output = gru2_output.reshape(B, N, L, self.rnn_units).transpose(1, 2)  # [B, L, N, rnn_units]
        
        return final_output


class HSDGNNResidualPredictor(nn.Module):
    """
    HSDGNN's residual learning approach adapted for multi-step forecasting
    
    WARNING: This is a highly complex model with many parameters (multiple GRUs, 
    adaptive weights, embeddings). It requires:
    - Large, diverse datasets (>10k samples recommended)
    - Strong regularization (dropout, weight decay)
    - Careful hyperparameter tuning
    - Sufficient computational resources
    
    The model is prone to overfitting on small datasets.
    """
    
    def __init__(self, n_waves: int, wave_features: int, d_model: int, rnn_units: int, 
                 seq_len: int, pred_len: int, n_blocks: int = 3, window_size: int = 10, threshold: float = 0.5):
        super().__init__()
        self.n_blocks = n_blocks
        self.pred_len = pred_len
        
        # Multiple HSDGNN blocks for residual learning
        self.blocks = nn.ModuleList([
            HierarchicalSpatiotemporalBlock(n_waves, wave_features, d_model, rnn_units, seq_len, window_size, threshold)
            for _ in range(n_blocks)
        ])
        
        # Output convolutions for each block
        self.output_convs = nn.ModuleList([
            nn.Conv2d(1, pred_len, kernel_size=(1, rnn_units), bias=True)
            for _ in range(n_blocks)
        ])
        
        # Residual convolutions
        self.residual_convs = nn.ModuleList([
            nn.Conv2d(1, seq_len, kernel_size=(1, rnn_units), bias=True)
            for _ in range(n_blocks - 1)
        ])
        
        # Enhanced regularization for complex model
        self.dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(n_blocks)])
        
        # Add batch normalization for stability
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(1) for _ in range(n_blocks)
        ])
        
    def forward(self, wave_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wave_data: [B, L, N_waves, 4] - Input wave data
        Returns:
            predictions: [B, pred_len, N_waves, 1] - Multi-step predictions
        """
        predictions = []
        current_input = wave_data
        
        for i in range(self.n_blocks):
            # Process through HSDGNN block
            block_output = self.blocks[i](current_input)  # [B, L, N_waves, rnn_units]
            block_output = block_output[:, -1:, :, :]  # Take last timestep
            
            # Apply enhanced regularization
            block_output = self.batch_norms[i](block_output)
            block_output = self.dropouts[i](block_output)
            
            # Generate prediction
            pred = self.output_convs[i](block_output)  # [B, pred_len, N_waves, 1]
            predictions.append(pred)
            
            # Compute residual for next block (except last block)
            if i < self.n_blocks - 1:
                residual = self.residual_convs[i](block_output)  # [B, L, N_waves, 1]
                current_input = current_input - residual.expand_as(current_input)
        
        # Sum all predictions (HSDGNN's residual approach)
        final_prediction = sum(predictions)
        
        return final_prediction