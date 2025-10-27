"""
Phase and temporal feature engineering for Enhanced SOTA PGAT
Contains phase analysis, delayed influence, and feature augmentation components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


class PhaseFeatureExtractor(nn.Module):
    """Extracts phase-aware features from wave data"""
    
    def __init__(self, d_model: int, enable_phase_features: bool = True):
        super().__init__()
        self.enable_phase_features = enable_phase_features
        if self.enable_phase_features:
            self.phase_feature_projector = nn.Linear(6, d_model)
    
    def forward(self, wave_window: torch.Tensor, wave_spatial: torch.Tensor) -> torch.Tensor:
        """Inject phase-aware features into wave node representations."""
        if not self.enable_phase_features:
            return wave_spatial
        if not hasattr(self, 'phase_feature_projector'):
            return wave_spatial
        if wave_window.dim() != 3:
            return wave_spatial

        phase_features = self._compute_phase_features(
            wave_window,
            wave_spatial.device,
            wave_spatial.dtype,
        )
        projected = self.phase_feature_projector(phase_features)
        
        # Align node dimension if mismatch occurs (e.g., enc_in includes targets)
        if projected.size(1) != wave_spatial.size(1):
            projected = projected.transpose(1, 2)
            projected = F.interpolate(projected, size=wave_spatial.size(1), mode='nearest')
            projected = projected.transpose(1, 2)
        
        return wave_spatial + projected
    
    def _compute_phase_features(self, wave_window: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute harmonic embeddings and relative phase statistics for wave nodes."""
        batch_size, seq_len, wave_nodes = wave_window.shape
        float_window = wave_window.float()
        freq_domain = torch.fft.rfft(float_window, dim=1)

        if freq_domain.size(1) > 1:
            dominant = freq_domain[:, 1, :]
        else:
            dominant = freq_domain[:, 0, :]

        amplitude = torch.log1p(torch.abs(dominant))
        phase = torch.angle(dominant)
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)

        if freq_domain.size(1) > 2:
            phase_spectrum = torch.angle(freq_domain[:, 1:, :])
            phase_diff = phase_spectrum[:, 1:, :] - phase_spectrum[:, :-1, :]
            phase_velocity = phase_diff.mean(dim=1)
        else:
            phase_velocity = torch.zeros_like(amplitude)

        phase_matrix = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        relative_sin = torch.sin(phase_matrix).mean(dim=-1)
        relative_cos = torch.cos(phase_matrix).mean(dim=-1)

        features = torch.stack(
            [amplitude, sin_phase, cos_phase, phase_velocity, relative_sin, relative_cos],
            dim=-1,
        )
        return features.to(device=device, dtype=dtype)


class DelayedInfluenceProcessor(nn.Module):
    """Processes delayed influence features for transition nodes"""
    
    def __init__(self, d_model: int, wave_nodes: int, transition_nodes: int, 
                 enable_delayed_influence: bool = True, delayed_max_lag: int = 3):
        super().__init__()
        self.enable_delayed_influence = enable_delayed_influence
        self.delayed_max_lag = max(1, delayed_max_lag)
        
        if self.enable_delayed_influence:
            self.delay_feature_projector = nn.Linear(self.delayed_max_lag, d_model)
            self.delay_wave_to_transition = nn.Parameter(torch.randn(wave_nodes, transition_nodes))
    
    def forward(self, wave_window: torch.Tensor, wave_spatial: torch.Tensor, 
                transition_nodes: int) -> Optional[torch.Tensor]:
        """Generate lag-aware features for transition nodes."""
        if not self.enable_delayed_influence:
            return None
        if not hasattr(self, 'delay_feature_projector') or not hasattr(self, 'delay_wave_to_transition'):
            return None
        if wave_window.dim() != 3:
            return None

        batch_size, seq_len, wave_nodes = wave_window.shape
        if seq_len < 2:
            return None

        lags = min(self.delayed_max_lag, max(1, seq_len - 1))
        lag_features: List[torch.Tensor] = []
        for lag in range(1, lags + 1):
            shifted = torch.roll(wave_window, shifts=lag, dims=1)
            corr = (wave_window * shifted).mean(dim=1)
            lag_features.append(corr)

        lag_tensor = torch.stack(lag_features, dim=-1)
        if lag_tensor.size(-1) < self.delayed_max_lag:
            pad_width = self.delayed_max_lag - lag_tensor.size(-1)
            lag_tensor = F.pad(lag_tensor, (0, pad_width))

        delay_wave = self.delay_feature_projector(lag_tensor)

        weight_matrix = self.delay_wave_to_transition.to(delay_wave.dtype)
        if weight_matrix.size(0) != wave_nodes or weight_matrix.size(1) != transition_nodes:
            weight_matrix = weight_matrix.unsqueeze(0).unsqueeze(0)
            weight_matrix = F.interpolate(
                weight_matrix,
                size=(wave_nodes, transition_nodes),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0).squeeze(0)

        weight_matrix = torch.softmax(weight_matrix, dim=0)
        delayed_transition = torch.einsum('bwd,wt->btd', delay_wave, weight_matrix)
        return delayed_transition.to(device=wave_spatial.device, dtype=wave_spatial.dtype)


class GroupInteractionProcessor(nn.Module):
    """Processes group interactions for higher-order adjacency"""
    
    def __init__(self, d_model: int, wave_nodes: int, transition_nodes: int, target_nodes: int,
                 enable_group_interactions: bool = True):
        super().__init__()
        self.enable_group_interactions = enable_group_interactions
        self.d_model = d_model
        
        if self.enable_group_interactions:
            self.group_interaction_wave = nn.Parameter(torch.randn(wave_nodes, transition_nodes))
            self.group_interaction_transition = nn.Parameter(torch.randn(transition_nodes, target_nodes))
            self.group_interaction_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, wave_spatial: torch.Tensor, transition_nodes: int, 
                target_nodes: int, total_nodes: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Construct a higher-order adjacency capturing group interactions."""
        if not self.enable_group_interactions:
            return None
        
        required_attrs = (
            hasattr(self, 'group_interaction_wave'),
            hasattr(self, 'group_interaction_transition'),
            hasattr(self, 'group_interaction_scale'),
        )
        if not all(required_attrs):
            return None

        batch_size, wave_nodes, _ = wave_spatial.shape
        if self.group_interaction_wave.size(0) != wave_nodes:
            return None
        if (self.group_interaction_transition.size(0) != transition_nodes or 
            self.group_interaction_transition.size(1) != target_nodes):
            return None

        synergy = torch.einsum('bid,bjd->bij', wave_spatial, wave_spatial) / math.sqrt(self.d_model)
        wave_weights = torch.softmax(self.group_interaction_wave.to(wave_spatial.dtype), dim=0)
        wave_to_transition = torch.relu(torch.einsum('bij,jk->bik', synergy, wave_weights))

        transition_weights = torch.softmax(self.group_interaction_transition.to(wave_spatial.dtype), dim=0)
        # Aggregate wave_to_transition from [batch, wave_nodes, transition_nodes] to [batch, transition_nodes, target_nodes]
        # First sum over wave dimension to get transition features
        transition_features = wave_to_transition.sum(dim=1)  # [batch, transition_nodes]
        # Then expand and apply transition weights
        transition_context = torch.relu(torch.einsum('bi,ij->bij', transition_features, transition_weights))

        adjacency = wave_spatial.new_zeros((batch_size, total_nodes, total_nodes))
        wave_start = 0
        transition_start = wave_nodes
        target_start = wave_nodes + transition_nodes

        adjacency[:, wave_start:transition_start, transition_start:target_start] = wave_to_transition
        adjacency[:, transition_start:target_start, target_start:] = transition_context

        scale = torch.relu(self.group_interaction_scale).to(device=wave_spatial.device, dtype=wave_spatial.dtype)
        adjacency = adjacency * scale
        weights = adjacency.clone()
        return adjacency, weights


class PatchAggregator:
    """Aggregates multi-scale patch outputs"""
    
    @staticmethod
    def aggregate_patch_collection(patch_outputs: Optional[List[torch.Tensor]], 
                                 key_prefix: str, batch_size: int, device: torch.device, 
                                 dtype: torch.dtype, d_model: int, 
                                 projection_manager) -> torch.Tensor:
        """Aggregate multi-scale patch outputs into a fixed-size context vector."""
        if not patch_outputs:
            return torch.zeros(batch_size, d_model, device=device, dtype=dtype)

        projections: List[torch.Tensor] = []
        for idx, patch_output in enumerate(patch_outputs):
            if patch_output is None:
                continue
            summary = patch_output
            while summary.dim() > 2:
                summary = summary.mean(dim=1)
            summary = summary.reshape(batch_size, -1)
            projections.append(projection_manager.project_context_summary(summary, device))

        if not projections:
            return torch.zeros(batch_size, d_model, device=device, dtype=dtype)

        stacked = torch.stack(projections, dim=0).mean(dim=0)
        return stacked