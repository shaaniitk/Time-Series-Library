"""
Regime Pattern Expert

Specialized expert for detecting and modeling regime changes in time series.
Handles structural breaks, regime switches, and non-stationary patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List
import math

from ..base_expert import TemporalExpert, ExpertOutput


class RegimeDetector(nn.Module):
    """Neural regime detection module."""
    
    def __init__(self, d_model: int, num_regimes: int = 3, detection_window: int = 20):
        super().__init__()
        self.d_model = d_model
        self.num_regimes = num_regimes
        self.detection_window = detection_window
        
        # Change point detector
        self.change_detector = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=detection_window, padding=detection_window//2),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_regimes),
            nn.Softmax(dim=-1)
        )
        
        # Regime embeddings
        self.regime_embeddings = nn.Parameter(
            torch.randn(num_regimes, d_model) * 0.1
        )
        
        # Transition probability matrix
        self.transition_probs = nn.Parameter(
            torch.ones(num_regimes, num_regimes) / num_regimes
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect regimes and change points."""
        batch_size, seq_len, d_model = x.shape
        
        # Detect change points
        change_scores = self.change_detector(x.transpose(1, 2)).transpose(1, 2)
        
        # Classify regimes for each time step
        regime_probs = self.regime_classifier(x)  # [batch_size, seq_len, num_regimes]
        
        # Apply Viterbi-like smoothing using transition probabilities
        smoothed_regimes = self._smooth_regime_sequence(regime_probs)
        
        # Get most likely regime for each time step
        regime_assignments = torch.argmax(smoothed_regimes, dim=-1)
        
        return {
            'change_scores': change_scores,
            'regime_probabilities': regime_probs,
            'smoothed_regimes': smoothed_regimes,
            'regime_assignments': regime_assignments,
            'transition_matrix': F.softmax(self.transition_probs, dim=-1)
        }
    
    def _smooth_regime_sequence(self, regime_probs: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing to regime probabilities."""
        batch_size, seq_len, num_regimes = regime_probs.shape
        
        # Normalize transition probabilities
        trans_probs = F.softmax(self.transition_probs, dim=-1)
        
        # Forward pass (simplified Viterbi)
        smoothed = regime_probs.clone()
        
        for t in range(1, seq_len):
            # Compute transition-weighted probabilities
            prev_probs = smoothed[:, t-1:t, :].unsqueeze(-1)  # [batch, 1, num_regimes, 1]
            trans_matrix = trans_probs.unsqueeze(0).unsqueeze(0)  # [1, 1, num_regimes, num_regimes]
            
            # Weighted transition probabilities
            weighted_trans = prev_probs * trans_matrix  # [batch, 1, num_regimes, num_regimes]
            transition_contrib = weighted_trans.sum(dim=2)  # [batch, 1, num_regimes]
            
            # Combine with observation probabilities
            obs_probs = regime_probs[:, t:t+1, :]  # [batch, 1, num_regimes]
            smoothed[:, t:t+1, :] = 0.7 * obs_probs + 0.3 * transition_contrib
        
        return smoothed


class RegimeCharacterizer(nn.Module):
    """Characterize different regimes with distinct patterns."""
    
    def __init__(self, d_model: int, num_regimes: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_regimes = num_regimes
        
        # Regime-specific feature extractors
        self.regime_extractors = nn.ModuleList()
        for _ in range(num_regimes):
            extractor = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            )
            self.regime_extractors.append(extractor)
        
        # Regime attention mechanism
        self.regime_attention = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=0.1, batch_first=True
        )
        
    def forward(self, x: torch.Tensor, regime_probs: torch.Tensor) -> torch.Tensor:
        """Extract regime-specific features."""
        batch_size, seq_len, d_model = x.shape
        
        # Extract features for each regime
        regime_features = []
        for i, extractor in enumerate(self.regime_extractors):
            regime_feat = extractor(x)
            # Weight by regime probability
            regime_weight = regime_probs[:, :, i:i+1]  # [batch, seq_len, 1]
            weighted_feat = regime_feat * regime_weight
            regime_features.append(weighted_feat)
        
        # Combine regime features
        combined_features = sum(regime_features)
        
        # Apply regime attention
        attended_features, _ = self.regime_attention(
            combined_features, combined_features, combined_features
        )
        
        return attended_features


class RegimePatternExpert(TemporalExpert):
    """Expert specialized in regime pattern detection and modeling."""
    
    def __init__(self, config, num_regimes: Optional[int] = None):
        super().__init__(config, 'regime_expert')
        
        # Default number of regimes
        if num_regimes is None:
            num_regimes = getattr(config, 'num_regimes', 3)
        self.num_regimes = num_regimes
        
        # Regime detector
        self.regime_detector = RegimeDetector(
            self.d_model,
            num_regimes,
            getattr(config, 'regime_detection_window', 20)
        )
        
        # Regime characterizer
        self.regime_characterizer = RegimeCharacterizer(self.d_model, num_regimes)
        
        # Regime stability estimator
        self.stability_estimator = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Change point strength estimator
        self.change_strength = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),  # original + regime features
            nn.ReLU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout)
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.expert_dim)
        
    def forward(self, x: torch.Tensor, **kwargs) -> ExpertOutput:
        """Process input through regime pattern expert."""
        batch_size, seq_len, d_model = x.shape
        
        # 1. Regime detection
        regime_results = self.regime_detector(x)
        regime_probs = regime_results['smoothed_regimes']
        change_scores = regime_results['change_scores']
        
        # 2. Regime characterization
        regime_features = self.regime_characterizer(x, regime_probs)
        
        # 3. Feature fusion
        combined_features = torch.cat([x, regime_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # 4. Apply temporal convolution and normalization
        temporal_features = self.temporal_conv(fused_features.transpose(1, 2)).transpose(1, 2)
        temporal_features = self.temporal_norm(temporal_features)
        
        # 5. Output projection
        output = self.output_proj(temporal_features)
        
        # 6. Compute confidence based on regime stability
        stability_scores = self.stability_estimator(fused_features)
        confidence = stability_scores
        
        # 7. Estimate change point strength
        change_strength_scores = self.change_strength(fused_features)
        
        # 8. Compile metadata
        dominant_regime = regime_probs.mean(dim=1).argmax(dim=-1)  # [batch_size]
        
        metadata = {
            'num_regimes': self.num_regimes,
            'regime_probabilities': regime_probs.mean(dim=1),  # Average over time
            'dominant_regime': dominant_regime,
            'change_point_scores': change_scores.mean().item(),
            'regime_stability': stability_scores.mean().item(),
            'transition_matrix': regime_results['transition_matrix'],
            'detected_change_points': self._detect_change_points(change_scores)
        }
        
        return ExpertOutput(
            output=output,
            confidence=confidence,
            metadata=metadata,
            attention_weights=None,
            uncertainty=change_strength_scores  # Use change strength as uncertainty
        )
    
    def _detect_change_points(self, change_scores: torch.Tensor, threshold: float = 0.7) -> List[List[int]]:
        """Detect significant change points from change scores."""
        batch_size, seq_len, _ = change_scores.shape
        change_points = []
        
        for batch_idx in range(batch_size):
            batch_scores = change_scores[batch_idx, :, 0]  # [seq_len]
            significant_changes = torch.where(batch_scores > threshold)[0]
            change_points.append(significant_changes.cpu().tolist())
        
        return change_points
    
    def get_regime_summary(self, x: torch.Tensor) -> Dict[str, Any]:
        """Get comprehensive regime analysis summary."""
        with torch.no_grad():
            regime_results = self.regime_detector(x)
            
            # Analyze regime characteristics
            regime_probs = regime_results['smoothed_regimes']
            regime_assignments = regime_results['regime_assignments']
            
            # Calculate regime statistics
            regime_durations = []
            for batch_idx in range(x.size(0)):
                batch_assignments = regime_assignments[batch_idx]
                durations = self._calculate_regime_durations(batch_assignments)
                regime_durations.append(durations)
            
            return {
                'regime_probabilities': regime_probs,
                'regime_assignments': regime_assignments,
                'regime_durations': regime_durations,
                'transition_matrix': regime_results['transition_matrix'],
                'change_points': self._detect_change_points(regime_results['change_scores']),
                'regime_stability': regime_probs.max(dim=-1)[0].mean(dim=1)  # Average max probability
            }
    
    def _calculate_regime_durations(self, assignments: torch.Tensor) -> Dict[int, List[int]]:
        """Calculate duration of each regime."""
        durations = {i: [] for i in range(self.num_regimes)}
        
        current_regime = assignments[0].item()
        current_duration = 1
        
        for t in range(1, len(assignments)):
            if assignments[t].item() == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                current_regime = assignments[t].item()
                current_duration = 1
        
        # Add final duration
        durations[current_regime].append(current_duration)
        
        return durations