"""
Celestial-to-Target Attention Module

Explicit bridge between celestial covariates (13 bodies) and market targets (4 assets).
Each target learns to attend to all 13 celestial bodies independently, enabling
measurement of covariate impact: "Which celestial bodies influence Bitcoin vs Ethereum?"

Architecture:
- 13 celestial covariate nodes (from Petri net graph processing)
- 4 target nodes (decoder features for each asset)
- Per-target attention: each target attends to all 13 celestial bodies
- Gated fusion: learned gates control how much celestial influence each target accepts

Key Features:
1. Target-specific celestial attention (separate for each target)
2. Interpretable attention weights (diagnostic output)
3. Normalized fusion (prevents magnitude imbalance)
4. Residual connections (preserves gradient flow)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging


class CelestialToTargetAttention(nn.Module):
    """
    Explicit attention-based bridge: 13 celestial covariates → 4 targets
    
    Each target independently attends to all celestial bodies, learning
    which covariates are most relevant for predicting that specific target.
    
    Args:
        num_celestial: Number of celestial body nodes (13)
        num_targets: Number of prediction targets (4)
        d_model: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_gated_fusion: Whether to use learned gates for fusion
        enable_diagnostics: Whether to collect attention diagnostics
    """
    
    def __init__(
        self,
        num_celestial: int = 13,
        num_targets: int = 4,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gated_fusion: bool = True,
        enable_diagnostics: bool = True,
        preserve_temporal: bool = True,
        use_full_temporal_context: bool = False,
        use_edge_bias: bool = False,
        edge_bias_scale: float = 1.0,
    ):
        super().__init__()
        self.num_celestial = num_celestial
        self.num_targets = num_targets
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_gated_fusion = use_gated_fusion
        self.enable_diagnostics = enable_diagnostics
        self.preserve_temporal = preserve_temporal
        self.use_full_temporal_context = use_full_temporal_context
        self.use_edge_bias = use_edge_bias
        self.edge_bias_scale = edge_bias_scale
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Per-target query projections (each target has its own query space)
        self.target_query_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_targets)
        ])
        
        # Shared celestial key/value projections (all targets see same celestial space)
        self.celestial_key_projection = nn.Linear(d_model, d_model)
        self.celestial_value_projection = nn.Linear(d_model, d_model)
        
        # Per-target multi-head attention
        self.target_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_targets)
        ])
        
        # FIX ISSUE #6: Gate entropy regularization to prevent saturation
        self.gate_entropy_weight = 0.01  # Regularization weight
        self.gate_init_gain = 0.1  # Small initial weights for gates
        
        # Optional gated fusion (learn how much celestial influence to accept)
        if use_gated_fusion:
            self.fusion_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model * 2, d_model),  # target + celestial_influence
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                    nn.Sigmoid()
                )
                for _ in range(num_targets)
            ])
            
            # Initialize gate final layers with small weights
            for gate in self.fusion_gates:
                # gate[-2] is the final Linear layer before Sigmoid
                if hasattr(gate[-2], 'weight'):
                    nn.init.xavier_uniform_(gate[-2].weight, gain=self.gate_init_gain)
                    if hasattr(gate[-2], 'bias') and gate[-2].bias is not None:
                        nn.init.constant_(gate[-2].bias, 0.0)
        
        # LayerNorms for stable fusion
        self.target_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_targets)
        ])
        self.celestial_influence_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_targets)
        ])
        
        # Output projection (optional refinement)
        self.output_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )
            for _ in range(num_targets)
        ])
        
        # Diagnostic storage
        self.latest_attention_weights: Optional[Dict[str, torch.Tensor]] = None
        self.latest_gate_values: Optional[Dict[str, torch.Tensor]] = None
        
    def forward(
        self,
        target_features: torch.Tensor,
        celestial_features: torch.Tensor,
        edge_prior: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Apply celestial-to-target attention
        
        Args:
            target_features: [batch, pred_len, num_targets, d_model]
                Features for each target (from decoder or target-specific encoder)
            celestial_features: [batch, seq_len, num_celestial, d_model]
                Features for each celestial body (from Petri net graph)
            return_diagnostics: Whether to return attention weights and gate values
        
        Returns:
            Tuple of:
                - Enhanced target features [batch, pred_len, num_targets, d_model]
                - Optional diagnostics dict with attention weights and gate stats
        """
        batch_size, pred_len, num_targets_in, d_model = target_features.shape
        batch_size_cel, seq_len, num_celestial_in, d_model_cel = celestial_features.shape
        
        # Validate dimensions
        assert num_targets_in == self.num_targets, f"Expected {self.num_targets} targets, got {num_targets_in}"
        assert num_celestial_in == self.num_celestial, f"Expected {self.num_celestial} celestial bodies, got {num_celestial_in}"
        assert batch_size == batch_size_cel, "Batch sizes must match"
        
        # Strategy for building keys/values from celestial sequence:
        # - If preserve_temporal and not use_full_temporal_context: align per-step
        #   keys/values using the last pred_len timesteps (no pooling).
        # - If preserve_temporal and use_full_temporal_context: each pred step
        #   attends over the entire history across all celestial nodes (seq_len * num_celestial tokens).
        # - Else: fallback to mean pooling across time (legacy behavior).

        # Precompute projected celestial keys/values across time
        # celestial_seq: [B, S, C, D]
        celestial_keys_seq = self.celestial_key_projection(celestial_features)
        celestial_vals_seq = self.celestial_value_projection(celestial_features)
        
        # Process each target independently
        enhanced_targets = []
        attention_weights_dict = {}
        gate_values_dict = {}
        
        for target_idx in range(self.num_targets):
            # Extract this target's features across all prediction timesteps
            target_i = target_features[:, :, target_idx, :]  # [batch, pred_len, d_model]
            
            # Project to query space (target-specific)
            query_i = self.target_query_projections[target_idx](target_i)
            # Shape: [batch, pred_len, d_model]
            
            # Build keys/values per configuration
            if self.preserve_temporal and not self.use_full_temporal_context:
                # Time-aligned per-step attention: use the last pred_len timesteps
                if seq_len >= pred_len:
                    keys_step = celestial_keys_seq[:, -pred_len:, :, :]   # [B, L, C, D]
                    vals_step = celestial_vals_seq[:, -pred_len:, :, :]
                else:
                    # If history shorter than pred_len, repeat the last timestep
                    last_k = celestial_keys_seq[:, -1:, :, :].expand(-1, pred_len, -1, -1)
                    last_v = celestial_vals_seq[:, -1:, :, :].expand(-1, pred_len, -1, -1)
                    keys_step, vals_step = last_k, last_v

                # Flatten: [B, L, C, D] -> [B*L, C, D]
                keys_flat = keys_step.reshape(batch_size * pred_len, self.num_celestial, d_model)
                values_flat = vals_step.reshape(batch_size * pred_len, self.num_celestial, d_model)
                query_flat = query_i.reshape(batch_size * pred_len, 1, d_model)

                # Optional additive attention bias derived from edges
                attn_mask = None
                if self.use_edge_bias and edge_prior is not None:
                    # edge_prior expected shape: [B, L, C]
                    try:
                        if edge_prior.dim() == 2:
                            # Allow [B, C] broadcast over L
                            edge_prior_exp = edge_prior.unsqueeze(1).expand(-1, pred_len, -1)
                        else:
                            edge_prior_exp = edge_prior
                        # Standardize per-step for stability (zero-mean)
                        ep = edge_prior_exp - edge_prior_exp.mean(dim=-1, keepdim=True)
                        # Scale and flatten to [B*L, 1, C]
                        ep_flat = (self.edge_bias_scale * ep).reshape(batch_size * pred_len, 1, self.num_celestial)

                        # PyTorch MHA requires attn_mask of shape (N*num_heads, L, S) for 3D
                        # Here N=B*L, L=1, S=C
                        ep_flat_heads = ep_flat.repeat_interleave(self.num_heads, dim=0)
                        attn_mask = ep_flat_heads  # [B*L*num_heads, 1, C]
                    except Exception as _exc:
                        # Never fail forward due to bias issues
                        self.logger.debug("C2T edge_bias construction failed: %s", str(_exc))
                        attn_mask = None

            elif self.preserve_temporal and self.use_full_temporal_context:
                # Each pred step attends over entire history across all celestial nodes
                # Collapse temporal+node dims: [B, S, C, D] -> [B, S*C, D]
                source_len = seq_len * self.num_celestial
                keys_all = celestial_keys_seq.reshape(batch_size, source_len, d_model)
                vals_all = celestial_vals_seq.reshape(batch_size, source_len, d_model)

                # Repeat per pred step for each sample: [B, S*C, D] -> [B*L, S*C, D]
                keys_flat = keys_all.unsqueeze(1).expand(-1, pred_len, -1, -1).reshape(
                    batch_size * pred_len, source_len, d_model
                )
                values_flat = vals_all.unsqueeze(1).expand(-1, pred_len, -1, -1).reshape(
                    batch_size * pred_len, source_len, d_model
                )
                query_flat = query_i.reshape(batch_size * pred_len, 1, d_model)
                attn_mask = None  # Not supported for full-context in v1 (S=C*S)

            else:
                # Legacy pooled behavior (temporal information compressed)
                celestial_pooled = celestial_features.mean(dim=1)  # [B, C, D]
                keys_expanded = celestial_pooled.unsqueeze(1).expand(-1, pred_len, -1, -1)
                values_expanded = celestial_pooled.unsqueeze(1).expand(-1, pred_len, -1, -1)
                query_flat = query_i.reshape(batch_size * pred_len, 1, d_model)
                keys_flat = keys_expanded.reshape(batch_size * pred_len, self.num_celestial, d_model)
                values_flat = values_expanded.reshape(batch_size * pred_len, self.num_celestial, d_model)
            
            # Apply multi-head attention
            celestial_influence_flat, attn_weights_flat = self.target_attentions[target_idx](
                query_flat, keys_flat, values_flat,
                attn_mask=attn_mask
            )
            # celestial_influence_flat: [batch*pred_len, 1, d_model]
            # attn_weights_flat: [batch*pred_len, 1, source_len]
            
            # Reshape back
            celestial_influence = celestial_influence_flat.squeeze(1).reshape(batch_size, pred_len, d_model)
            # Determine source length for attention weights based on mode
            if self.preserve_temporal and self.use_full_temporal_context:
                src_len = seq_len * self.num_celestial
            else:
                src_len = self.num_celestial
            attn_weights = attn_weights_flat.squeeze(1).reshape(batch_size, pred_len, src_len)
            # Shapes: [batch, pred_len, d_model] and [batch, pred_len, src_len]
            
            # Normalize celestial influence
            celestial_influence = self.celestial_influence_norms[target_idx](celestial_influence)
            target_i_normalized = self.target_norms[target_idx](target_i)
            
            # Gated fusion (optional)
            if self.use_gated_fusion:
                gate_input = torch.cat([target_i_normalized, celestial_influence], dim=-1)
                fusion_gate = self.fusion_gates[target_idx](gate_input)
                # Shape: [batch, pred_len, d_model]
                
                # Apply gate
                celestial_influence_gated = fusion_gate * celestial_influence
                
                # Store gate values for diagnostics
                if self.enable_diagnostics:
                    gate_values_dict[f'target_{target_idx}_gate'] = fusion_gate.detach()
            else:
                celestial_influence_gated = celestial_influence
            
            # Residual connection
            enhanced_target = target_i_normalized + celestial_influence_gated
            
            # Optional output refinement
            enhanced_target = self.output_projections[target_idx](enhanced_target)
            
            # Final residual
            enhanced_target = enhanced_target + target_i
            
            enhanced_targets.append(enhanced_target)
            
            # Store attention weights for diagnostics
            if self.enable_diagnostics:
                attention_weights_dict[f'target_{target_idx}_attn'] = attn_weights.detach()
        
        # Stack enhanced targets
        enhanced_target_features = torch.stack(enhanced_targets, dim=2)
        # Shape: [batch, pred_len, num_targets, d_model]
        
        # Store latest diagnostics
        if self.enable_diagnostics:
            self.latest_attention_weights = attention_weights_dict
            self.latest_gate_values = gate_values_dict
        
        # Prepare diagnostics output
        diagnostics = None
        if return_diagnostics and self.enable_diagnostics:
            diagnostics = {
                'attention_weights': attention_weights_dict,
                'gate_values': gate_values_dict,
                'summary': self._compute_diagnostics_summary(attention_weights_dict, gate_values_dict)
            }
        
        # FIX ISSUE #6: Compute gate entropy regularization loss
        gate_entropy_loss = 0.0
        if self.use_gated_fusion and self.training and len(gate_values_dict) > 0:
            # Encourage gates to maintain diversity (avoid saturation to 0 or 1)
            for gate_values in gate_values_dict.values():
                # gate_values: [batch, pred_len, d_model]
                # Compute binary entropy: -p*log(p) - (1-p)*log(1-p)
                eps = 1e-8
                p = gate_values.clamp(eps, 1-eps)
                entropy_per_element = -(p * torch.log(p) + (1-p) * torch.log(1-p))
                # Negative entropy loss (we want to MAXIMIZE entropy, i.e., minimize negative entropy)
                gate_entropy_loss += -entropy_per_element.mean() * self.gate_entropy_weight
        
        # Add gate entropy loss to diagnostics
        if return_diagnostics and diagnostics is not None:
            diagnostics['gate_entropy_loss'] = gate_entropy_loss.item() if isinstance(gate_entropy_loss, torch.Tensor) else gate_entropy_loss
        
        return enhanced_target_features, diagnostics, gate_entropy_loss
    
    def _compute_diagnostics_summary(
        self,
        attention_weights: Dict[str, torch.Tensor],
        gate_values: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute summary statistics for diagnostics"""
        summary = {}
        
        # Attention weights summary
        for key, attn in attention_weights.items():
            # attn: [batch, pred_len, num_celestial]
            summary[f'{key}_mean'] = attn.mean().item()
            summary[f'{key}_std'] = attn.std().item()
            summary[f'{key}_max'] = attn.max().item()
            summary[f'{key}_entropy'] = self._compute_entropy(attn).item()
        
        # Gate values summary (if available)
        for key, gate in gate_values.items():
            # gate: [batch, pred_len, d_model]
            summary[f'{key}_mean'] = gate.mean().item()
            summary[f'{key}_std'] = gate.std().item()
        
        return summary
    
    @staticmethod
    def _compute_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
        """Compute attention entropy (lower = more focused, higher = more diffuse)"""
        # Avoid log(0) by adding small epsilon
        eps = 1e-8
        entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1).mean()
        return entropy
    
    def get_attention_diagnostics(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get latest attention weights (for external analysis)"""
        return self.latest_attention_weights
    
    def get_gate_diagnostics(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get latest gate values (for external analysis)"""
        return self.latest_gate_values
    
    def print_diagnostics_summary(self):
        """Print human-readable diagnostics summary"""
        if not self.enable_diagnostics or self.latest_attention_weights is None:
            self.logger.info("No diagnostics available (either disabled or forward not yet called)")
            return
        
        self.logger.info("=" * 80)
        self.logger.info("Celestial-to-Target Attention Diagnostics")
        self.logger.info("=" * 80)
        
        # Attention weights analysis
        for target_idx in range(self.num_targets):
            attn_key = f'target_{target_idx}_attn'
            if attn_key in self.latest_attention_weights:
                attn = self.latest_attention_weights[attn_key]
                # Average over batch and time to get [num_celestial] weights
                attn_mean = attn.mean(dim=(0, 1))  # [num_celestial]
                
                self.logger.info(f"\nTarget {target_idx} attention to celestial bodies:")
                for cel_idx in range(self.num_celestial):
                    self.logger.info(f"  Celestial {cel_idx}: {attn_mean[cel_idx]:.4f}")
                
                entropy = self._compute_entropy(attn).item()
                self.logger.info(f"  Attention entropy: {entropy:.4f} (lower = more focused)")
        
        # Gate values analysis (if available)
        if self.use_gated_fusion and self.latest_gate_values:
            self.logger.info("\nGate statistics (how much celestial influence accepted):")
            for target_idx in range(self.num_targets):
                gate_key = f'target_{target_idx}_gate'
                if gate_key in self.latest_gate_values:
                    gate = self.latest_gate_values[gate_key]
                    self.logger.info(
                        f"  Target {target_idx}: mean={gate.mean():.4f} std={gate.std():.4f}"
                    )
        
        self.logger.info("=" * 80)
