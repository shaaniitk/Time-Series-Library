"""
Celestial Graph Combiner - Hierarchical Attention Fusion

Sophisticated combination of astronomical, learned, and attention-based edge relationships
using hierarchical attention mechanisms and cross-modal interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import math
from layers.modular.graph.registry import GraphComponentRegistry

@GraphComponentRegistry.register("celestial_graph_combiner")
class CelestialGraphCombiner(nn.Module):
    """
    Advanced graph combiner using hierarchical attention fusion
    
    Combines three types of edges:
    1. Astronomical edges - Fixed astrological relationships
    2. Learned edges - Data-driven patterns  
    3. Attention edges - Dynamic attention-based connections
    
    Uses sophisticated fusion beyond simple addition.
    """
    
    def __init__(
        self, 
        num_nodes: int,
        d_model: int, 
        num_attention_heads: int = 8,
        fusion_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.fusion_layers = fusion_layers
        self.dropout = dropout
        
        # Edge type embeddings
        self.edge_type_embeddings = nn.Parameter(
            torch.randn(3, d_model) * 0.02  # [astronomical, learned, attention]
        )
        
        # Edge transformation networks
        self.edge_transforms = nn.ModuleDict({
            'astronomical': self._build_edge_transform(),
            'learned': self._build_edge_transform(), 
            'attention': self._build_edge_transform()
        })
        
        # Multi-head attention for edge type fusion
        self.edge_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Hierarchical fusion layers
        self.fusion_layers_net = nn.ModuleList([
            HierarchicalFusionLayer(d_model, num_attention_heads, dropout)
            for _ in range(fusion_layers)
        ])
        
        # Cross-interaction networks
        self.pairwise_interactions = nn.ModuleDict({
            'astro_learned': CrossInteractionLayer(d_model),
            'astro_attention': CrossInteractionLayer(d_model),
            'learned_attention': CrossInteractionLayer(d_model)
        })
        
        # Market regime adaptation
        self.regime_adapter = MarketRegimeAdapter(d_model, num_regimes=4)
        
        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Tanh()  # Bounded output for edge weights
        )
        
        # Learnable combination weights
        self.combination_weights = nn.Parameter(torch.ones(3) / 3)  # Equal initial weights
        
        # Edge strength calibration
        self.edge_calibrator = EdgeStrengthCalibrator(d_model)
        
    def _build_edge_transform(self) -> nn.Module:
        """Build transformation network for each edge type"""
        return nn.Sequential(
            nn.Linear(1, self.d_model // 4),
            nn.GELU(),
            nn.Linear(self.d_model // 4, self.d_model // 2),
            nn.GELU(), 
            nn.Linear(self.d_model // 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
    
    def forward(
        self,
        astronomical_edges: torch.Tensor,
        learned_edges: torch.Tensor, 
        attention_edges: torch.Tensor,
        enc_out: torch.Tensor, # Changed from market_context
        market_regime: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with hierarchical attention fusion, now adapted for dynamic inputs.
        
        Args:
            astronomical_edges: [batch, seq_len, num_nodes, num_nodes]
            learned_edges: [batch, seq_len, num_nodes, num_nodes]
            attention_edges: [batch, seq_len, num_nodes, num_nodes]
            enc_out: [batch, seq_len, d_model] Full encoder output
            market_regime: [batch] Optional market regime indicators
            
        Returns:
            Tuple of:
            - combined_edges: [batch, seq_len, num_nodes, num_nodes] Final combined adjacency
            - metadata: Dict with fusion information from the last time step
        """
        # DEBUG: Memory and shape logging at start
        import torch
        import psutil
        import os
        
        def log_memory(stage):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"🔍 COMBINER [{stage}]: GPU Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
            else:
                process = psutil.Process(os.getpid())
                cpu_mem = process.memory_info().rss / 1024**3
                print(f"🔍 COMBINER [{stage}]: CPU={cpu_mem:.2f}GB")
        
        batch_size, seq_len, _, _ = astronomical_edges.shape
        print(f"🔍 COMBINER START: Processing {seq_len} timesteps, batch_size={batch_size}")
        log_memory("START")
        
        combined_edges_over_time = []
        last_metadata = {}

        # Iterate over each time step
        for t in range(seq_len):
            if t % 50 == 0 or t < 5:  # Log first 5 and every 50th timestep
                print(f"🔍 COMBINER: Processing timestep {t}/{seq_len}")
                log_memory(f"TIMESTEP_{t}")
            
            # Slice inputs for the current time step to make them static
            astro_t = astronomical_edges[:, t, :, :]
            learned_t = learned_edges[:, t, :, :]
            attn_t = attention_edges[:, t, :, :]
            context_t = enc_out[:, t, :] # Use the encoder output for this step as context

            if t == 0:  # Debug shapes for first timestep
                print(f"🔍 COMBINER SHAPES: astro_t={astro_t.shape}, learned_t={learned_t.shape}, attn_t={attn_t.shape}, context_t={context_t.shape}")

            # Call the original fusion logic for this single time step
            try:
                combined_t, metadata_t = self._forward_static(
                    astro_t, learned_t, attn_t, context_t, market_regime
                )
                combined_edges_over_time.append(combined_t)
                last_metadata = metadata_t
                
                if t == 0:
                    print(f"🔍 COMBINER: First timestep output shape: {combined_t.shape}")
                    log_memory("AFTER_FIRST_TIMESTEP")
                
            except Exception as e:
                print(f"❌ COMBINER ERROR at timestep {t}: {e}")
                log_memory(f"ERROR_TIMESTEP_{t}")
                raise

        print(f"🔍 COMBINER: About to stack {len(combined_edges_over_time)} tensors")
        log_memory("BEFORE_STACK")
        
        # Stack the results from all time steps
        combined_edges = torch.stack(combined_edges_over_time, dim=1)
        
        print(f"🔍 COMBINER: Final output shape: {combined_edges.shape}")
        log_memory("END")
        
        return combined_edges, last_metadata

    def _forward_static(
        self,
        astronomical_edges: torch.Tensor,
        learned_edges: torch.Tensor, 
        attention_edges: torch.Tensor,
        market_context: torch.Tensor,
        market_regime: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Original forward pass logic, now used for a single time step.
        """
        # DEBUG: Track memory through each step
        import torch
        import psutil
        import os
        
        def log_step_memory(step_name):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"    🔍 STATIC [{step_name}]: GPU={allocated:.2f}GB")
            else:
                process = psutil.Process(os.getpid())
                cpu_mem = process.memory_info().rss / 1024**3
                print(f"    🔍 STATIC [{step_name}]: CPU={cpu_mem:.2f}GB")
        
        log_step_memory("START")
        
        # 1. Transform each edge type to feature space
        try:
            edge_features = self._transform_edges_to_features(
                astronomical_edges, learned_edges, attention_edges
            )  # [batch, 3, num_nodes, num_nodes, d_model]
            print(f"    🔍 STATIC: edge_features shape: {edge_features.shape}")
            log_step_memory("AFTER_EDGE_TRANSFORM")
        except Exception as e:
            print(f"    ❌ STATIC ERROR in edge transform: {e}")
            log_step_memory("ERROR_EDGE_TRANSFORM")
            raise
        
        # 2. Hierarchical attention fusion (returns [batch, num_nodes, num_nodes, d_model])
        try:
            fused_features = self._hierarchical_fusion(edge_features, market_context)
            print(f"    🔍 STATIC: fused_features shape: {fused_features.shape}")
            log_step_memory("AFTER_HIERARCHICAL_FUSION")
        except Exception as e:
            print(f"    ❌ STATIC ERROR in hierarchical fusion: {e}")
            log_step_memory("ERROR_HIERARCHICAL_FUSION")
            raise
        
        # 3. Cross-modal interactions
        try:
            interaction_features = self._compute_cross_interactions(
                astronomical_edges, learned_edges, attention_edges, market_context
            )  # [batch, num_nodes, num_nodes, d_model]
            print(f"    🔍 STATIC: interaction_features shape: {interaction_features.shape}")
            log_step_memory("AFTER_CROSS_INTERACTIONS")
        except Exception as e:
            print(f"    ❌ STATIC ERROR in cross interactions: {e}")
            log_step_memory("ERROR_CROSS_INTERACTIONS")
            raise
        
        # 4. Market regime adaptation
        if market_regime is not None:
            try:
                regime_adapted = self.regime_adapter(fused_features, market_regime)
                log_step_memory("AFTER_REGIME_ADAPTATION")
            except Exception as e:
                print(f"    ❌ STATIC ERROR in regime adaptation: {e}")
                log_step_memory("ERROR_REGIME_ADAPTATION")
                raise
        else:
            regime_adapted = fused_features
        
        # 5. Combine base fusion with interactions
        try:
            combined_features = regime_adapted + interaction_features
            print(f"    🔍 STATIC: combined_features shape: {combined_features.shape}")
            log_step_memory("AFTER_FEATURE_COMBINATION")
        except Exception as e:
            print(f"    ❌ STATIC ERROR in feature combination: {e}")
            log_step_memory("ERROR_FEATURE_COMBINATION")
            raise
        
        # 6. Edge strength calibration
        try:
            calibrated_features = self.edge_calibrator(combined_features, market_context)
            print(f"    🔍 STATIC: calibrated_features shape: {calibrated_features.shape}")
            log_step_memory("AFTER_EDGE_CALIBRATION")
        except Exception as e:
            print(f"    ❌ STATIC ERROR in edge calibration: {e}")
            log_step_memory("ERROR_EDGE_CALIBRATION")
            raise
        
        # 7. Final projection to edge weights
        try:
            combined_edges = self.output_projection(calibrated_features).squeeze(-1)
            # [batch, num_nodes, num_nodes]
            print(f"    🔍 STATIC: combined_edges shape: {combined_edges.shape}")
            log_step_memory("AFTER_OUTPUT_PROJECTION")
        except Exception as e:
            print(f"    ❌ STATIC ERROR in output projection: {e}")
            log_step_memory("ERROR_OUTPUT_PROJECTION")
            raise
        
        # 8. Generate metadata for interpretability
        try:
            metadata = self._generate_metadata(
                astronomical_edges, learned_edges, attention_edges, 
                combined_edges, market_context
            )
            log_step_memory("AFTER_METADATA_GENERATION")
        except Exception as e:
            print(f"    ❌ STATIC ERROR in metadata generation: {e}")
            log_step_memory("ERROR_METADATA_GENERATION")
            raise
        
        return combined_edges, metadata
    
    def _transform_edges_to_features(
        self, 
        astronomical_edges: torch.Tensor,
        learned_edges: torch.Tensor,
        attention_edges: torch.Tensor
    ) -> torch.Tensor:
        """Transform edge adjacencies to feature representations"""
        
        batch_size, num_nodes, _ = astronomical_edges.shape
        
        # Flatten edges for transformation
        astro_flat = astronomical_edges.view(batch_size, -1, 1)  # [batch, N*N, 1]
        learned_flat = learned_edges.view(batch_size, -1, 1)
        attention_flat = attention_edges.view(batch_size, -1, 1)
        
        # Transform to feature space
        astro_features = self.edge_transforms['astronomical'](astro_flat)
        learned_features = self.edge_transforms['learned'](learned_flat)  
        attention_features = self.edge_transforms['attention'](attention_flat)
        # Each: [batch, N*N, d_model]
        
        # Reshape back to adjacency format
        astro_features = astro_features.view(batch_size, num_nodes, num_nodes, self.d_model)
        learned_features = learned_features.view(batch_size, num_nodes, num_nodes, self.d_model)
        attention_features = attention_features.view(batch_size, num_nodes, num_nodes, self.d_model)
        
        # Stack edge types
        edge_features = torch.stack([astro_features, learned_features, attention_features], dim=1)
        # [batch, 3, num_nodes, num_nodes, d_model]
        
        return edge_features
    
    def _hierarchical_fusion(
        self, 
        edge_features: torch.Tensor, 
        market_context: torch.Tensor
    ) -> torch.Tensor:
        """Apply hierarchical fusion layers"""
        
        batch_size, num_edge_types, num_nodes, _, d_model = edge_features.shape
        
        # Flatten spatial dimensions for processing
        features_flat = edge_features.view(batch_size, num_edge_types, -1, d_model)
        # [batch, 3, N*N, d_model]
        
        # Apply fusion layers
        current_features = features_flat
        for fusion_layer in self.fusion_layers_net:
            current_features = fusion_layer(current_features, market_context)
        
        # Aggregate across edge types with learned attention
        aggregated = self._aggregate_edge_types(current_features, market_context)
        # [batch, N*N, d_model]
        
        # Reshape back to adjacency format
        fused_features = aggregated.view(batch_size, num_nodes, num_nodes, d_model)
        
        return fused_features
    
    def _aggregate_edge_types(
        self, 
        edge_features: torch.Tensor, 
        market_context: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate across edge types using attention"""
        
        batch_size, num_edge_types, seq_len, d_model = edge_features.shape
        
        # Prepare for attention: [seq_len, batch, d_model] format
        features_for_attention = edge_features.permute(2, 0, 1, 3).contiguous()
        # [N*N, batch, 3, d_model]
        
        features_for_attention = features_for_attention.view(seq_len, batch_size * num_edge_types, d_model)
        
        # Use market context as query
        market_query = market_context.unsqueeze(0).expand(seq_len, -1, -1)  # [N*N, batch, d_model]
        
        # Apply attention
        attended_features, attention_weights = self.edge_attention(
            query=market_query,
            key=features_for_attention,
            value=features_for_attention
        )  # [N*N, batch, d_model]
        
        # Reshape back
        aggregated = attended_features.permute(1, 0, 2)  # [batch, N*N, d_model]
        
        return aggregated
    
    def _compute_cross_interactions(
        self,
        astronomical_edges: torch.Tensor,
        learned_edges: torch.Tensor,
        attention_edges: torch.Tensor, 
        market_context: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise interactions between edge types"""
        
        batch_size, num_nodes, _ = astronomical_edges.shape
        
        # Compute pairwise interactions
        astro_learned = self.pairwise_interactions['astro_learned'](
            astronomical_edges, learned_edges, market_context
        )
        
        astro_attention = self.pairwise_interactions['astro_attention'](
            astronomical_edges, attention_edges, market_context
        )
        
        learned_attention = self.pairwise_interactions['learned_attention'](
            learned_edges, attention_edges, market_context
        )
        
        # Combine interactions
        total_interactions = astro_learned + astro_attention + learned_attention
        # [batch, num_nodes, num_nodes, d_model]
        
        return total_interactions
    
    def _generate_metadata(
        self,
        astronomical_edges: torch.Tensor,
        learned_edges: torch.Tensor, 
        attention_edges: torch.Tensor,
        combined_edges: torch.Tensor,
        market_context: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generate interpretability metadata"""
        
        # Edge type contributions
        astro_strength = astronomical_edges.abs().mean()
        learned_strength = learned_edges.abs().mean()
        attention_strength = attention_edges.abs().mean()
        
        total_strength = astro_strength + learned_strength + attention_strength
        
        metadata = {
            'astronomical_contribution': astro_strength / total_strength,
            'learned_contribution': learned_strength / total_strength,
            'attention_contribution': attention_strength / total_strength,
            'combination_weights': F.softmax(self.combination_weights, dim=0),
            'final_edge_density': (combined_edges.abs() > 0.1).float().mean(),
            'market_context_norm': market_context.norm(dim=-1).mean()
        }
        
        return metadata


class HierarchicalFusionLayer(nn.Module):
    """Single layer of hierarchical fusion"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, edge_features: torch.Tensor, market_context: torch.Tensor) -> torch.Tensor:
        """Memory-efficient hierarchical fusion over edge features.

        Args:
            edge_features: [batch, num_edge_types, seq_len, d_model] flattened adjacency features.
            market_context: [batch, d_model]

        Returns:
            torch.Tensor: [batch, num_edge_types, seq_len, d_model] refined edge features.
        """
        batch_size, num_edge_types, seq_len, d_model = edge_features.shape
        num_nodes = math.isqrt(seq_len)
        has_square_layout = num_nodes > 0 and num_nodes * num_nodes == seq_len

        if has_square_layout:
            # Reshape to structured adjacency for per-node grouping
            adjacency = edge_features.view(batch_size, num_edge_types, num_nodes, num_nodes, d_model)

            # Group edges per source node to keep attention windows small (tokens = edge_types * num_nodes)
            node_grouped = adjacency.permute(0, 2, 1, 3, 4).contiguous()
            node_grouped = node_grouped.view(batch_size * num_nodes, num_edge_types * num_nodes, d_model)

            # Self-attention within each node group
            attended, _ = self.self_attention(node_grouped, node_grouped, node_grouped)
            node_grouped = self.norm1(node_grouped + self.dropout(attended))

            # Cross-attention with market context (broadcast context per group)
            market_context_expanded = market_context.unsqueeze(1).expand(batch_size, num_nodes, d_model)
            market_context_flat = market_context_expanded.reshape(batch_size * num_nodes, 1, d_model)
            market_context_broadcast = market_context_flat.expand(-1, num_edge_types * num_nodes, -1)
            cross_attended, _ = self.cross_attention(node_grouped, market_context_broadcast, market_context_broadcast)
            node_grouped = self.norm2(node_grouped + self.dropout(cross_attended))

            # Feed-forward refinement
            ff_output = self.feed_forward(node_grouped)
            node_grouped = self.norm3(node_grouped + self.dropout(ff_output))

            # Restore original flattened layout
            node_grouped = node_grouped.view(batch_size, num_nodes, num_edge_types, num_nodes, d_model)
            fused = node_grouped.permute(0, 2, 1, 3, 4).contiguous()
            return fused.view(batch_size, num_edge_types, seq_len, d_model)

        # Fallback: treat sequence generically when adjacency layout is unavailable
        tokens_grouped = edge_features.view(batch_size * num_edge_types, seq_len, d_model)

        attended, _ = self.self_attention(tokens_grouped, tokens_grouped, tokens_grouped)
        tokens_grouped = self.norm1(tokens_grouped + self.dropout(attended))

        market_context_repeated = market_context.repeat_interleave(num_edge_types, dim=0)
        market_context_repeated = market_context_repeated.unsqueeze(1).expand(-1, seq_len, -1)
        cross_attended, _ = self.cross_attention(tokens_grouped, market_context_repeated, market_context_repeated)
        tokens_grouped = self.norm2(tokens_grouped + self.dropout(cross_attended))

        ff_output = self.feed_forward(tokens_grouped)
        tokens_grouped = self.norm3(tokens_grouped + self.dropout(ff_output))

        return tokens_grouped.view(batch_size, num_edge_types, seq_len, d_model)


class CrossInteractionLayer(nn.Module):
    """Compute interactions between two edge types"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        self.interaction_net = nn.Sequential(
            nn.Linear(2, d_model // 4),  # Two edge values as input
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        self.context_modulation = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        edges1: torch.Tensor, 
        edges2: torch.Tensor, 
        market_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            edges1, edges2: [batch, num_nodes, num_nodes]
            market_context: [batch, d_model]
        """
        batch_size, num_nodes, _ = edges1.shape
        
        # Stack edge values
        edge_pairs = torch.stack([edges1, edges2], dim=-1)  # [batch, N, N, 2]
        
        # Compute interactions
        interactions = self.interaction_net(edge_pairs)  # [batch, N, N, d_model]
        
        # Modulate with market context
        context_modulation = self.context_modulation(market_context)  # [batch, d_model]
        context_expanded = context_modulation.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, d_model]
        
        modulated_interactions = interactions * context_expanded
        
        return modulated_interactions


class MarketRegimeAdapter(nn.Module):
    """Adapt fusion based on market regime"""
    
    def __init__(self, d_model: int, num_regimes: int):
        super().__init__()
        self.d_model = d_model
        self.num_regimes = num_regimes
        
        # Regime-specific adaptation networks
        self.regime_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            ) for _ in range(num_regimes)
        ])
        
        # Regime classification from features
        self.regime_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_regimes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features: torch.Tensor, regime: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [batch, num_nodes, num_nodes, d_model]
            regime: [batch] Optional regime indicators
        """
        batch_size = features.size(0)
        
        if regime is None:
            # Infer regime from features
            feature_summary = features.mean(dim=(1, 2))  # [batch, d_model]
            regime_probs = self.regime_classifier(feature_summary)  # [batch, num_regimes]
            regime = torch.argmax(regime_probs, dim=-1)  # [batch]
        
        # Apply regime-specific adaptation
        adapted_features = torch.zeros_like(features)
        
        for i in range(self.num_regimes):
            regime_mask = (regime == i)
            if regime_mask.any():
                regime_features = features[regime_mask]
                adapted = self.regime_adapters[i](regime_features)
                adapted_features[regime_mask] = adapted
        
        return adapted_features


class EdgeStrengthCalibrator(nn.Module):
    """Calibrate edge strengths based on market conditions"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        self.calibration_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Features + context
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()  # Calibration factors
        )
    
    def forward(self, features: torch.Tensor, market_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, num_nodes, num_nodes, d_model]
            market_context: [batch, d_model]
        """
        batch_size, num_nodes, _, d_model = features.shape
        
        # Expand market context
        context_expanded = market_context.unsqueeze(1).unsqueeze(1).expand(-1, num_nodes, num_nodes, -1)
        
        # Combine features with context
        combined = torch.cat([features, context_expanded], dim=-1)  # [batch, N, N, 2*d_model]
        
        # Compute calibration factors
        calibration_factors = self.calibration_net(combined)  # [batch, N, N, d_model]
        
        # Apply calibration
        calibrated_features = features * calibration_factors
        
        return calibrated_features