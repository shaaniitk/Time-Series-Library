"""
CovariateProcessor - Handles hierarchical processing of covariate families through Mamba blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from utils.logger import logger

from layers.MambaBlock import CovariateMambaBlock
from layers.modular.fusion.hierarchical_fusion import HierarchicalFusion


class CovariateProcessor(nn.Module):
    """
    Processes covariates through: Family Grouping → Mamba Blocks → Hierarchical Attention
    Handles 40 covariates grouped into 10 families of 4, each processed by separate Mamba blocks.
    """
    
    def __init__(
        self,
        num_covariates: int,
        family_size: int,
        seq_len: int,
        d_model: int,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        hierarchical_attention_heads: int = 8,
        dropout: float = 0.1,
        use_family_attention: bool = True,
        fusion_strategy: str = 'weighted_concat',
        covariate_families: Optional[List[int]] = None
    ):
        super(CovariateProcessor, self).__init__()
        
        self.num_covariates = num_covariates
        self.seq_len = seq_len
        self.d_model = d_model
        self.fusion_strategy = fusion_strategy
        
        # Handle custom family configurations
        if covariate_families is not None:
            self.covariate_families = covariate_families
            self.num_families = len(covariate_families)
            assert sum(covariate_families) == num_covariates, \
                f"Sum of family sizes ({sum(covariate_families)}) must equal num_covariates ({num_covariates})"
            logger.info(f"CovariateProcessor: Custom families {covariate_families}")
        else:
            # Default uniform families
            self.family_size = family_size
            self.num_families = num_covariates // family_size
            self.covariate_families = [family_size] * self.num_families
            assert num_covariates % family_size == 0, \
                f"num_covariates ({num_covariates}) must be divisible by family_size ({family_size})"
            logger.info(f"CovariateProcessor: {self.num_families} uniform families of {family_size} covariates")
        
        # Log detailed configuration
        family_dump = {
            'total_covariates': num_covariates,
            'num_families': self.num_families,
            'family_sizes': self.covariate_families,
            'family_configuration': 'custom' if covariate_families is not None else 'uniform'
        }
        logger.info(f"CovariateProcessor Family Configuration Dump: {family_dump}")
        
        # Mamba block for each covariate family (with variable sizes)
        self.family_mamba_blocks = nn.ModuleList([
            CovariateMambaBlock(
                input_dim=self.covariate_families[i],
                d_model=d_model,
                family_size=self.covariate_families[i],
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=dropout,
                use_family_attention=use_family_attention
            ) for i in range(self.num_families)
        ])
        
        # Hierarchical attention for combining family outputs
        try:
            from layers.modular.fusion.hierarchical_fusion import HierarchicalFusion
            self.hierarchical_fusion = HierarchicalFusion(
                d_model=d_model,
                n_levels=self.num_families,
                fusion_strategy=fusion_strategy,
                n_heads=hierarchical_attention_heads
            )
            logger.info(f"Using modular HierarchicalFusion with strategy: {fusion_strategy}")
        except ImportError:
            logger.warning("Modular HierarchicalFusion not found, using basic implementation")
            self.hierarchical_fusion = self._create_basic_hierarchical_fusion(
                d_model, self.num_families, hierarchical_attention_heads, dropout
            )
        
        # Context aggregation and projection
        self.context_aggregator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Final context projection
        self.context_projection = nn.Linear(d_model, d_model)
        
        logger.info(f"CovariateProcessor initialized: {self.num_families} families of {family_size} covariates each")
    
    def _create_basic_hierarchical_fusion(self, d_model: int, num_families: int, 
                                        attention_heads: int, dropout: float):
        """Create basic hierarchical fusion if modular version not available."""
        class BasicHierarchicalFusion(nn.Module):
            def __init__(self, d_model, num_families, attention_heads, dropout):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=attention_heads,
                    dropout=dropout,
                    batch_first=True
                )
                self.norm = nn.LayerNorm(d_model)
                
            def forward(self, family_outputs, target_length):
                # Stack family outputs: [B, num_families, D]
                if isinstance(family_outputs, list):
                    # Average pool each family output across sequence dimension
                    pooled_outputs = []
                    for output in family_outputs:
                        if output.dim() == 3:  # [B, L, D]
                            pooled = output.mean(dim=1)  # [B, D]
                        else:
                            pooled = output
                        pooled_outputs.append(pooled)
                    
                    stacked = torch.stack(pooled_outputs, dim=1)  # [B, num_families, D]
                else:
                    stacked = family_outputs
                
                # Apply self-attention across families
                attended, _ = self.attention(stacked, stacked, stacked)
                attended = self.norm(attended + stacked)
                
                # Global pooling to get single context vector
                context = attended.mean(dim=1)  # [B, D]
                return context
        
        return BasicHierarchicalFusion(d_model, num_families, attention_heads, dropout)
    
    def _split_covariates_into_families(self, covariates: torch.Tensor) -> List[torch.Tensor]:
        """
        Split covariates tensor into family groups.
        
        Args:
            covariates: [batch_size, seq_len, num_covariates]
            
        Returns:
            List of family tensors, each [batch_size, seq_len, family_size]
        """
        batch_size, seq_len, num_features = covariates.shape
        
        if num_features != self.num_covariates:
            logger.warning(f"Expected {self.num_covariates} covariates, got {num_features}")
            # Pad or truncate as needed
            if num_features < self.num_covariates:
                padding = torch.zeros(batch_size, seq_len, 
                                    self.num_covariates - num_features,
                                    device=covariates.device, dtype=covariates.dtype)
                covariates = torch.cat([covariates, padding], dim=-1)
            else:
                covariates = covariates[:, :, :self.num_covariates]
        
        families = []
        start_idx = 0
        for i in range(self.num_families):
            family_size = self.covariate_families[i]
            end_idx = start_idx + family_size
            family = covariates[:, :, start_idx:end_idx]
            families.append(family)
            start_idx = end_idx
            
            # Log family split details
            logger.debug(f"Family {i}: indices {start_idx-family_size}:{end_idx}, size {family_size}, shape {family.shape}")
        
        return families
    
    def forward(
        self, 
        covariates: torch.Tensor, 
        covariate_mask: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process covariates through family Mamba blocks and hierarchical attention.
        
        Args:
            covariates: Historical covariates [batch_size, seq_len, num_covariates]
            covariate_mask: Optional mask for covariates
            future_covariates: Optional future covariates (Hilbert-transformed)
            
        Returns:
            context_vector: Single context vector [batch_size, d_model]
        """
        batch_size, seq_len, num_features = covariates.shape
        
        logger.debug(f"CovariateProcessor forward: input shape {covariates.shape}")
        
        # Store original sequence length for later use
        original_seq_len = seq_len
        
        # Handle future covariates if provided
        if future_covariates is not None:
            logger.debug(f"Using future covariates: {future_covariates.shape}")
            # Concatenate historical and future covariates along sequence dimension
            covariates = torch.cat([covariates, future_covariates], dim=1)
            seq_len = covariates.size(1)  # Update sequence length
            logger.debug(f"Combined covariates shape: {covariates.shape}")
        
        # Step 1: Split covariates into families
        try:
            covariate_families = self._split_covariates_into_families(covariates)
            logger.debug(f"Split into {len(covariate_families)} families")
        except Exception as e:
            logger.error(f"Failed to split covariates into families: {e}")
            raise
        
        # Step 2: Process each family through its Mamba block
        family_outputs = []
        for i, family in enumerate(covariate_families):
            try:
                family_output = self.family_mamba_blocks[i](family, covariate_mask)
                family_outputs.append(family_output)
                logger.debug(f"Family {i} Mamba output shape: {family_output.shape}")
            except Exception as e:
                logger.error(f"Family {i} Mamba processing failed: {e}")
                # Create zero output as fallback
                zero_output = torch.zeros(batch_size, family.size(1), self.d_model,
                                        device=family.device, dtype=family.dtype)
                family_outputs.append(zero_output)
        
        # Step 3: Apply hierarchical attention/fusion
        try:
            if hasattr(self.hierarchical_fusion, 'forward'):
                # Use modular hierarchical fusion
                fused_context = self.hierarchical_fusion(family_outputs, target_length=original_seq_len)
            else:
                # Use basic fusion
                fused_context = self.hierarchical_fusion(family_outputs, original_seq_len)
            
            logger.debug(f"Hierarchical fusion output shape: {fused_context.shape}")
            
        except Exception as e:
            logger.error(f"Hierarchical fusion failed: {e}")
            # Fallback: simple mean pooling
            pooled_families = []
            for output in family_outputs:
                if output.dim() == 3:
                    pooled = output.mean(dim=1)  # [B, D]
                else:
                    pooled = output
                pooled_families.append(pooled)
            
            fused_context = torch.stack(pooled_families, dim=1).mean(dim=1)  # [B, D]
        
        # Step 4: Context aggregation and projection
        try:
            # Apply context aggregation
            if fused_context.dim() == 3:
                # If still has sequence dimension, pool it
                fused_context = fused_context.mean(dim=1)
            
            aggregated_context = self.context_aggregator(fused_context)
            final_context = self.context_projection(aggregated_context)
            
            logger.debug(f"Final covariate context vector shape: {final_context.shape}")
            
        except Exception as e:
            logger.error(f"Context aggregation failed: {e}")
            # Fallback: simple projection
            if fused_context.dim() == 3:
                fused_context = fused_context.mean(dim=1)
            final_context = self.context_projection(fused_context)
        
        return final_context
    
    def get_family_outputs(self, covariates: torch.Tensor, 
                          covariate_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Get individual family outputs for analysis/visualization.
        
        Args:
            covariates: Input covariates tensor
            covariate_mask: Optional mask
            
        Returns:
            List of family output tensors
        """
        covariate_families = self._split_covariates_into_families(covariates)
        family_outputs = []
        
        for i, family in enumerate(covariate_families):
            family_output = self.family_mamba_blocks[i](family, covariate_mask)
            family_outputs.append(family_output)
        
        return family_outputs