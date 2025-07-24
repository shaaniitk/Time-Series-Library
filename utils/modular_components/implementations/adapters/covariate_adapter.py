"""
CovariateAdapter: Universal adapter that adds covariate support to any backbone.
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .base import BaseAdapter

logger = logging.getLogger(__name__)

class CovariateAdapter(BaseAdapter):
    """
    Universal adapter that adds covariate support to any backbone.
    Works by injecting covariates at the data preparation level,
    then passing enhanced data to the original backbone unchanged.
    This preserves the backbone's pretrained knowledge while enabling
    covariate-aware forecasting.
    """
    def __init__(self, backbone: BaseAdapter, covariate_config: Dict[str, Any]):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        super().__init__(MockConfig())
        self.backbone = backbone
        self.covariate_config = covariate_config
        self.covariate_dim = covariate_config.get('covariate_dim', 0)
        self.fusion_method = covariate_config.get('fusion_method', 'project')
        self.embedding_dim = covariate_config.get('embedding_dim', self.backbone.get_d_model())
        self.temporal_features = covariate_config.get('temporal_features', True)
        self._build_covariate_layers()
        logger.info(f"CovariateAdapter initialized:")
        logger.info(f"  - Backbone: {type(self.backbone).__name__}")
        logger.info(f"  - Covariate dim: {self.covariate_dim}")
        logger.info(f"  - Fusion method: {self.fusion_method}")
        logger.info(f"  - Embedding dim: {self.embedding_dim}")

    def _build_covariate_layers(self):
        if self.covariate_dim == 0:
            self.covariate_processor = nn.Identity()
            return
        if self.fusion_method == 'project':
            self.ts_projector = nn.Linear(1, self.embedding_dim // 2)
            self.covariate_projector = nn.Linear(self.covariate_dim, self.embedding_dim // 2)
            self.fusion_projector = nn.Linear(self.embedding_dim, 1)
        elif self.fusion_method == 'concat':
            self.dimension_adapter = nn.Linear(1 + self.covariate_dim, 1)
        elif self.fusion_method == 'add':
            self.covariate_projector = nn.Linear(self.covariate_dim, 1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        self.layer_norm = nn.LayerNorm(1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_ts: torch.Tensor, x_covariates: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        enhanced_ts = self._fuse_covariates(x_ts, x_covariates)
        backbone_input = self._prepare_backbone_input(enhanced_ts)
        if hasattr(self.backbone, 'tokenizer') and self.backbone.tokenizer is not None:
            return self._forward_with_tokenization(backbone_input, attention_mask, **kwargs)
        else:
            return self.backbone.forward(backbone_input, attention_mask=attention_mask, **kwargs)

    def _fuse_covariates(self, x_ts: torch.Tensor, x_covariates: Optional[torch.Tensor]) -> torch.Tensor:
        if x_covariates is None or self.covariate_dim == 0:
            return x_ts
        batch_size, seq_len, ts_dim = x_ts.shape
        if self.fusion_method == 'project':
            ts_emb = self.ts_projector(x_ts)
            cov_emb = self.covariate_projector(x_covariates)
            combined_emb = torch.cat([ts_emb, cov_emb], dim=-1)
            enhanced_ts = self.fusion_projector(combined_emb)
        elif self.fusion_method == 'concat':
            combined = torch.cat([x_ts, x_covariates], dim=-1)
            enhanced_ts = self.dimension_adapter(combined)
        elif self.fusion_method == 'add':
            cov_projected = self.covariate_projector(x_covariates)
            enhanced_ts = x_ts + cov_projected
        enhanced_ts = self.layer_norm(enhanced_ts)
        enhanced_ts = self.dropout(enhanced_ts)
        return enhanced_ts

    def _prepare_backbone_input(self, enhanced_ts: torch.Tensor) -> torch.Tensor:
        return enhanced_ts

    def _forward_with_tokenization(self, enhanced_ts: torch.Tensor, attention_mask: Optional[torch.Tensor], **kwargs) -> torch.Tensor:
        try:
            tokenizer = self.backbone.tokenizer
            enhanced_values = enhanced_ts.squeeze(-1)
            if hasattr(tokenizer, 'encode'):
                input_ids = []
                for batch_idx in range(enhanced_values.size(0)):
                    series = enhanced_values[batch_idx].detach().cpu().numpy()
                    try:
                        if hasattr(series, '__iter__') and not isinstance(series, (str, bytes)):
                            tokens = tokenizer.encode(series)
                        else:
                            tokens = tokenizer.encode([float(series)])
                        if isinstance(tokens, list):
                            token_tensor = torch.tensor(tokens, dtype=torch.long)
                        elif isinstance(tokens, torch.Tensor):
                            token_tensor = tokens.long()
                        else:
                            token_tensor = torch.tensor([tokens], dtype=torch.long)
                    except Exception as token_e:
                        logger.warning(f"Tokenization failed for batch {batch_idx}: {token_e}")
                        token_tensor = torch.tensor(
                            [int(x * 100) % 1000 for x in series.flatten()],
                            dtype=torch.long
                        )
                    input_ids.append(token_tensor)
                max_len = max(t.size(0) for t in input_ids)
                padded_input_ids = []
                for t in input_ids:
                    if t.size(0) < max_len:
                        padding = torch.zeros(max_len - t.size(0), dtype=torch.long)
                        t = torch.cat([t, padding])
                    elif t.size(0) > max_len:
                        t = t[:max_len]
                    padded_input_ids.append(t)
                input_ids = torch.stack(padded_input_ids).to(enhanced_values.device)
            else:
                input_ids = enhanced_values.long()
            return self.backbone.forward(input_ids, attention_mask=attention_mask, **kwargs)
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}, falling back to direct forward")
            return self.backbone.forward(enhanced_ts, attention_mask=attention_mask, **kwargs)

    def get_d_model(self) -> int:
        return self.backbone.get_d_model()

    def supports_seq2seq(self) -> bool:
        return self.backbone.supports_seq2seq()

    def get_backbone_type(self) -> str:
        return f"covariate_adapted_{self.backbone.get_backbone_type()}"

    def get_capabilities(self) -> Dict[str, Any]:
        base_capabilities = self.backbone.get_capabilities()
        base_capabilities.update({
            'supports_covariates': True,
            'covariate_dim': self.covariate_dim,
            'fusion_method': self.fusion_method,
            'adapter_type': 'covariate',
            'base_backbone': self.backbone.get_backbone_type()
        })
        return base_capabilities
