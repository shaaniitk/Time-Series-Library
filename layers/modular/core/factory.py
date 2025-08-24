"""Unified factory leveraging the unified registry and config schemas."""
from __future__ import annotations
from typing import Any, Dict
from .registry import unified_registry, ComponentFamily
from configs.schemas import ComponentType, ModularAutoformerConfig
from layers.modular.core.logger import logger

COMPONENT_FAMILY_MAP: Dict[ComponentType, ComponentFamily] = {
    # Attention
    ComponentType.AUTOCORRELATION: ComponentFamily.ATTENTION,
    ComponentType.ADAPTIVE_AUTOCORRELATION: ComponentFamily.ATTENTION,
    ComponentType.CROSS_RESOLUTION: ComponentFamily.ATTENTION,
    ComponentType.FOURIER_ATTENTION: ComponentFamily.ATTENTION,
    ComponentType.WAVELET_ATTENTION: ComponentFamily.ATTENTION,
    # Decomposition
    ComponentType.MOVING_AVG: ComponentFamily.DECOMPOSITION,
    ComponentType.LEARNABLE_DECOMP: ComponentFamily.DECOMPOSITION,
    ComponentType.WAVELET_DECOMP: ComponentFamily.DECOMPOSITION,
    # Encoder / Decoder
    ComponentType.STANDARD_ENCODER: ComponentFamily.ENCODER,
    ComponentType.ENHANCED_ENCODER: ComponentFamily.ENCODER,
    ComponentType.HIERARCHICAL_ENCODER: ComponentFamily.ENCODER,
    ComponentType.STANDARD_DECODER: ComponentFamily.DECODER,
    ComponentType.ENHANCED_DECODER: ComponentFamily.DECODER,
    # Sampling
    ComponentType.DETERMINISTIC: ComponentFamily.SAMPLING,
    ComponentType.BAYESIAN: ComponentFamily.SAMPLING,
    # Output Head
    ComponentType.STANDARD_HEAD: ComponentFamily.OUTPUT_HEAD,
    ComponentType.QUANTILE: ComponentFamily.OUTPUT_HEAD,
    # Loss
    ComponentType.MSE: ComponentFamily.LOSS,
    ComponentType.MAE: ComponentFamily.LOSS,
    ComponentType.QUANTILE_LOSS: ComponentFamily.LOSS,
    ComponentType.BAYESIAN_MSE: ComponentFamily.LOSS,
    ComponentType.BAYESIAN_QUANTILE: ComponentFamily.LOSS,
    # Normalization
    ComponentType.LAYER_NORM: ComponentFamily.NORMALIZATION,
    ComponentType.RMS_NORM: ComponentFamily.NORMALIZATION,
    # Embedding
    ComponentType.POSITIONAL_EMBEDDING: ComponentFamily.EMBEDDING,
    ComponentType.TOKEN_EMBEDDING: ComponentFamily.EMBEDDING,
    ComponentType.FIXED_EMBEDDING: ComponentFamily.EMBEDDING,
    ComponentType.TEMPORAL_EMBEDDING: ComponentFamily.EMBEDDING,
    ComponentType.TIME_FEATURE_EMBEDDING: ComponentFamily.EMBEDDING,
    ComponentType.DATA_EMBEDDING: ComponentFamily.EMBEDDING,
    ComponentType.DATA_EMBEDDING_INVERTED: ComponentFamily.EMBEDDING,
    ComponentType.DATA_EMBEDDING_WO_POS: ComponentFamily.EMBEDDING,
    ComponentType.PATCH_EMBEDDING: ComponentFamily.EMBEDDING,
}

class UnifiedFactory:
    def create_from_config(self, config: ModularAutoformerConfig) -> Dict[str, Any]:
        # Build components using mapping
        attn_cfg = config.attention
        decomp_cfg = config.decomposition
        enc_cfg = config.encoder
        dec_cfg = config.decoder

        attention = self._create(
            attn_cfg.type,
            d_model=config.d_model,
            n_heads=getattr(attn_cfg, 'n_heads', 8),
            factor=getattr(attn_cfg, 'factor', 1),
            dropout=attn_cfg.dropout,
            output_attention=getattr(attn_cfg, 'output_attention', False),
        )

        # Map schema fields to component constructor names
        decomp_kwargs = {
            "input_dim": getattr(decomp_cfg, 'input_dim', config.d_model),
            "d_model": config.d_model,
        }
        if decomp_cfg.type == ComponentType.LEARNABLE_DECOMP:
            decomp_kwargs["init_kernel_size"] = getattr(decomp_cfg, 'kernel_size', 25)
        else:
            decomp_kwargs["kernel_size"] = getattr(decomp_cfg, 'kernel_size', 25)

        decomposition = self._create(decomp_cfg.type, **decomp_kwargs)

        encoder = self._create(
            enc_cfg.type,
            num_encoder_layers=getattr(enc_cfg, 'num_encoder_layers', getattr(enc_cfg, 'e_layers', 2)),
            d_model=config.d_model,
            n_heads=getattr(enc_cfg, 'n_heads', 8),
            d_ff=enc_cfg.d_ff,
            dropout=enc_cfg.dropout,
            activation=enc_cfg.activation,
            attention_comp=attention,
            decomp_comp=decomposition,
        )

        decoder = self._create(
            dec_cfg.type,
            num_decoder_layers=getattr(dec_cfg, 'num_decoder_layers', getattr(dec_cfg, 'd_layers', 1)),
            d_model=config.d_model,
            n_heads=getattr(dec_cfg, 'n_heads', 8),
            d_ff=dec_cfg.d_ff,
            dropout=dec_cfg.dropout,
            activation=dec_cfg.activation,
            c_out=dec_cfg.c_out,
            self_attention_comp=attention,
            cross_attention_comp=attention,
            decomp_comp=decomposition,
        )

        sampling = self._create(
            config.sampling.type,
            n_samples=getattr(config.sampling, 'n_samples', 1),
            quantile_levels=(config.quantile_levels or getattr(config.sampling, 'quantile_levels', None)),
        )

        output_head = self._create(
            config.output_head.type,
            d_model=config.output_head.d_model,
            c_out=config.output_head.c_out,
            num_quantiles=getattr(config.output_head, 'num_quantiles', None),
        )

        loss = self._create(config.loss.type, quantiles=getattr(config.loss, 'quantiles', None))
        embedding = None
        if config.embedding:
            embedding = self._create(config.embedding.type,
                                    c_in=config.embedding.c_in,
                                    d_model=config.embedding.d_model,
                                    max_len=getattr(config.embedding, 'max_len', 5000),
                                    dropout=getattr(config.embedding, 'dropout', 0.1),
                                    time_features=getattr(config.embedding, 'time_features', True),
                                    freq=getattr(config.embedding, 'freq', 'h'))
        normalization = None
        if config.normalization:
            normalization = self._create(config.normalization.type,
                                        normalized_shape=config.normalization.normalized_shape,
                                        eps=getattr(config.normalization, 'eps', 1e-5),
                                        affine=getattr(config.normalization, 'affine', True))
        return {
            'embedding': embedding,
            'normalization': normalization,
            'attention': attention,
            'decomposition': decomposition,
            'encoder': encoder,
            'decoder': decoder,
            'sampling': sampling,
            'output_head': output_head,
            'loss': loss,
            'embedding': embedding,
            'normalization': normalization
        }

    def _create(self, comp_type: ComponentType, **kwargs) -> Any:
        family = COMPONENT_FAMILY_MAP.get(comp_type)
        if not family:
            raise ValueError(f"Unsupported component type: {comp_type}")
        name = comp_type.value  # rely on enum value mapping to registration key
        instance = unified_registry.create(family, name, **kwargs)
        logger.info(f"Created {family.value}:{name}")
        return instance

unified_factory = UnifiedFactory()
