"""
Quick smoke test for wrapped decoders registered as processors via unified registry.
"""
import torch
from unified_component_registry import unified_registry
from utils.implementations.decoder.wrapped_decoders import DecoderProcessorConfig
from utils.implementations.encoder.wrapped_encoders import EncoderProcessorConfig


def main() -> None:
    d_model = 64
    seq_len = 32
    pred_len = 16
    batch = 2

    # encoder output (memory) same dim
    enc_out = torch.randn(batch, seq_len, d_model)
    # decoder input (seasonal init)
    dec_in = torch.randn(batch, pred_len, d_model)

    # Build attentions and decomposition components
    from utils_algorithm_adapters import RestoredFourierConfig, RestoredFourierAttention
    from utils.implementations.decomposition.wrapped_decompositions import DecompositionProcessorConfig, SeriesDecompositionProcessor

    self_attn = RestoredFourierAttention(RestoredFourierConfig(d_model=d_model, num_heads=4, dropout=0.1))
    cross_attn = RestoredFourierAttention(RestoredFourierConfig(d_model=d_model, num_heads=4, dropout=0.1))
    decomp = SeriesDecompositionProcessor(DecompositionProcessorConfig(d_model=d_model, seq_len=seq_len))

    base_cfg = DecoderProcessorConfig(d_model=d_model, seq_len=seq_len, pred_len=pred_len, c_out=d_model)
    base_cfg.custom_params.update({
        'self_attention_component': self_attn,
        'cross_attention_component': cross_attn,
        'decomposition_component': decomp,
        'd_layers': 2,
        'd_ff': 4 * d_model,
        'activation': 'relu',
        'c_out': d_model,
    })

    for name in [
        'decoder_standard_processor',
        'decoder_enhanced_processor',
    ]:
        dec_cls = unified_registry.get_component('processor', name)
        dec = dec_cls(base_cfg)
        out = dec.process_sequence(dec_in, enc_out, target_length=pred_len, trend=torch.zeros_like(dec_in))
        assert out.shape == (batch, pred_len, d_model), f"{name} produced {out.shape}"
        print(f"{name}: OK with shape {out.shape}")

    print("DECODER SMOKE CHECK: SUCCESS")


if __name__ == "__main__":
    main()
