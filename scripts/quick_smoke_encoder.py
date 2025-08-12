"""
Quick smoke test for wrapped encoders registered as processors via unified registry.
"""
import torch
from unified_component_registry import unified_registry
from utils.implementations.encoder.wrapped_encoders import EncoderProcessorConfig


def main() -> None:
    # Ensure registry has encoders (auto-registered by unified registry)
    d_model = 64
    seq_len = 32
    batch = 2

    x = torch.randn(batch, seq_len, d_model)

    base_cfg = EncoderProcessorConfig(d_model=d_model, seq_len=seq_len, pred_len=seq_len)

    # Provide minimal components required by encoder layers
    from utils_algorithm_adapters import RestoredFourierConfig, RestoredFourierAttention
    from utils.implementations.decomposition.wrapped_decompositions import DecompositionProcessorConfig, SeriesDecompositionProcessor

    attn = RestoredFourierAttention(RestoredFourierConfig(d_model=d_model, num_heads=4, dropout=0.1))
    decomp = SeriesDecompositionProcessor(DecompositionProcessorConfig(d_model=d_model, seq_len=seq_len))

    # Inject components via custom_params
    base_cfg.custom_params["attention_component"] = attn
    base_cfg.custom_params["decomposition_component"] = decomp

    for name in [
        "encoder_standard_processor",
        "encoder_enhanced_processor",
    ]:
        enc_cls = unified_registry.get_component("processor", name)
        enc = enc_cls(base_cfg)
        out = enc.process_sequence(x, None, target_length=seq_len)
        assert out.shape == x.shape, f"{name} output shape {out.shape} != input {x.shape}"
        print(f"{name}: OK with shape {out.shape}")

    print("ENCODER SMOKE CHECK: SUCCESS")


if __name__ == "__main__":
    main()
