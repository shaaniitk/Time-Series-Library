import torch
import traceback

from models.Celestial_Enhanced_PGAT import Model


class MockConfig:
    def __init__(self):
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        # Trigger aggregation: enc_in equals num_input_waves
        self.enc_in = 64
        self.dec_in = 4
        self.c_out = 4
        self.d_model = 64
        self.n_heads = 4
        self.e_layers = 2
        self.d_layers = 1
        self.dropout = 0.1
        self.embed = 'timeF'
        self.freq = 'h'
        self.use_mixture_decoder = True
        self.use_stochastic_learner = True
        self.use_hierarchical_mapping = True
        self.aggregate_waves_to_celestial = True
        self.num_input_waves = 64


def main():
    B = 2
    cfg = MockConfig()
    model = Model(cfg)
    model.eval()

    x_enc = torch.randn(B, cfg.seq_len, cfg.enc_in)
    # Valid temporal marks: [month, day, weekday, hour]
    month = torch.randint(0, 13, (B, cfg.seq_len))
    day = torch.randint(0, 32, (B, cfg.seq_len))
    weekday = torch.randint(0, 8, (B, cfg.seq_len))
    hour = torch.randint(0, 25, (B, cfg.seq_len))
    x_mark_enc = torch.stack([month, day, weekday, hour], dim=-1)

    x_dec = torch.randn(B, cfg.label_len + cfg.pred_len, cfg.dec_in)
    month_d = torch.randint(0, 13, (B, cfg.label_len + cfg.pred_len))
    day_d = torch.randint(0, 32, (B, cfg.label_len + cfg.pred_len))
    weekday_d = torch.randint(0, 8, (B, cfg.label_len + cfg.pred_len))
    hour_d = torch.randint(0, 25, (B, cfg.label_len + cfg.pred_len))
    x_mark_dec = torch.stack([month_d, day_d, weekday_d, hour_d], dim=-1)

    try:
        outputs, metadata = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print("Forward OK. Output type:", type(outputs))
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                print(f"  {k}: {v.shape}")
        else:
            print("  output:", outputs.shape)
        print("Metadata keys:", list(metadata.keys()))
        print("original_targets:", None if metadata.get('original_targets') is None else metadata['original_targets'].shape)
    except Exception as e:
        print("Forward failed:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()