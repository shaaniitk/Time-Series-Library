import json
import torch

from models.Celestial_Enhanced_PGAT import Model


class SimpleConfig:
    def __init__(self):
        # Sequence config
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        # Epochs
        self.num_epochs = 20
        # Inputs/outputs
        self.enc_in = 118  # aligns with aggregator indices
        self.num_input_waves = 118
        self.dec_in = 4
        self.c_out = 4
        # Model sizes
        self.d_model = 32
        self.n_heads = 4
        self.e_layers = 1
        self.d_layers = 1
        self.dropout = 0.0
        # Embedding config
        self.embed = 'timeF'
        self.freq = 'h'
        # Advanced features OFF except aggregation
        self.use_mixture_decoder = False
        self.use_stochastic_learner = False
        self.use_hierarchical_mapping = False
        self.aggregate_waves_to_celestial = True


def tensor_shape(x):
    try:
        return list(x.shape)
    except Exception:
        return None


def collect_metadata_shapes(meta):
    shapes = {}
    if isinstance(meta, dict):
        for k, v in meta.items():
            if isinstance(v, torch.Tensor):
                shapes[k] = tensor_shape(v)
            elif isinstance(v, dict):
                shapes[k] = {sk: tensor_shape(sv) for sk, sv in v.items() if isinstance(sv, torch.Tensor)}
    return shapes


def _full_tensor_value(t: torch.Tensor):
    try:
        x = t.detach().cpu()
        # For 3D tensors, dump the full 2D matrix of batch 0 to keep JSON reasonable
        if x.dim() == 3:
            return x[0].tolist()
        if x.dim() in (1, 2):
            return x.tolist()
        return None
    except Exception:
        return None


def collect_adj_and_attn(meta):
    """Traverse metadata to extract adjacency matrices and attention masks/scores.
    Captures shapes and a small sample for traceability without huge JSON size.
    """
    result = {
        'adjacency': [],
        'attention': [],
    }

    def walk(d, prefix=""):
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else k
            try:
                if isinstance(v, torch.Tensor):
                    key_lower = k.lower()
                    shape = tensor_shape(v)
                    value = _full_tensor_value(v)
                    if ('adj' in key_lower) or ('adjacency' in key_lower):
                        result['adjacency'].append({'key': path, 'shape': shape, 'value': value})
                    if ('attn' in key_lower) or ('attention' in key_lower) or ('mask' in key_lower):
                        result['attention'].append({'key': path, 'shape': shape, 'value': value})
                elif isinstance(v, dict):
                    walk(v, path)
            except Exception:
                # Skip problematic entries, continue traversing
                continue

    walk(meta)
    return result


def main():
    torch.manual_seed(11)
    cfg = SimpleConfig()
    model = Model(cfg)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    B = 8
    # Encoder inputs
    x_enc = torch.randn(B, cfg.seq_len, cfg.enc_in)
    # Valid temporal markers
    month = torch.randint(0, 13, (B, cfg.seq_len))
    day = torch.randint(0, 32, (B, cfg.seq_len))
    weekday = torch.randint(0, 8, (B, cfg.seq_len))
    hour = torch.randint(0, 25, (B, cfg.seq_len))
    x_mark_enc = torch.stack([month, day, weekday, hour], dim=-1)

    # Decoder inputs and marks
    x_dec = torch.randn(B, cfg.label_len + cfg.pred_len, cfg.dec_in)
    month_d = torch.randint(0, 13, (B, cfg.label_len + cfg.pred_len))
    day_d = torch.randint(0, 32, (B, cfg.label_len + cfg.pred_len))
    weekday_d = torch.randint(0, 8, (B, cfg.label_len + cfg.pred_len))
    hour_d = torch.randint(0, 25, (B, cfg.label_len + cfg.pred_len))
    x_mark_dec = torch.stack([month_d, day_d, weekday_d, hour_d], dim=-1)

    # Identity-like targets over prediction horizon
    targets = x_dec[:, -cfg.pred_len :, : cfg.c_out].detach().clone()

    losses = []
    grad_norms = []
    meta_shapes = None
    adj_attn_dump = None
    epoch_traces = []

    for epoch in range(cfg.num_epochs):
        optimizer.zero_grad(set_to_none=True)
        preds, metadata = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # Record metadata shapes and adjacency/attention once
        if meta_shapes is None:
            meta_shapes = collect_metadata_shapes(metadata)
        if adj_attn_dump is None:
            adj_attn_dump = collect_adj_and_attn(metadata)
        # Per-epoch full dump of adjacency/attention (batch-0 for 3D tensors)
        epoch_dump = collect_adj_and_attn(metadata)
        preds = preds[:, -cfg.pred_len :, :]
        loss = criterion(preds, targets)
        loss.backward()
        # Gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item()
        grad_norms.append(total_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(loss.item())
        epoch_traces.append({
            'epoch': epoch,
            'loss': losses[-1],
            'grad_norm': grad_norms[-1],
            'adjacency': epoch_dump['adjacency'],
            'attention': epoch_dump['attention'],
        })

    # Sample prediction from final epoch (first batch)
    with torch.no_grad():
        final_preds, _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        final_preds = final_preds[:, -cfg.pred_len :, :]
        pred_sample = final_preds[0].cpu().tolist()

    dump = {
        'config': {
            'seq_len': cfg.seq_len,
            'label_len': cfg.label_len,
            'pred_len': cfg.pred_len,
            'enc_in': cfg.enc_in,
            'num_input_waves': cfg.num_input_waves,
            'dec_in': cfg.dec_in,
            'c_out': cfg.c_out,
            'd_model': cfg.d_model,
            'n_heads': cfg.n_heads,
            'e_layers': cfg.e_layers,
            'd_layers': cfg.d_layers,
            'dropout': cfg.dropout,
            'embed': cfg.embed,
            'freq': cfg.freq,
            'use_mixture_decoder': cfg.use_mixture_decoder,
            'use_stochastic_learner': cfg.use_stochastic_learner,
            'use_hierarchical_mapping': cfg.use_hierarchical_mapping,
            'aggregate_waves_to_celestial': cfg.aggregate_waves_to_celestial,
        },
        'losses': losses,
        'grad_norms': grad_norms,
        'metadata_shapes': meta_shapes,
        'adjacency_and_attention': adj_attn_dump,
        'epoch_traces': epoch_traces,
        'pred_sample': pred_sample,
    }

    out_path = 'reports/pgat_training_dump.json'
    with open(out_path, 'w') as f:
        json.dump(dump, f)
    print(f"Dumped training info to {out_path}")


if __name__ == '__main__':
    main()