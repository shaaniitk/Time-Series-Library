import math

import torch
from torch.utils.data import TensorDataset


def _wrap_phase(phase):
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def _phase_trajectory(time_index, period, phase_offset, velocity_scale, drift_strength, accel_strength, modulation_period):
    time_view = time_index.view(1, -1, 1)
    base_velocity = (2.0 * math.pi / period) * velocity_scale
    drift = 1.0 + drift_strength * torch.sin((2.0 * math.pi * time_view / modulation_period) + 0.35 * phase_offset)
    accel = 1.0 + accel_strength * (time_view / max(float(time_index.numel() - 1), 1.0))
    phase_velocity = base_velocity * drift * accel
    phase = phase_offset + torch.cumsum(phase_velocity, dim=1)
    return phase, phase_velocity


def _wave_from_phase(phase, amplitude):
    return amplitude * torch.sin(phase)


def _build_feature_tensor(feature_bank, wanted_channels):
    features = torch.cat(feature_bank, dim=-1)
    if features.shape[-1] >= wanted_channels:
        return features[..., :wanted_channels]

    extra_channels = wanted_channels - features.shape[-1]
    extra_noise = 0.05 * torch.randn(*features.shape[:-1], extra_channels, device=features.device)
    return torch.cat([features, extra_noise], dim=-1)


def make_multiscale_tft_tensors(
    seq_len,
    label_len,
    pred_len,
    enc_in,
    c_out,
    known_len,
    n_samples,
    device=None,
    noise_std=0.01,
    low_period=96.0,
    medium_period=24.0,
    high_period=6.0,
):
    device = device or torch.device("cpu")
    full_time = torch.arange(seq_len + pred_len, device=device, dtype=torch.float32)
    modulation_period = max(float(seq_len + pred_len), 8.0)

    phase_base = 2.0 * math.pi * torch.rand(n_samples, 1, c_out, device=device)
    phase_offset = torch.linspace(0.0, math.pi / 3.0, c_out, device=device).view(1, 1, c_out)
    low_phase_offset = phase_base + phase_offset
    medium_phase_offset = 1.7 * phase_base + 0.5 * phase_offset
    high_phase_offset = 2.3 * phase_base + 0.8 * phase_offset

    low_velocity_scale = 0.92 + 0.10 * torch.rand(n_samples, 1, c_out, device=device)
    medium_velocity_scale = 0.95 + 0.18 * torch.rand(n_samples, 1, c_out, device=device)
    high_velocity_scale = 0.90 + 0.28 * torch.rand(n_samples, 1, c_out, device=device)

    low_amplitude = 1.1 + 0.2 * torch.rand(n_samples, 1, c_out, device=device)
    medium_amplitude = 0.7 + 0.2 * torch.rand(n_samples, 1, c_out, device=device)
    high_amplitude = 0.35 + 0.15 * torch.rand(n_samples, 1, c_out, device=device)

    low_phase, low_velocity = _phase_trajectory(
        full_time,
        low_period,
        low_phase_offset,
        low_velocity_scale,
        drift_strength=0.08,
        accel_strength=0.03,
        modulation_period=modulation_period,
    )
    medium_phase, medium_velocity = _phase_trajectory(
        full_time,
        medium_period,
        medium_phase_offset,
        medium_velocity_scale,
        drift_strength=0.12,
        accel_strength=0.05,
        modulation_period=modulation_period,
    )
    high_phase, high_velocity = _phase_trajectory(
        full_time,
        high_period,
        high_phase_offset,
        high_velocity_scale,
        drift_strength=0.18,
        accel_strength=0.08,
        modulation_period=modulation_period,
    )

    low_full = _wave_from_phase(low_phase, low_amplitude)
    medium_full = _wave_from_phase(medium_phase, medium_amplitude)
    high_full = _wave_from_phase(high_phase, high_amplitude)

    delta_lm = _wrap_phase(low_phase - medium_phase)
    delta_mh = _wrap_phase(medium_phase - high_phase)
    delta_lh = _wrap_phase(low_phase - high_phase)

    ratio_ml = medium_velocity / (low_velocity.abs() + 1e-6)
    ratio_hm = high_velocity / (medium_velocity.abs() + 1e-6)
    ratio_hl = high_velocity / (low_velocity.abs() + 1e-6)

    phase_alignment = torch.cos(delta_lm) * low_full + torch.sin(delta_mh) * medium_full
    phase_difference_drive = torch.sin(delta_lm) * torch.cos(delta_lh)
    velocity_ratio_drive = torch.tanh(ratio_hm - 2.0) + 0.5 * torch.tanh(ratio_ml - 3.5)
    high_burst_gate = torch.sigmoid(4.0 * (torch.cos(delta_lh) + 0.35 * torch.tanh(ratio_hl - 6.0)))
    regime_score = 1.1 * torch.cos(delta_lm) + 0.8 * torch.sin(delta_mh) + 0.5 * torch.tanh(ratio_hm - 2.0)
    regime_low = torch.sigmoid(-3.0 * regime_score)
    regime_mid = torch.sigmoid(3.0 * (regime_score + 0.35)) * torch.sigmoid(3.0 * (0.35 - regime_score))
    regime_high = torch.sigmoid(3.0 * regime_score)
    regime_normalizer = regime_low + regime_mid + regime_high + 1e-6
    regime_low = regime_low / regime_normalizer
    regime_mid = regime_mid / regime_normalizer
    regime_high = regime_high / regime_normalizer
    regime_burst_gate = regime_low * (0.25 + 0.35 * high_burst_gate) + regime_mid * (0.60 + 0.25 * high_burst_gate) + regime_high * (0.95 + 0.45 * high_burst_gate)

    interaction_full = low_full * medium_full
    harmonic_full = medium_full * high_full
    regime_interaction = regime_low * low_full + regime_mid * interaction_full + regime_high * harmonic_full
    target_full = (
        0.42 * low_full
        + 0.16 * medium_full
        + 0.08 * regime_burst_gate * high_full
        + 0.10 * interaction_full
        + 0.06 * harmonic_full
        + 0.08 * phase_alignment
        + 0.06 * phase_difference_drive
        + 0.07 * velocity_ratio_drive * high_full
        + 0.06 * regime_interaction
    )

    shared_phase = 2.0 * math.pi * torch.rand(n_samples, 1, 1, device=device)
    known_low_phase, known_low_velocity = _phase_trajectory(
        full_time,
        low_period,
        shared_phase,
        0.95 + 0.06 * torch.rand(n_samples, 1, 1, device=device),
        drift_strength=0.05,
        accel_strength=0.02,
        modulation_period=modulation_period,
    )
    known_medium_phase, known_medium_velocity = _phase_trajectory(
        full_time,
        medium_period,
        1.4 * shared_phase,
        1.00 + 0.08 * torch.rand(n_samples, 1, 1, device=device),
        drift_strength=0.08,
        accel_strength=0.03,
        modulation_period=modulation_period,
    )
    known_high_phase, known_high_velocity = _phase_trajectory(
        full_time,
        high_period,
        2.1 * shared_phase,
        1.05 + 0.10 * torch.rand(n_samples, 1, 1, device=device),
        drift_strength=0.12,
        accel_strength=0.05,
        modulation_period=modulation_period,
    )
    known_low_full = torch.sin(known_low_phase)
    known_medium_full = torch.sin(known_medium_phase)
    known_high_full = torch.sin(known_high_phase)
    known_interaction_full = known_low_full * known_medium_full
    known_harmonic_full = known_medium_full * known_high_full
    known_delta_lm = _wrap_phase(known_low_phase - known_medium_phase)
    known_delta_mh = _wrap_phase(known_medium_phase - known_high_phase)
    known_ratio_ml = known_medium_velocity / (known_low_velocity.abs() + 1e-6)
    known_ratio_hm = known_high_velocity / (known_medium_velocity.abs() + 1e-6)
    known_phase_alignment = torch.cos(known_delta_lm) + torch.sin(known_delta_mh)
    known_velocity_ratio = torch.tanh(known_ratio_ml - 3.5) + torch.tanh(known_ratio_hm - 2.0)
    known_burst_gate = torch.sigmoid(4.0 * (torch.cos(_wrap_phase(known_low_phase - known_high_phase)) + 0.25 * known_velocity_ratio))
    known_regime_score = 1.1 * torch.cos(known_delta_lm) + 0.8 * torch.sin(known_delta_mh) + 0.5 * torch.tanh(known_ratio_hm - 2.0)
    known_regime_low = torch.sigmoid(-3.0 * known_regime_score)
    known_regime_mid = torch.sigmoid(3.0 * (known_regime_score + 0.35)) * torch.sigmoid(3.0 * (0.35 - known_regime_score))
    known_regime_high = torch.sigmoid(3.0 * known_regime_score)
    known_regime_norm = known_regime_low + known_regime_mid + known_regime_high + 1e-6
    known_regime_low = known_regime_low / known_regime_norm
    known_regime_mid = known_regime_mid / known_regime_norm
    known_regime_high = known_regime_high / known_regime_norm
    known_regime_gate = known_regime_low * 0.35 + known_regime_mid * 0.70 + known_regime_high * 1.10
    known_trend_full = full_time.view(1, -1, 1).expand(n_samples, -1, 1) / max(float(seq_len + pred_len - 1), 1.0)
    known_mix_full = known_low_full + known_medium_full + known_high_full

    projection_weights = torch.linspace(0.25, 0.65, c_out, device=device).view(1, 1, c_out)
    known_projection = (known_mix_full[:, -pred_len:, :] + 0.5 * known_phase_alignment[:, -pred_len:, :]) * projection_weights
    recent_target = target_full[:, seq_len - pred_len:seq_len, :]
    ratio_projection = known_velocity_ratio[:, -pred_len:, :].expand(-1, -1, c_out)
    phase_projection = known_burst_gate[:, -pred_len:, :].expand(-1, -1, c_out)
    y_future = (
        0.52 * target_full[:, -pred_len:, :]
        + 0.18 * recent_target
        + 0.15 * known_projection
        + 0.09 * ratio_projection
        + 0.06 * phase_projection * target_full[:, -pred_len:, :]
        + 0.05 * known_regime_gate[:, -pred_len:, :].expand(-1, -1, c_out)
    )
    y_future = y_future + noise_std * torch.randn_like(y_future)

    history_target = target_full[:, :seq_len, :] + noise_std * torch.randn(n_samples, seq_len, c_out, device=device)
    low_history = low_full[:, :seq_len, :]
    medium_history = medium_full[:, :seq_len, :]
    high_history = high_full[:, :seq_len, :]
    interaction_history = interaction_full[:, :seq_len, :]
    harmonic_history = harmonic_full[:, :seq_len, :]
    phase_alignment_history = phase_alignment[:, :seq_len, :]
    velocity_ratio_history = velocity_ratio_drive[:, :seq_len, :]
    burst_gate_history = high_burst_gate[:, :seq_len, :]
    regime_gate_history = regime_burst_gate[:, :seq_len, :]
    regime_low_history = regime_low[:, :seq_len, :]
    regime_mid_history = regime_mid[:, :seq_len, :]
    regime_high_history = regime_high[:, :seq_len, :]
    residual_history = history_target - (
        0.42 * low_history + 0.16 * medium_history + 0.08 * regime_gate_history * high_history
    )

    x_enc = _build_feature_tensor(
        [
            history_target,
            low_history,
            medium_history,
            high_history,
            interaction_history,
            harmonic_history,
            phase_alignment_history,
            velocity_ratio_history,
            burst_gate_history,
            regime_gate_history,
            regime_low_history,
            regime_mid_history,
            regime_high_history,
            residual_history,
        ],
        enc_in,
    )

    known_history = _build_feature_tensor(
        [
            known_low_full[:, :seq_len, :],
            known_medium_full[:, :seq_len, :],
            known_high_full[:, :seq_len, :],
            known_interaction_full[:, :seq_len, :],
            known_harmonic_full[:, :seq_len, :],
            known_phase_alignment[:, :seq_len, :],
            known_velocity_ratio[:, :seq_len, :],
            known_burst_gate[:, :seq_len, :],
            known_regime_gate[:, :seq_len, :],
            known_regime_low[:, :seq_len, :],
            known_regime_mid[:, :seq_len, :],
            known_regime_high[:, :seq_len, :],
            known_mix_full[:, :seq_len, :],
            known_trend_full[:, :seq_len, :],
        ],
        known_len,
    )
    known_decoder = _build_feature_tensor(
        [
            known_low_full[:, seq_len - label_len:seq_len + pred_len, :],
            known_medium_full[:, seq_len - label_len:seq_len + pred_len, :],
            known_high_full[:, seq_len - label_len:seq_len + pred_len, :],
            known_interaction_full[:, seq_len - label_len:seq_len + pred_len, :],
            known_harmonic_full[:, seq_len - label_len:seq_len + pred_len, :],
            known_phase_alignment[:, seq_len - label_len:seq_len + pred_len, :],
            known_velocity_ratio[:, seq_len - label_len:seq_len + pred_len, :],
            known_burst_gate[:, seq_len - label_len:seq_len + pred_len, :],
            known_regime_gate[:, seq_len - label_len:seq_len + pred_len, :],
            known_regime_low[:, seq_len - label_len:seq_len + pred_len, :],
            known_regime_mid[:, seq_len - label_len:seq_len + pred_len, :],
            known_regime_high[:, seq_len - label_len:seq_len + pred_len, :],
            known_mix_full[:, seq_len - label_len:seq_len + pred_len, :],
            known_trend_full[:, seq_len - label_len:seq_len + pred_len, :],
        ],
        known_len,
    )

    label_context = history_target[:, -label_len:, :c_out]
    x_dec = torch.cat([
        label_context,
        torch.zeros(n_samples, pred_len, c_out, device=device),
    ], dim=1)
    batch_y = torch.cat([label_context, y_future], dim=1)

    return {
        "x_enc": x_enc,
        "x_mark_enc": known_history,
        "x_dec": x_dec,
        "x_mark_dec": known_decoder,
        "y_future": y_future,
        "batch_y": batch_y,
        "wave_components": {
            "low": low_full,
            "medium": medium_full,
            "high": high_full,
            "phase_difference_low_medium": delta_lm,
            "phase_difference_medium_high": delta_mh,
            "phase_velocity_ratio_medium_low": ratio_ml,
            "phase_velocity_ratio_high_medium": ratio_hm,
            "regime_low": regime_low,
            "regime_mid": regime_mid,
            "regime_high": regime_high,
            "regime_burst_gate": regime_burst_gate,
        },
    }


def make_multiscale_tft_dataset(**kwargs):
    tensors = make_multiscale_tft_tensors(**kwargs)
    return TensorDataset(
        tensors["x_enc"],
        tensors["x_mark_enc"],
        tensors["x_dec"],
        tensors["x_mark_dec"],
        tensors["y_future"],
    )