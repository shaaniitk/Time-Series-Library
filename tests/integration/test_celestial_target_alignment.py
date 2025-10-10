import torch
import pytest

from models.Celestial_Enhanced_PGAT import Model
from layers.modular.decoder.sequential_mixture_decoder import SequentialMixtureNLLLoss


class MockConfig:
    def __init__(self):
        # Core sequence settings
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24

        # I/O dims
        # To trigger wave aggregation, encoder input must equal num_input_waves
        self.enc_in = 118
        self.dec_in = 4
        self.c_out = 4

        # Model hyperparameters
        self.d_model = 64
        self.n_heads = 4
        self.e_layers = 2
        self.d_layers = 1
        self.dropout = 0.1

        # Embedding parameters expected by the model
        self.embed = 'timeF'
        self.freq = 'h'

        # Feature flags
        self.use_mixture_decoder = True
        self.use_stochastic_learner = True
        self.use_hierarchical_mapping = True

        # Celestial aggregation
        self.aggregate_waves_to_celestial = True
        self.num_input_waves = 118


@pytest.fixture
def config():
    return MockConfig()


@pytest.fixture
def sample_batch(config):
    B = 3
    # Encoder inputs: raw waves to trigger aggregation
    x_enc = torch.randn(B, config.seq_len, config.enc_in)
    # Temporal marks as valid integer indices: [month, day, weekday, hour]
    month = torch.randint(0, 13, (B, config.seq_len))
    day = torch.randint(0, 32, (B, config.seq_len))
    weekday = torch.randint(0, 8, (B, config.seq_len))
    hour = torch.randint(0, 25, (B, config.seq_len))
    x_mark_enc = torch.stack([month, day, weekday, hour], dim=-1)
    # Decoder inputs
    x_dec = torch.randn(B, config.label_len + config.pred_len, config.dec_in)
    month_d = torch.randint(0, 13, (B, config.label_len + config.pred_len))
    day_d = torch.randint(0, 32, (B, config.label_len + config.pred_len))
    weekday_d = torch.randint(0, 8, (B, config.label_len + config.pred_len))
    hour_d = torch.randint(0, 25, (B, config.label_len + config.pred_len))
    x_mark_dec = torch.stack([month_d, day_d, weekday_d, hour_d], dim=-1)
    # Targets
    targets = torch.randn(B, config.pred_len, config.c_out)
    return x_enc, x_mark_enc, x_dec, x_mark_dec, targets


def _extract_outputs_tuple(outputs):
    if isinstance(outputs, dict):
        return outputs["means"], outputs["log_stds"], outputs["log_weights"]
    return None


def test_metadata_original_targets_present_with_aggregation(config, sample_batch):
    x_enc, x_mark_enc, x_dec, x_mark_dec, targets = sample_batch
    model = Model(config)
    model.eval()

    with torch.no_grad():
        outputs, metadata = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert isinstance(metadata, dict), "Metadata should be a dict"
    assert "original_targets" in metadata, "original_targets must be present when aggregation enabled"

    original_targets_full = metadata["original_targets"]
    # Model stores full-sequence targets; align to prediction horizon
    assert original_targets_full is not None
    original_targets = original_targets_full[:, -config.pred_len:, :]
    assert original_targets.shape[:2] == (x_enc.size(0), config.pred_len), "original_targets must align to pred_len"
    assert original_targets.size(-1) == config.c_out, "original_targets last dim must match c_out"

    # Loss computation sanity-check with sequential MDN loss when outputs are mixture params
    mixture_params = _extract_outputs_tuple(outputs)
    loss_fn = SequentialMixtureNLLLoss(reduction="mean")
    if mixture_params is not None:
        loss = loss_fn(mixture_params, original_targets)
        assert torch.isfinite(loss)
        assert loss.dim() == 0 and loss.item() >= 0
    else:
        mse = torch.nn.MSELoss()
        loss = mse(outputs, original_targets)
        assert torch.isfinite(loss)


def test_teacher_forcing_decoder_inputs_change_output(config, sample_batch):
    x_enc, x_mark_enc, x_dec, x_mark_dec, targets = sample_batch
    model = Model(config)
    model.eval()

    # Create slightly different decoder inputs for the prediction horizon
    x_dec_variant = x_dec.clone()
    # Nudge the prediction segment to simulate teacher-forcing differences
    x_dec_variant[:, -config.pred_len:, :] += 0.25

    with torch.no_grad():
        outputs_a, metadata_a = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        outputs_b, metadata_b = model(x_enc, x_mark_enc, x_dec_variant, x_mark_dec)

    # Validate different outputs under different decoder inputs
    if isinstance(outputs_a, dict):
        # Compare means tensors
        diff = torch.abs(outputs_a["means"] - outputs_b["means"]).sum().item()
    else:
        diff = torch.abs(outputs_a - outputs_b).sum().item()

    assert diff > 1e-6, "Different decoder inputs should produce different outputs"

    # Also verify target alignment remains consistent
    assert "original_targets" in metadata_a
    assert "original_targets" in metadata_b
    ta_full = metadata_a["original_targets"]
    tb_full = metadata_b["original_targets"]
    assert ta_full is not None and tb_full is not None
    ta = ta_full[:, -config.pred_len:, :]
    tb = tb_full[:, -config.pred_len:, :]
    assert ta.shape == tb.shape == (x_enc.size(0), config.pred_len, config.c_out)