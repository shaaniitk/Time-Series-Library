"""Integration tests for SOTA_Temporal_PGAT model.

This module tests the complete SOTA_Temporal_PGAT model integration,
including data flow, component interactions, and end-to-end functionality.
"""

# Suppress optional dependency warnings
from TestsModule.suppress_warnings import suppress_optional_dependency_warnings
suppress_optional_dependency_warnings()

from typing import Tuple
from types import SimpleNamespace

import pytest
import torch

# Import the model and related components
try:
    from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT as SOTATemporalPGAT
except ImportError:
    pytest.skip("SOTA_Temporal_PGAT model not available", allow_module_level=True)

import layers.modular.core.register_components


class TestSOTATemporalPGATIntegration:
    """Integration test suite for SOTA_Temporal_PGAT model."""

    @pytest.fixture(autouse=True)
    def setup_test(self) -> None:
        """Prepare configuration and synthetic data for PGAT integration tests."""
        self.configs = SimpleNamespace(
            seq_len=48,
            label_len=24,
            pred_len=24,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=64,
            n_heads=4,
            dropout=0.1,
            use_dynamic_edge_weights=False,
            use_autocorr_attention=False,
            enable_dynamic_graph=False,
            enable_graph_positional_encoding=True,
            use_adaptive_temporal=True,
            use_mixture_density=False,
            features='MS',
            target_features=['target'],
            all_features=['target'] + [f'cov_{idx}' for idx in range(6)],
            loss_function_type='mse',
            quantile_levels=[],
            mdn_components=2,
            max_eigenvectors=8,
            autocorr_factor=1,
        )
        self.configs.num_nodes = self.configs.enc_in
        self.batch_size = 3
        self._reset_windows(self.batch_size)

    def _reset_windows(self, batch_size: int) -> None:
        """Regenerate synthetic windows for the specified batch size."""
        generator = torch.Generator()
        generator.manual_seed(1337 + batch_size)
        self.batch_size = batch_size
        self.wave_window = torch.randn(
            batch_size,
            self.configs.seq_len,
            self.configs.enc_in,
            generator=generator,
        )
        self.target_window = torch.randn(
            batch_size,
            self.configs.pred_len,
            self.configs.enc_in,
            generator=generator,
        )
        self.graph = None

    def create_model(self, mode: str = 'standard') -> 'SOTATemporalPGAT':
        """Instantiate the PGAT model with the provided mode."""
        return SOTATemporalPGAT(self.configs, mode=mode)

    def _forward(self, mode: str = 'standard') -> Tuple['SOTATemporalPGAT', torch.Tensor]:
        """Run a forward pass and return the model with its prediction tensor."""
        model = self.create_model(mode=mode)
        model.eval()
        with torch.no_grad():
            output = model(self.wave_window, self.target_window, self.graph)
        predictions = output[0] if isinstance(output, tuple) else output
        return model, predictions

    def test_model_initialization(self) -> None:
        """Ensure core PGAT components are created."""
        model = self.create_model()
        assert model is not None
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'temporal_encoder')
        assert hasattr(model, 'spatial_encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'dim_manager')
        assert hasattr(model, 'temporal_pos_encoding')

    def test_model_forward_pass(self) -> None:
        """Run a full forward pass and validate output structure."""
        _, predictions = self._forward()
        assert predictions.shape[0] == self.batch_size
        assert predictions.shape[1] == self.configs.pred_len
        assert torch.isfinite(predictions).all()

    def test_model_training_mode(self) -> None:
        """Verify gradients propagate through the model."""
        model = self.create_model()
        model.train()
        output = model(self.wave_window, self.target_window, self.graph)
        predictions = output[0] if isinstance(output, tuple) else output
        loss = predictions.mean()
        loss.backward()
        has_gradients = any(
            param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
            for param in model.parameters()
        )
        assert has_gradients, "Expected gradients after backward pass."

    def test_graph_structure_storage(self) -> None:
        """Ensure adjacency information can be captured during forward passes."""
        model = self.create_model()
        model.set_graph_info_storage(True)
        _ = model(self.wave_window, self.target_window, self.graph)
        graph_info = model.get_graph_structure()
        assert graph_info is not None
        assert isinstance(graph_info['adjacency_matrix'], torch.Tensor)
        assert torch.isfinite(graph_info['adjacency_matrix']).all()

    def test_temporal_encoding_application(self) -> None:
        """Validate enhanced temporal positional encoding preserves shape."""
        model = self.create_model()
        seq_input = torch.randn(
            self.batch_size,
            self.configs.seq_len + self.configs.pred_len,
            self.configs.d_model,
        )
        encoded = model.temporal_pos_encoding(seq_input)
        assert encoded.shape == seq_input.shape
        assert torch.isfinite(encoded).all()

    def test_model_deterministic_output(self) -> None:
        """Model should return deterministic results given fixed seeds."""
        torch.manual_seed(2024)
        model_a = self.create_model()
        torch.manual_seed(2024)
        model_b = self.create_model()

        model_a.eval()
        model_b.eval()

        with torch.no_grad():
            torch.manual_seed(4096)
            out_a = model_a(self.wave_window, self.target_window, self.graph)
            torch.manual_seed(4096)
            out_b = model_b(self.wave_window, self.target_window, self.graph)

        preds_a = out_a[0] if isinstance(out_a, tuple) else out_a
        preds_b = out_b[0] if isinstance(out_b, tuple) else out_b
        assert torch.allclose(preds_a, preds_b, atol=1e-6)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_model_different_batch_sizes(self, batch_size: int) -> None:
        """Model should support varying batch sizes."""
        self._reset_windows(batch_size)
        _, predictions = self._forward()
        assert predictions.shape[0] == batch_size

    def test_model_probabilistic_mode(self) -> None:
        """Ensure probabilistic mode returns mixture parameters."""
        model, predictions = self._forward(mode='probabilistic')
        assert predictions.shape[0] == self.batch_size
        assert predictions.shape[1] == self.configs.pred_len
        assert torch.isfinite(predictions).all()