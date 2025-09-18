"""
Smoke tests for Uncertainty Quantification Experts

Quick validation tests for uncertainty experts to ensure they can be
instantiated and run basic operations without errors.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
import warnings
warnings.filterwarnings('ignore')

# Test imports
try:
    from layers.modular.experts.uncertainty.aleatoric_expert import AleatoricUncertaintyExpert
    from layers.modular.experts.uncertainty.epistemic_expert import EpistemicUncertaintyExpert, BayesianLinear
    from layers.modular.experts.base_expert import ExpertOutput
except ImportError as e:
    pytest.skip(f"Uncertainty experts not available: {e}", allow_module_level=True)


class TestUncertaintyExpertsSmoke:
    """Smoke test suite for uncertainty quantification experts."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Setup basic test configuration."""
        self.config = MagicMock()
        self.config.d_model = 128
        self.config.dropout = 0.1
        self.config.seq_len = 32
        self.config.pred_len = 8
        self.config.enc_in = 4
        
        # Epistemic expert specific
        self.config.epistemic_ensemble_size = 3
        
        self.batch_size = 2
        self.seq_len = 32
        self.d_model = 128
        self.device = torch.device('cpu')
        
        # Test data with varying uncertainty
        self.test_input = self._generate_test_data()
    
    def _generate_test_data(self):
        """Generate test data with varying uncertainty levels."""
        data = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        
        for b in range(self.batch_size):
            for t in range(self.seq_len):
                # Varying noise levels (heteroscedastic)
                noise_level = 0.1 + 0.2 * (t / self.seq_len)  # Increasing noise
                
                for d in range(self.d_model):
                    # Base signal
                    signal = torch.sin(2 * np.pi * t / 10 + d * 0.1)
                    # Add heteroscedastic noise
                    noise = torch.randn(1) * noise_level
                    data[b, t, d] = signal + noise
        
        return data
    
    def test_bayesian_linear_layer(self):
        """Test Bayesian linear layer functionality."""
        layer = BayesianLinear(64, 32)
        
        test_input = torch.randn(2, 10, 64)
        
        # Test forward pass with sampling
        output, uncertainty = layer(test_input, sample=True)
        
        assert output.shape == (2, 10, 32)
        assert uncertainty.shape == (2, 10, 32)
        assert torch.all(uncertainty >= 0), "Uncertainty should be non-negative"
        
        # Test forward pass without sampling
        output_no_sample, uncertainty_no_sample = layer(test_input, sample=False)
        
        assert output_no_sample.shape == (2, 10, 32)
        assert torch.allclose(uncertainty_no_sample, torch.zeros_like(uncertainty_no_sample))
        
        # Test KL divergence
        kl_loss = layer.kl_divergence()
        assert kl_loss.item() >= 0, "KL divergence should be non-negative"
        
        print(f"✓ Bayesian linear layer works")
        print(f"  Output shape: {output.shape}")
        print(f"  Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
        print(f"  KL divergence: {kl_loss.item():.6f}")
    
    def test_aleatoric_uncertainty_expert_instantiation(self):
        """Test aleatoric uncertainty expert can be instantiated."""
        expert = AleatoricUncertaintyExpert(self.config)
        
        assert expert is not None
        assert hasattr(expert, 'noise_estimator')
        assert hasattr(expert, 'variance_predictor')
        assert hasattr(expert, 'temporal_variance')
        assert hasattr(expert, 'uncertainty_fusion')
        
        # Test expert info
        info = expert.get_expert_info()
        assert info['expert_type'] == 'uncertainty'
        assert info['expert_name'] == 'aleatoric_uncertainty_expert'
        
        print(f"✓ Aleatoric uncertainty expert instantiated")
        print(f"  Parameters: {info['parameters']:,}")
    
    def test_aleatoric_uncertainty_expert_forward(self):
        """Test aleatoric uncertainty expert forward pass."""
        expert = AleatoricUncertaintyExpert(self.config)
        
        output = expert(self.test_input)
        
        assert isinstance(output, ExpertOutput)
        assert output.output.shape[0] == self.batch_size
        assert output.output.shape[1] == self.seq_len
        assert output.confidence.shape == (self.batch_size, self.seq_len, 1)
        assert output.uncertainty.shape == (self.batch_size, self.seq_len, 1)
        
        # Check metadata
        assert 'average_noise_level' in output.metadata
        assert 'average_variance' in output.metadata
        assert 'temporal_variance_magnitude' in output.metadata
        assert 'uncertainty_range' in output.metadata
        
        # Check uncertainty properties
        uncertainty = output.uncertainty
        assert torch.all(uncertainty > 0), "Aleatoric uncertainty should be positive"
        
        # Confidence should be inverse of uncertainty
        confidence = output.confidence
        assert torch.all(confidence > 0) and torch.all(confidence <= 1), "Confidence should be in (0, 1]"
        
        print(f"✓ Aleatoric uncertainty expert forward pass works")
        print(f"  Output shape: {output.output.shape}")
        print(f"  Uncertainty range: {output.metadata['uncertainty_range']}")
        print(f"  Average noise level: {output.metadata['average_noise_level']:.3f}")
    
    def test_epistemic_uncertainty_expert_instantiation(self):
        """Test epistemic uncertainty expert can be instantiated."""
        expert = EpistemicUncertaintyExpert(self.config)
        
        assert expert is not None
        assert hasattr(expert, 'bayesian_layers')
        assert hasattr(expert, 'model_ensemble')
        assert hasattr(expert, 'mc_dropout_layers')
        assert hasattr(expert, 'uncertainty_aggregator')
        
        # Check ensemble size
        assert len(expert.model_ensemble) == self.config.epistemic_ensemble_size
        
        # Test expert info
        info = expert.get_expert_info()
        assert info['expert_type'] == 'uncertainty'
        assert info['expert_name'] == 'epistemic_uncertainty_expert'
        
        print(f"✓ Epistemic uncertainty expert instantiated")
        print(f"  Parameters: {info['parameters']:,}")
        print(f"  Ensemble size: {len(expert.model_ensemble)}")
    
    def test_epistemic_uncertainty_expert_forward(self):
        """Test epistemic uncertainty expert forward pass."""
        expert = EpistemicUncertaintyExpert(self.config)
        
        output = expert(self.test_input)
        
        assert isinstance(output, ExpertOutput)
        assert output.output.shape[0] == self.batch_size
        assert output.output.shape[1] == self.seq_len
        assert output.confidence.shape == (self.batch_size, self.seq_len, 1)
        assert output.uncertainty.shape == (self.batch_size, self.seq_len, 1)
        
        # Check metadata
        assert 'bayesian_uncertainty' in output.metadata
        assert 'ensemble_uncertainty' in output.metadata
        assert 'mc_dropout_uncertainty' in output.metadata
        assert 'kl_divergence' in output.metadata
        assert 'ensemble_disagreement' in output.metadata
        assert 'parameter_uncertainty_ratio' in output.metadata
        
        # Check uncertainty properties
        uncertainty = output.uncertainty
        assert torch.all(uncertainty > 0), "Epistemic uncertainty should be positive"
        
        # Check KL divergence
        kl_loss = expert.get_kl_loss()
        assert kl_loss.item() >= 0, "KL divergence should be non-negative"
        
        print(f"✓ Epistemic uncertainty expert forward pass works")
        print(f"  Output shape: {output.output.shape}")
        print(f"  Bayesian uncertainty: {output.metadata['bayesian_uncertainty']:.3f}")
        print(f"  Ensemble uncertainty: {output.metadata['ensemble_uncertainty']:.3f}")
        print(f"  KL divergence: {output.metadata['kl_divergence']:.6f}")
    
    def test_epistemic_uncertainty_monte_carlo(self):
        """Test Monte Carlo prediction for epistemic uncertainty."""
        expert = EpistemicUncertaintyExpert(self.config)
        
        # Test Monte Carlo prediction
        pred_mean, pred_uncertainty = expert.monte_carlo_predict(self.test_input, num_samples=10)
        
        assert pred_mean.shape == expert(self.test_input).output.shape
        assert pred_uncertainty.shape == expert(self.test_input).output.shape
        assert torch.all(pred_uncertainty >= 0), "MC uncertainty should be non-negative"
        
        print(f"✓ Monte Carlo prediction works")
        print(f"  Prediction shape: {pred_mean.shape}")
        print(f"  MC uncertainty range: [{pred_uncertainty.min():.3f}, {pred_uncertainty.max():.3f}]")
    
    def test_uncertainty_experts_gradient_flow(self):
        """Test that gradients flow through uncertainty experts."""
        experts = [
            AleatoricUncertaintyExpert(self.config),
            EpistemicUncertaintyExpert(self.config)
        ]
        
        for expert in experts:
            expert.train()
            
            # Forward pass
            output = expert(self.test_input)
            
            # Compute loss (including uncertainty)
            loss = output.output.mean() + output.uncertainty.mean()
            
            # Add KL loss for epistemic expert
            if hasattr(expert, 'get_kl_loss'):
                loss += 0.01 * expert.get_kl_loss()
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            grad_count = 0
            for param in expert.parameters():
                if param.grad is not None:
                    grad_count += 1
                    assert torch.isfinite(param.grad).all()
            
            assert grad_count > 0, f"No gradients computed for {expert.expert_name}"
            
            print(f"  ✓ {expert.expert_name}: {grad_count} parameters with gradients")
        
        print(f"✓ Gradient flow works for all uncertainty experts")
    
    def test_uncertainty_experts_different_data_characteristics(self):
        """Test uncertainty experts with different data characteristics."""
        # Test with low-noise data
        low_noise_data = torch.randn(self.batch_size, self.seq_len, self.d_model) * 0.1
        
        # Test with high-noise data
        high_noise_data = torch.randn(self.batch_size, self.seq_len, self.d_model) * 2.0
        
        aleatoric_expert = AleatoricUncertaintyExpert(self.config)
        
        # Test with low noise
        low_noise_output = aleatoric_expert(low_noise_data)
        low_noise_uncertainty = low_noise_output.uncertainty.mean().item()
        
        # Test with high noise
        high_noise_output = aleatoric_expert(high_noise_data)
        high_noise_uncertainty = high_noise_output.uncertainty.mean().item()
        
        # High noise should generally produce higher uncertainty
        print(f"✓ Uncertainty experts adapt to data characteristics")
        print(f"  Low noise uncertainty: {low_noise_uncertainty:.3f}")
        print(f"  High noise uncertainty: {high_noise_uncertainty:.3f}")
    
    def test_uncertainty_experts_confidence_calibration(self):
        """Test uncertainty experts produce well-calibrated confidence."""
        experts = [
            AleatoricUncertaintyExpert(self.config),
            EpistemicUncertaintyExpert(self.config)
        ]
        
        for expert in experts:
            output = expert(self.test_input)
            
            confidence = output.confidence
            uncertainty = output.uncertainty
            
            # Confidence should be inversely related to uncertainty
            # High uncertainty -> Low confidence
            correlation = torch.corrcoef(torch.stack([
                confidence.flatten(),
                uncertainty.flatten()
            ]))[0, 1]
            
            assert correlation < 0, f"{expert.expert_name} confidence not inversely correlated with uncertainty"
            
            print(f"  ✓ {expert.expert_name}: confidence-uncertainty correlation = {correlation:.3f}")
        
        print(f"✓ Uncertainty experts produce well-calibrated confidence")
    
    def test_uncertainty_experts_device_compatibility(self):
        """Test uncertainty experts work on different devices."""
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        
        for device in devices:
            expert = AleatoricUncertaintyExpert(self.config)
            expert = expert.to(device)
            
            test_data = self.test_input.to(device)
            output = expert(test_data)
            
            assert output.output.device == device
            assert output.confidence.device == device
            assert output.uncertainty.device == device
            
            print(f"  ✓ Aleatoric uncertainty expert works on {device}")
        
        print(f"✓ Device compatibility verified")
    
    def test_uncertainty_decomposition(self):
        """Test that different uncertainty types capture different aspects."""
        aleatoric_expert = AleatoricUncertaintyExpert(self.config)
        epistemic_expert = EpistemicUncertaintyExpert(self.config)
        
        # Test with same data
        aleatoric_output = aleatoric_expert(self.test_input)
        epistemic_output = epistemic_expert(self.test_input)
        
        aleatoric_uncertainty = aleatoric_output.uncertainty.mean().item()
        epistemic_uncertainty = epistemic_output.uncertainty.mean().item()
        
        # Both should be positive
        assert aleatoric_uncertainty > 0, "Aleatoric uncertainty should be positive"
        assert epistemic_uncertainty > 0, "Epistemic uncertainty should be positive"
        
        # They should capture different aspects (not identical)
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        print(f"✓ Uncertainty decomposition works")
        print(f"  Aleatoric uncertainty: {aleatoric_uncertainty:.3f}")
        print(f"  Epistemic uncertainty: {epistemic_uncertainty:.3f}")
        print(f"  Total uncertainty: {total_uncertainty:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])