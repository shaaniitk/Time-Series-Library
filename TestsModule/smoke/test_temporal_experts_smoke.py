"""
Smoke tests for Temporal Pattern Experts

Quick validation tests for temporal experts to ensure they can be
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
    from layers.modular.experts.temporal.seasonal_expert import SeasonalPatternExpert
    from layers.modular.experts.temporal.trend_expert import TrendPatternExpert
    from layers.modular.experts.temporal.volatility_expert import VolatilityPatternExpert
    from layers.modular.experts.temporal.regime_expert import RegimePatternExpert
    from layers.modular.experts.base_expert import ExpertOutput
except ImportError as e:
    pytest.skip(f"Temporal experts not available: {e}", allow_module_level=True)


class TestTemporalExpertsSmoke:
    """Smoke test suite for temporal pattern experts."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Setup basic test configuration."""
        self.config = MagicMock()
        self.config.d_model = 128
        self.config.dropout = 0.1
        self.config.seq_len = 48
        self.config.pred_len = 12
        self.config.enc_in = 4
        
        # Seasonal expert specific
        self.config.seasonal_decomp_kernel = 15
        self.config.seasonal_harmonics = 5
        self.config.seasonal_heads = 4
        
        # Trend expert specific
        self.config.trend_diff_order = 2
        self.config.trend_heads = 4
        self.config.trend_windows = [3, 6, 12]
        
        # Volatility expert specific
        self.config.garch_order = (1, 1)
        
        # Regime expert specific
        self.config.num_regimes = 3
        self.config.regime_detection_window = 10
        
        self.batch_size = 2
        self.seq_len = 48
        self.d_model = 128
        self.device = torch.device('cpu')
        
        # Test data with temporal patterns
        self.test_input = self._generate_test_data()
    
    def _generate_test_data(self):
        """Generate test data with known temporal patterns."""
        t = torch.linspace(0, 4*np.pi, self.seq_len)
        data = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        
        for b in range(self.batch_size):
            for d in range(self.d_model):
                # Add seasonal component
                seasonal = torch.sin(2 * t + d * 0.1)
                # Add trend component
                trend = 0.01 * t * (d + 1)
                # Add noise
                noise = torch.randn(self.seq_len) * 0.1
                
                data[b, :, d] = seasonal + trend + noise
        
        return data
    
    def test_seasonal_expert_instantiation(self):
        """Test seasonal expert can be instantiated."""
        expert = SeasonalPatternExpert(self.config)
        
        assert expert is not None
        assert hasattr(expert, 'seasonal_decomp')
        assert hasattr(expert, 'fourier_encoder')
        assert hasattr(expert, 'seasonal_attention')
        
        # Test expert info
        info = expert.get_expert_info()
        assert info['expert_type'] == 'temporal'
        assert info['expert_name'] == 'seasonal_expert'
        
        print(f"✓ Seasonal expert instantiated")
        print(f"  Parameters: {info['parameters']:,}")
    
    def test_seasonal_expert_forward(self):
        """Test seasonal expert forward pass."""
        expert = SeasonalPatternExpert(self.config)
        
        output = expert(self.test_input)
        
        assert isinstance(output, ExpertOutput)
        assert output.output.shape[0] == self.batch_size
        assert output.output.shape[1] == self.seq_len
        assert output.confidence.shape == (self.batch_size, self.seq_len, 1)
        
        # Check metadata
        assert 'seasonal_periods' in output.metadata
        assert 'seasonal_strengths' in output.metadata
        assert 'dominant_period' in output.metadata
        
        print(f"✓ Seasonal expert forward pass works")
        print(f"  Output shape: {output.output.shape}")
        print(f"  Dominant period: {output.metadata['dominant_period']}")
    
    def test_seasonal_expert_seasonality_detection(self):
        """Test seasonal expert seasonality detection."""
        expert = SeasonalPatternExpert(self.config)
        
        seasonality_info = expert.detect_seasonality(self.test_input)
        
        assert 'seasonal_strengths' in seasonality_info
        assert 'seasonality_score' in seasonality_info
        assert 'dominant_frequencies' in seasonality_info
        
        print(f"✓ Seasonality detection works")
        print(f"  Seasonality score: {seasonality_info['seasonality_score']:.3f}")
    
    def test_trend_expert_instantiation(self):
        """Test trend expert can be instantiated."""
        expert = TrendPatternExpert(self.config)
        
        assert expert is not None
        assert hasattr(expert, 'trend_decomp')
        assert hasattr(expert, 'trend_strength')
        
        # Test expert info
        info = expert.get_expert_info()
        assert info['expert_type'] == 'temporal'
        assert info['expert_name'] == 'trend_expert'
        
        print(f"✓ Trend expert instantiated")
        print(f"  Parameters: {info['parameters']:,}")
    
    def test_trend_expert_forward(self):
        """Test trend expert forward pass."""
        expert = TrendPatternExpert(self.config)
        
        output = expert(self.test_input)
        
        assert isinstance(output, ExpertOutput)
        assert output.output.shape[0] == self.batch_size
        assert output.output.shape[1] == self.seq_len
        assert output.confidence.shape == (self.batch_size, self.seq_len, 1)
        
        # Check metadata
        assert 'trend_types' in output.metadata
        assert 'trend_strengths' in output.metadata
        assert 'dominant_trend_type' in output.metadata
        
        print(f"✓ Trend expert forward pass works")
        print(f"  Output shape: {output.output.shape}")
        print(f"  Dominant trend: {output.metadata['dominant_trend_type']}")
    
    def test_volatility_expert_instantiation(self):
        """Test volatility expert can be instantiated."""
        expert = VolatilityPatternExpert(self.config)
        
        assert expert is not None
        assert hasattr(expert, 'volatility_estimator')
        assert hasattr(expert, 'garch_layer')
        assert hasattr(expert, 'clustering_detector')
        
        # Test expert info
        info = expert.get_expert_info()
        assert info['expert_type'] == 'temporal'
        assert info['expert_name'] == 'volatility_expert'
        
        print(f"✓ Volatility expert instantiated")
        print(f"  Parameters: {info['parameters']:,}")
    
    def test_volatility_expert_forward(self):
        """Test volatility expert forward pass."""
        expert = VolatilityPatternExpert(self.config)
        
        output = expert(self.test_input)
        
        assert isinstance(output, ExpertOutput)
        assert output.output.shape[0] == self.batch_size
        assert output.output.shape[1] == self.seq_len
        assert output.confidence.shape == (self.batch_size, self.seq_len, 1)
        
        # Check metadata
        assert 'window_sizes' in output.metadata
        assert 'volatility_strength_score' in output.metadata
        assert 'garch_parameters' in output.metadata
        
        # Check uncertainty (volatility clustering)
        assert output.uncertainty is not None
        assert output.uncertainty.shape == (self.batch_size, self.seq_len, 1)
        
        print(f"✓ Volatility expert forward pass works")
        print(f"  Output shape: {output.output.shape}")
        print(f"  Volatility strength: {output.metadata['volatility_strength_score']:.3f}")
    
    def test_regime_expert_instantiation(self):
        """Test regime expert can be instantiated."""
        expert = RegimePatternExpert(self.config)
        
        assert expert is not None
        assert hasattr(expert, 'regime_detector')
        assert hasattr(expert, 'regime_characterizer')
        assert hasattr(expert, 'stability_estimator')
        
        # Test expert info
        info = expert.get_expert_info()
        assert info['expert_type'] == 'temporal'
        assert info['expert_name'] == 'regime_expert'
        
        print(f"✓ Regime expert instantiated")
        print(f"  Parameters: {info['parameters']:,}")
    
    def test_regime_expert_forward(self):
        """Test regime expert forward pass."""
        expert = RegimePatternExpert(self.config)
        
        output = expert(self.test_input)
        
        assert isinstance(output, ExpertOutput)
        assert output.output.shape[0] == self.batch_size
        assert output.output.shape[1] == self.seq_len
        assert output.confidence.shape == (self.batch_size, self.seq_len, 1)
        
        # Check metadata
        assert 'num_regimes' in output.metadata
        assert 'regime_probabilities' in output.metadata
        assert 'dominant_regime' in output.metadata
        assert 'detected_change_points' in output.metadata
        
        # Check uncertainty (change strength)
        assert output.uncertainty is not None
        assert output.uncertainty.shape == (self.batch_size, self.seq_len, 1)
        
        print(f"✓ Regime expert forward pass works")
        print(f"  Output shape: {output.output.shape}")
        print(f"  Dominant regime: {output.metadata['dominant_regime']}")
    
    def test_regime_expert_analysis(self):
        """Test regime expert analysis capabilities."""
        expert = RegimePatternExpert(self.config)
        
        # Test regime summary
        summary = expert.get_regime_summary(self.test_input)
        
        assert 'regime_probabilities' in summary
        assert 'regime_assignments' in summary
        assert 'regime_durations' in summary
        assert 'change_points' in summary
        
        print(f"✓ Regime analysis works")
        print(f"  Detected change points: {len(summary['change_points'][0])}")
    
    def test_temporal_experts_with_different_configs(self):
        """Test temporal experts with various configurations."""
        # Test seasonal expert with different periods
        seasonal_config = MagicMock()
        seasonal_config.d_model = 64
        seasonal_config.dropout = 0.1
        seasonal_config.seq_len = 24
        
        seasonal_expert = SeasonalPatternExpert(seasonal_config, seasonal_periods=[7, 14])
        test_data = torch.randn(1, 24, 64)
        output = seasonal_expert(test_data)
        
        assert output.metadata['seasonal_periods'] == [7, 14]
        
        # Test trend expert with different types
        trend_expert = TrendPatternExpert(seasonal_config, trend_types=['linear', 'exponential'])
        output = trend_expert(test_data)
        
        assert output.metadata['trend_types'] == ['linear', 'exponential']
        
        # Test volatility expert with different windows
        volatility_expert = VolatilityPatternExpert(seasonal_config, window_sizes=[3, 6])
        output = volatility_expert(test_data)
        
        assert output.metadata['window_sizes'] == [3, 6]
        
        print(f"✓ Temporal experts work with different configurations")
    
    def test_temporal_experts_gradient_flow(self):
        """Test that gradients flow through temporal experts."""
        experts = [
            SeasonalPatternExpert(self.config),
            TrendPatternExpert(self.config),
            VolatilityPatternExpert(self.config),
            RegimePatternExpert(self.config)
        ]
        
        for expert in experts:
            expert.train()
            
            # Forward pass
            output = expert(self.test_input)
            
            # Compute loss
            loss = output.output.mean()
            
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
        
        print(f"✓ Gradient flow works for all temporal experts")
    
    def test_temporal_experts_device_compatibility(self):
        """Test temporal experts work on different devices."""
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        
        for device in devices:
            expert = SeasonalPatternExpert(self.config)
            expert = expert.to(device)
            
            test_data = self.test_input.to(device)
            output = expert(test_data)
            
            assert output.output.device == device
            assert output.confidence.device == device
            
            print(f"  ✓ Seasonal expert works on {device}")
        
        print(f"✓ Device compatibility verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])