"""
Smoke tests for Spatial Pattern Experts

Quick validation tests for spatial experts to ensure they can be
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
    from layers.modular.experts.spatial.local_expert import LocalSpatialExpert
    from layers.modular.experts.spatial.global_expert import GlobalSpatialExpert
    from layers.modular.experts.spatial.hierarchical_expert import HierarchicalSpatialExpert
    from layers.modular.experts.base_expert import ExpertOutput
except ImportError as e:
    pytest.skip(f"Spatial experts not available: {e}", allow_module_level=True)


class TestSpatialExpertsSmoke:
    """Smoke test suite for spatial pattern experts."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Setup basic test configuration."""
        self.config = MagicMock()
        self.config.d_model = 128
        self.config.dropout = 0.1
        self.config.seq_len = 32
        self.config.pred_len = 8
        self.config.enc_in = 4
        
        # Local expert specific
        self.config.local_neighborhood_size = 3
        self.config.local_spatial_heads = 4
        
        # Global expert specific
        self.config.global_spatial_heads = 8
        
        # Hierarchical expert specific
        self.config.spatial_hierarchy_levels = 3
        self.config.hierarchical_heads = 4
        
        self.batch_size = 2
        self.seq_len = 32
        self.d_model = 128
        self.device = torch.device('cpu')
        
        # Test data with spatial patterns
        self.test_input = self._generate_test_data()
    
    def _generate_test_data(self):
        """Generate test data with spatial patterns."""
        data = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        
        for b in range(self.batch_size):
            for t in range(self.seq_len):
                for d in range(self.d_model):
                    # Add local spatial correlation
                    if d > 0:
                        data[b, t, d] = 0.7 * data[b, t, d-1] + torch.randn(1) * 0.3
                    else:
                        data[b, t, d] = torch.randn(1)
                    
                    # Add global spatial pattern
                    data[b, t, d] += 0.1 * torch.sin(2 * np.pi * d / self.d_model)
        
        return data
    
    def test_local_spatial_expert_instantiation(self):
        """Test local spatial expert can be instantiated."""
        expert = LocalSpatialExpert(self.config)
        
        assert expert is not None
        assert hasattr(expert, 'local_conv')
        assert hasattr(expert, 'local_attention')
        assert hasattr(expert, 'spatial_strength')
        
        # Test expert info
        info = expert.get_expert_info()
        assert info['expert_type'] == 'spatial'
        assert info['expert_name'] == 'local_spatial_expert'
        
        print(f"✓ Local spatial expert instantiated")
        print(f"  Parameters: {info['parameters']:,}")
        print(f"  Neighborhood size: {expert.neighborhood_size}")
    
    def test_local_spatial_expert_forward(self):
        """Test local spatial expert forward pass."""
        expert = LocalSpatialExpert(self.config)
        
        output = expert(self.test_input)
        
        assert isinstance(output, ExpertOutput)
        assert output.output.shape[0] == self.batch_size
        assert output.output.shape[1] == self.seq_len
        assert output.confidence.shape == (self.batch_size, self.seq_len, 1)
        
        # Check metadata
        assert 'neighborhood_size' in output.metadata
        assert 'spatial_strength_score' in output.metadata
        assert 'local_connectivity' in output.metadata
        
        # Check attention weights
        assert output.attention_weights is not None
        
        print(f"✓ Local spatial expert forward pass works")
        print(f"  Output shape: {output.output.shape}")
        print(f"  Spatial strength: {output.metadata['spatial_strength_score']:.3f}")
    
    def test_global_spatial_expert_instantiation(self):
        """Test global spatial expert can be instantiated."""
        expert = GlobalSpatialExpert(self.config)
        
        assert expert is not None
        assert hasattr(expert, 'global_attention')
        assert hasattr(expert, 'global_aggregator')
        assert hasattr(expert, 'pattern_detector')
        
        # Test expert info
        info = expert.get_expert_info()
        assert info['expert_type'] == 'spatial'
        assert info['expert_name'] == 'global_spatial_expert'
        
        print(f"✓ Global spatial expert instantiated")
        print(f"  Parameters: {info['parameters']:,}")
    
    def test_global_spatial_expert_forward(self):
        """Test global spatial expert forward pass."""
        expert = GlobalSpatialExpert(self.config)
        
        output = expert(self.test_input)
        
        assert isinstance(output, ExpertOutput)
        assert output.output.shape[0] == self.batch_size
        assert output.output.shape[1] == self.seq_len
        assert output.confidence.shape == (self.batch_size, self.seq_len, 1)
        
        # Check metadata
        assert 'global_pattern_strength' in output.metadata
        assert 'global_connectivity' in output.metadata
        assert 'attention_entropy' in output.metadata
        
        # Check attention weights
        assert output.attention_weights is not None
        
        print(f"✓ Global spatial expert forward pass works")
        print(f"  Output shape: {output.output.shape}")
        print(f"  Pattern strength: {output.metadata['global_pattern_strength']:.3f}")
        print(f"  Attention entropy: {output.metadata['attention_entropy']:.3f}")
    
    def test_hierarchical_spatial_expert_instantiation(self):
        """Test hierarchical spatial expert can be instantiated."""
        expert = HierarchicalSpatialExpert(self.config)
        
        assert expert is not None
        assert hasattr(expert, 'spatial_processors')
        assert hasattr(expert, 'hierarchical_attention')
        assert hasattr(expert, 'level_fusion')
        assert hasattr(expert, 'hierarchy_strength')
        
        # Test expert info
        info = expert.get_expert_info()
        assert info['expert_type'] == 'spatial'
        assert info['expert_name'] == 'hierarchical_spatial_expert'
        
        print(f"✓ Hierarchical spatial expert instantiated")
        print(f"  Parameters: {info['parameters']:,}")
        print(f"  Hierarchy levels: {expert.hierarchy_levels}")
    
    def test_hierarchical_spatial_expert_forward(self):
        """Test hierarchical spatial expert forward pass."""
        expert = HierarchicalSpatialExpert(self.config)
        
        output = expert(self.test_input)
        
        assert isinstance(output, ExpertOutput)
        assert output.output.shape[0] == self.batch_size
        assert output.output.shape[1] == self.seq_len
        assert output.confidence.shape == (self.batch_size, self.seq_len, 1)
        
        # Check metadata
        assert 'hierarchy_levels' in output.metadata
        assert 'hierarchy_strengths' in output.metadata
        assert 'dominant_hierarchy_level' in output.metadata
        assert 'hierarchy_coherence' in output.metadata
        assert 'level_attention_patterns' in output.metadata
        
        # Check attention weights (top-level)
        assert output.attention_weights is not None
        
        print(f"✓ Hierarchical spatial expert forward pass works")
        print(f"  Output shape: {output.output.shape}")
        print(f"  Dominant level: {output.metadata['dominant_hierarchy_level']}")
        print(f"  Hierarchy coherence: {output.metadata['hierarchy_coherence']:.3f}")
    
    def test_spatial_experts_with_different_configs(self):
        """Test spatial experts with various configurations."""
        # Test local expert with different neighborhood size
        local_config = MagicMock()
        local_config.d_model = 64
        local_config.dropout = 0.1
        local_config.local_neighborhood_size = 5
        local_config.local_spatial_heads = 2
        
        local_expert = LocalSpatialExpert(local_config, neighborhood_size=7)
        test_data = torch.randn(1, 16, 64)
        output = local_expert(test_data)
        
        assert output.metadata['neighborhood_size'] == 7
        
        # Test hierarchical expert with different levels
        hierarchical_config = MagicMock()
        hierarchical_config.d_model = 64
        hierarchical_config.dropout = 0.1
        hierarchical_config.hierarchical_heads = 2
        
        hierarchical_expert = HierarchicalSpatialExpert(hierarchical_config, hierarchy_levels=2)
        output = hierarchical_expert(test_data)
        
        assert output.metadata['hierarchy_levels'] == 2
        
        print(f"✓ Spatial experts work with different configurations")
    
    def test_spatial_experts_gradient_flow(self):
        """Test that gradients flow through spatial experts."""
        experts = [
            LocalSpatialExpert(self.config),
            GlobalSpatialExpert(self.config),
            HierarchicalSpatialExpert(self.config)
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
        
        print(f"✓ Gradient flow works for all spatial experts")
    
    def test_spatial_experts_attention_patterns(self):
        """Test spatial experts produce meaningful attention patterns."""
        # Test local expert attention
        local_expert = LocalSpatialExpert(self.config)
        local_output = local_expert(self.test_input)
        
        local_attention = local_output.attention_weights
        assert local_attention.shape[0] == self.batch_size  # batch dimension
        assert local_attention.dim() >= 3  # at least [batch, heads, seq_len]
        
        # Test global expert attention
        global_expert = GlobalSpatialExpert(self.config)
        global_output = global_expert(self.test_input)
        
        global_attention = global_output.attention_weights
        assert global_attention.shape[0] == self.batch_size
        
        # Test hierarchical expert attention
        hierarchical_expert = HierarchicalSpatialExpert(self.config)
        hierarchical_output = hierarchical_expert(self.test_input)
        
        hierarchical_attention = hierarchical_output.attention_weights
        assert hierarchical_attention.shape[0] == self.batch_size
        
        print(f"✓ Spatial experts produce valid attention patterns")
        print(f"  Local attention shape: {local_attention.shape}")
        print(f"  Global attention shape: {global_attention.shape}")
        print(f"  Hierarchical attention shape: {hierarchical_attention.shape}")
    
    def test_spatial_experts_confidence_estimation(self):
        """Test spatial experts produce reasonable confidence estimates."""
        experts = [
            LocalSpatialExpert(self.config),
            GlobalSpatialExpert(self.config),
            HierarchicalSpatialExpert(self.config)
        ]
        
        for expert in experts:
            output = expert(self.test_input)
            
            # Check confidence range
            confidence = output.confidence
            assert torch.all(confidence >= 0.0), f"{expert.expert_name} confidence below 0"
            assert torch.all(confidence <= 1.0), f"{expert.expert_name} confidence above 1"
            
            # Check confidence variability
            confidence_std = confidence.std().item()
            assert confidence_std > 0.001, f"{expert.expert_name} confidence too uniform"
            
            print(f"  ✓ {expert.expert_name}: confidence ∈ [{confidence.min():.3f}, {confidence.max():.3f}], std={confidence_std:.3f}")
        
        print(f"✓ Spatial experts produce reasonable confidence estimates")
    
    def test_spatial_experts_device_compatibility(self):
        """Test spatial experts work on different devices."""
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        
        for device in devices:
            expert = LocalSpatialExpert(self.config)
            expert = expert.to(device)
            
            test_data = self.test_input.to(device)
            output = expert(test_data)
            
            assert output.output.device == device
            assert output.confidence.device == device
            assert output.attention_weights.device == device
            
            print(f"  ✓ Local spatial expert works on {device}")
        
        print(f"✓ Device compatibility verified")
    
    def test_spatial_experts_different_sequence_lengths(self):
        """Test spatial experts work with different sequence lengths."""
        sequence_lengths = [8, 16, 32, 64]
        
        for seq_len in sequence_lengths:
            test_data = torch.randn(1, seq_len, self.d_model)
            
            # Test local expert
            local_expert = LocalSpatialExpert(self.config)
            local_output = local_expert(test_data)
            
            assert local_output.output.shape[1] == seq_len
            assert local_output.confidence.shape[1] == seq_len
            
            print(f"  ✓ Spatial experts work with seq_len={seq_len}")
        
        print(f"✓ Spatial experts handle different sequence lengths")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])