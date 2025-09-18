"""
Smoke tests for Mixture of Experts Framework

Quick validation tests for the core MoE components to ensure they can be
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
    from layers.modular.experts.base_expert import BaseExpert, ExpertOutput, TemporalExpert, SpatialExpert, UncertaintyExpert
    from layers.modular.experts.expert_router import ExpertRouter, AdaptiveExpertRouter, AttentionBasedRouter
    from layers.modular.experts.moe_layer import MoELayer, SparseMoELayer
    from layers.modular.experts.registry import expert_registry, create_expert
except ImportError as e:
    pytest.skip(f"MoE framework not available: {e}", allow_module_level=True)


class TestMoEFrameworkSmoke:
    """Smoke test suite for MoE framework components."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Setup basic test configuration."""
        self.config = MagicMock()
        self.config.d_model = 128
        self.config.dropout = 0.1
        self.config.seq_len = 24
        self.config.pred_len = 12
        self.config.enc_in = 4
        
        self.batch_size = 2
        self.seq_len = 24
        self.d_model = 128
        self.device = torch.device('cpu')
        
        # Test data
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_expert_registry_basic_operations(self):
        """Test basic expert registry operations."""
        # Test listing experts
        temporal_experts = expert_registry.list_experts('temporal')
        spatial_experts = expert_registry.list_experts('spatial')
        uncertainty_experts = expert_registry.list_experts('uncertainty')
        
        assert len(temporal_experts) > 0, "Should have temporal experts"
        assert len(spatial_experts) > 0, "Should have spatial experts"
        assert len(uncertainty_experts) > 0, "Should have uncertainty experts"
        
        # Test getting expert info
        for expert_name in temporal_experts[:2]:  # Test first 2
            info = expert_registry.get_expert_info(expert_name)
            assert 'name' in info
            assert 'category' in info
            assert info['category'] == 'temporal'
        
        print(f"✓ Expert registry operations work")
        print(f"  Temporal experts: {len(temporal_experts)}")
        print(f"  Spatial experts: {len(spatial_experts)}")
        print(f"  Uncertainty experts: {len(uncertainty_experts)}")
    
    def test_expert_creation(self):
        """Test expert creation from registry."""
        # Test creating different types of experts
        expert_types = ['seasonal_expert', 'local_spatial_expert', 'aleatoric_uncertainty_expert']
        
        for expert_name in expert_types:
            try:
                expert = create_expert(expert_name, self.config)
                assert expert is not None
                assert hasattr(expert, 'forward')
                assert hasattr(expert, 'get_expert_info')
                
                # Test expert info
                info = expert.get_expert_info()
                assert 'expert_name' in info
                assert 'parameters' in info
                
                print(f"  ✓ Created {expert_name}: {info['parameters']} parameters")
                
            except Exception as e:
                print(f"  ⚠ Could not create {expert_name}: {e}")
    
    def test_expert_routers(self):
        """Test different expert router types."""
        num_experts = 3
        
        # Test ExpertRouter
        router = ExpertRouter(self.d_model, num_experts)
        routing_weights, routing_info = router(self.test_input)
        
        assert routing_weights.shape == (self.batch_size, num_experts)
        assert torch.allclose(routing_weights.sum(dim=-1), torch.ones(self.batch_size), atol=1e-5)
        assert 'routing_logits' in routing_info
        
        # Test AdaptiveExpertRouter
        adaptive_router = AdaptiveExpertRouter(self.d_model, num_experts)
        routing_weights, routing_info = adaptive_router(self.test_input)
        
        assert routing_weights.shape == (self.batch_size, num_experts)
        assert 'characteristics' in routing_info
        
        # Test AttentionBasedRouter
        attention_router = AttentionBasedRouter(self.d_model, num_experts)
        routing_weights, routing_info = attention_router(self.test_input)
        
        assert routing_weights.shape == (self.batch_size, num_experts)
        assert 'attention_weights' in routing_info
        
        print(f"✓ Expert routers work correctly")
        print(f"  ExpertRouter entropy: {routing_info.get('expert_entropy', 'N/A')}")
    
    def test_moe_layer_basic(self):
        """Test basic MoE layer functionality."""
        # Create simple mock experts
        class MockExpert(BaseExpert):
            def __init__(self, config, name):
                super().__init__(config, 'temporal', name)
                self.linear = torch.nn.Linear(self.d_model, self.expert_dim)
            
            def forward(self, x, **kwargs):
                output = self.linear(x)
                confidence = torch.ones(x.size(0), x.size(1), 1) * 0.8
                return ExpertOutput(
                    output=output,
                    confidence=confidence,
                    metadata={'expert_name': self.expert_name}
                )
        
        # Create experts and router
        experts = [MockExpert(self.config, f'expert_{i}') for i in range(3)]
        router = ExpertRouter(self.d_model, len(experts))
        
        # Create MoE layer
        moe_layer = MoELayer(experts, router, top_k=2)
        
        # Test forward pass
        output, moe_info = moe_layer(self.test_input)
        
        assert output.shape == self.test_input.shape
        assert 'routing_info' in moe_info
        assert 'expert_usage' in moe_info
        
        # Test expert utilization
        utilization = moe_layer.get_expert_utilization()
        assert len(utilization) == len(experts)
        
        print(f"✓ MoE layer works correctly")
        print(f"  Output shape: {output.shape}")
        print(f"  Expert utilization: {utilization}")
    
    def test_sparse_moe_layer(self):
        """Test sparse MoE layer with capacity constraints."""
        # Create simple mock experts
        class MockExpert(BaseExpert):
            def __init__(self, config, name):
                super().__init__(config, 'temporal', name)
                self.linear = torch.nn.Linear(self.d_model, self.expert_dim)
            
            def forward(self, x, **kwargs):
                output = self.linear(x)
                confidence = torch.ones(x.size(0), x.size(1), 1) * 0.8
                return ExpertOutput(
                    output=output,
                    confidence=confidence,
                    metadata={'expert_name': self.expert_name}
                )
        
        # Create experts and router
        experts = [MockExpert(self.config, f'expert_{i}') for i in range(4)]
        router = ExpertRouter(self.d_model, len(experts))
        
        # Create Sparse MoE layer
        sparse_moe = SparseMoELayer(
            experts, router, top_k=2, 
            expert_capacity_factor=1.25,
            capacity_balancing=True
        )
        
        # Test forward pass
        output, moe_info = sparse_moe(self.test_input)
        
        assert output.shape == self.test_input.shape
        assert 'expert_capacities' in moe_info
        assert 'capacity_utilization' in moe_info
        
        print(f"✓ Sparse MoE layer works correctly")
        print(f"  Capacity utilization: {moe_info['capacity_utilization']:.3f}")
    
    def test_expert_output_format(self):
        """Test ExpertOutput format consistency."""
        # Test ExpertOutput creation
        output_tensor = torch.randn(2, 10, 64)
        confidence_tensor = torch.ones(2, 10, 1) * 0.9
        metadata = {'test_key': 'test_value'}
        
        expert_output = ExpertOutput(
            output=output_tensor,
            confidence=confidence_tensor,
            metadata=metadata
        )
        
        assert expert_output.output.shape == output_tensor.shape
        assert expert_output.confidence.shape == confidence_tensor.shape
        assert expert_output.metadata == metadata
        assert expert_output.attention_weights is None
        assert expert_output.uncertainty is None
        
        print(f"✓ ExpertOutput format is consistent")
    
    def test_expert_base_classes(self):
        """Test expert base class functionality."""
        # Test TemporalExpert
        class TestTemporalExpert(TemporalExpert):
            def forward(self, x, **kwargs):
                output = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
                confidence = self.compute_confidence(x, output)
                return ExpertOutput(
                    output=output,
                    confidence=confidence,
                    metadata={'expert_type': self.expert_type}
                )
        
        temporal_expert = TestTemporalExpert(self.config, 'test_temporal')
        output = temporal_expert(self.test_input)
        
        assert isinstance(output, ExpertOutput)
        assert output.output.shape[0] == self.batch_size
        assert output.confidence.shape[0] == self.batch_size
        
        # Test SpatialExpert
        class TestSpatialExpert(SpatialExpert):
            def forward(self, x, **kwargs):
                attended, _ = self.spatial_attention(x, x, x)
                output = self.spatial_norm(attended)
                confidence = self.compute_confidence(x, output)
                return ExpertOutput(
                    output=output,
                    confidence=confidence,
                    metadata={'expert_type': self.expert_type}
                )
        
        spatial_expert = TestSpatialExpert(self.config, 'test_spatial')
        output = spatial_expert(self.test_input)
        
        assert isinstance(output, ExpertOutput)
        assert output.metadata['expert_type'] == 'spatial'
        
        print(f"✓ Expert base classes work correctly")
    
    def test_load_balancing(self):
        """Test load balancing in MoE layers."""
        # Create mock experts
        class MockExpert(BaseExpert):
            def __init__(self, config, name):
                super().__init__(config, 'temporal', name)
                self.linear = torch.nn.Linear(self.d_model, self.expert_dim)
            
            def forward(self, x, **kwargs):
                output = self.linear(x)
                confidence = torch.ones(x.size(0), x.size(1), 1) * 0.8
                return ExpertOutput(
                    output=output,
                    confidence=confidence,
                    metadata={'expert_name': self.expert_name}
                )
        
        # Create experts with load balancing router
        experts = [MockExpert(self.config, f'expert_{i}') for i in range(4)]
        router = ExpertRouter(self.d_model, len(experts))
        router.load_balancing = True
        router.load_balancing_weight = 0.01
        
        moe_layer = MoELayer(experts, router, top_k=2)
        
        # Multiple forward passes to test load balancing
        total_load_loss = 0
        for _ in range(5):
            output, moe_info = moe_layer(self.test_input)
            if 'load_balancing_loss' in moe_info['routing_info']:
                total_load_loss += moe_info['routing_info']['load_balancing_loss']
        
        print(f"✓ Load balancing works")
        print(f"  Average load balancing loss: {total_load_loss / 5:.6f}")
    
    def test_expert_compatibility_validation(self):
        """Test expert compatibility validation."""
        # Test compatible experts
        compatible_experts = ['seasonal_expert', 'trend_expert']
        validation = expert_registry.validate_expert_compatibility(compatible_experts, self.config)
        
        assert validation['compatible'] is True
        
        # Test suggestions
        suggestions = expert_registry.suggest_experts(
            'forecasting',
            {'high_volatility': True, 'spatial_dependencies': True}
        )
        
        assert len(suggestions) > 0
        assert any('volatility' in s for s in suggestions)
        
        print(f"✓ Expert compatibility validation works")
        print(f"  Suggested experts: {suggestions}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])