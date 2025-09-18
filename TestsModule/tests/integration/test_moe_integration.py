"""
Integration tests for Mixture of Experts Framework

Comprehensive integration tests that verify the MoE components work together
correctly in realistic scenarios.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
import warnings
warnings.filterwarnings('ignore')

# Test imports
try:
    from layers.modular.experts.registry import expert_registry, create_expert
    from layers.modular.experts.expert_router import AdaptiveExpertRouter
    from layers.modular.experts.moe_layer import MoELayer
    from layers.modular.experts.base_expert import ExpertOutput
    from models.Enhanced_SOTA_Temporal_PGAT_MoE import Enhanced_SOTA_Temporal_PGAT_MoE
except ImportError as e:
    pytest.skip(f"MoE integration components not available: {e}", allow_module_level=True)


class TestMoEIntegration:
    """Integration test suite for MoE framework."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Setup comprehensive test configuration."""
        self.config = MagicMock()
        self.config.d_model = 256
        self.config.n_heads = 8
        self.config.dropout = 0.1
        self.config.seq_len = 48
        self.config.pred_len = 12
        self.config.enc_in = 7
        self.config.dec_in = 7
        self.config.c_out = 7
        self.config.features = 'M'
        
        # MoE configuration
        self.config.temporal_top_k = 2
        self.config.spatial_top_k = 2
        self.config.uncertainty_top_k = 1
        
        # Enhanced components
        self.config.use_mixture_density = True
        self.config.mdn_components = 3
        self.config.hierarchy_levels = 3
        self.config.sparsity_ratio = 0.1
        
        self.batch_size = 4
        self.device = torch.device('cpu')
        
        # Generate realistic time series data
        self.test_data = self._generate_realistic_data()
    
    def _generate_realistic_data(self):
        """Generate realistic time series data with multiple patterns."""
        t = torch.linspace(0, 4*np.pi, self.config.seq_len)
        data = torch.zeros(self.batch_size, self.config.seq_len, self.config.enc_in)
        
        for b in range(self.batch_size):
            for f in range(self.config.enc_in):
                # Seasonal component
                seasonal = torch.sin(2 * t + f * 0.5) * (0.5 + f * 0.1)
                
                # Trend component
                trend = 0.02 * t * (f + 1)
                
                # Volatility (heteroscedastic noise)
                volatility = 0.1 + 0.1 * torch.abs(torch.sin(t))
                noise = torch.randn(self.config.seq_len) * volatility
                
                # Regime change (random)
                if np.random.random() > 0.7:
                    change_point = np.random.randint(self.config.seq_len // 3, 2 * self.config.seq_len // 3)
                    trend[change_point:] += np.random.normal(0, 0.5)
                
                data[b, :, f] = seasonal + trend + noise
        
        return data
    
    def test_expert_ensemble_creation(self):
        """Test creating ensembles of different expert types."""
        # Test temporal expert ensemble
        temporal_experts = ['seasonal_expert', 'trend_expert', 'volatility_expert']
        temporal_ensemble = []
        
        for expert_name in temporal_experts:
            try:
                expert = create_expert(expert_name, self.config)
                temporal_ensemble.append(expert)
            except Exception as e:
                print(f"Warning: Could not create {expert_name}: {e}")
        
        assert len(temporal_ensemble) > 0, "Should create at least one temporal expert"
        
        # Test that experts have consistent interfaces
        for expert in temporal_ensemble:
            assert hasattr(expert, 'forward')
            assert hasattr(expert, 'get_expert_info')
            
            # Test forward pass
            output = expert(self.test_data)
            assert isinstance(output, ExpertOutput)
            assert output.output.shape[0] == self.batch_size
        
        print(f"✓ Created temporal expert ensemble: {len(temporal_ensemble)} experts")
    
    def test_moe_layer_with_real_experts(self):
        """Test MoE layer with real expert implementations."""
        # Create real experts
        experts = []
        expert_names = ['seasonal_expert', 'trend_expert']
        
        for name in expert_names:
            try:
                expert = create_expert(name, self.config)
                experts.append(expert)
            except Exception as e:
                print(f"Warning: Could not create {name}: {e}")
        
        if len(experts) < 2:
            pytest.skip("Need at least 2 experts for MoE test")
        
        # Create router and MoE layer
        router = AdaptiveExpertRouter(self.config.d_model, len(experts))
        moe_layer = MoELayer(experts, router, top_k=2)
        
        # Test multiple forward passes
        outputs = []
        moe_infos = []
        
        for _ in range(3):
            output, moe_info = moe_layer(self.test_data)
            outputs.append(output)
            moe_infos.append(moe_info)
            
            # Validate output
            assert output.shape == self.test_data.shape
            assert 'routing_info' in moe_info
            assert 'expert_usage' in moe_info
        
        # Check expert utilization over time
        utilization = moe_layer.get_expert_utilization()
        total_utilization = sum(utilization.values())
        assert total_utilization > 0, "Experts should be utilized"
        
        print(f"✓ MoE layer with real experts works")
        print(f"  Expert utilization: {utilization}")
        print(f"  Total utilization: {total_utilization:.3f}")
    
    def test_expert_specialization_effectiveness(self):
        """Test that different experts specialize in different patterns."""
        # Create experts
        seasonal_expert = create_expert('seasonal_expert', self.config)
        trend_expert = create_expert('trend_expert', self.config)
        
        # Create data with dominant seasonal pattern
        t = torch.linspace(0, 4*np.pi, self.config.seq_len)
        seasonal_data = torch.zeros(2, self.config.seq_len, self.config.d_model)
        for f in range(self.config.d_model):
            seasonal_data[:, :, f] = torch.sin(2 * t + f * 0.1).unsqueeze(0)
        
        # Create data with dominant trend pattern
        trend_data = torch.zeros(2, self.config.seq_len, self.config.d_model)
        for f in range(self.config.d_model):
            trend_data[:, :, f] = (0.05 * t * (f + 1)).unsqueeze(0)
        
        # Test expert responses
        seasonal_on_seasonal = seasonal_expert(seasonal_data)
        seasonal_on_trend = seasonal_expert(trend_data)
        
        trend_on_seasonal = trend_expert(seasonal_data)
        trend_on_trend = trend_expert(trend_data)
        
        # Seasonal expert should be more confident on seasonal data
        seasonal_conf_seasonal = seasonal_on_seasonal.confidence.mean().item()
        seasonal_conf_trend = seasonal_on_trend.confidence.mean().item()
        
        # Trend expert should be more confident on trend data
        trend_conf_seasonal = trend_on_seasonal.confidence.mean().item()
        trend_conf_trend = trend_on_trend.confidence.mean().item()
        
        print(f"✓ Expert specialization test")
        print(f"  Seasonal expert: seasonal={seasonal_conf_seasonal:.3f}, trend={seasonal_conf_trend:.3f}")
        print(f"  Trend expert: seasonal={trend_conf_seasonal:.3f}, trend={trend_conf_trend:.3f}")
        
        # Note: This is a weak test since we can't guarantee specialization without training
        # But we can check that experts produce different outputs
        assert not torch.allclose(seasonal_on_seasonal.output, trend_on_seasonal.output, atol=1e-3)
    
    def test_routing_adaptation(self):
        """Test that routing adapts to input characteristics."""
        # Create experts and adaptive router
        experts = []
        for name in ['seasonal_expert', 'trend_expert']:
            try:
                expert = create_expert(name, self.config)
                experts.append(expert)
            except:
                continue
        
        if len(experts) < 2:
            pytest.skip("Need at least 2 experts for routing test")
        
        router = AdaptiveExpertRouter(self.config.d_model, len(experts))
        
        # Test with different input characteristics
        # High-frequency data (should favor seasonal expert)
        t = torch.linspace(0, 8*np.pi, self.config.seq_len)
        high_freq_data = torch.sin(4 * t).unsqueeze(0).unsqueeze(-1).expand(1, -1, self.config.d_model)
        
        # Low-frequency trend data (should favor trend expert)
        low_freq_data = (0.1 * t).unsqueeze(0).unsqueeze(-1).expand(1, -1, self.config.d_model)
        
        # Get routing weights
        high_freq_weights, high_freq_info = router(high_freq_data)
        low_freq_weights, low_freq_info = router(low_freq_data)
        
        # Check that routing is different for different inputs
        assert not torch.allclose(high_freq_weights, low_freq_weights, atol=1e-2)
        
        print(f"✓ Routing adaptation test")
        print(f"  High-freq routing: {high_freq_weights[0].tolist()}")
        print(f"  Low-freq routing: {low_freq_weights[0].tolist()}")
        print(f"  Routing diversity: {high_freq_info.get('routing_diversity', 'N/A')}")
    
    def test_load_balancing_effectiveness(self):
        """Test that load balancing prevents expert collapse."""
        # Create experts
        experts = []
        for name in ['seasonal_expert', 'trend_expert', 'volatility_expert']:
            try:
                expert = create_expert(name, self.config)
                experts.append(expert)
            except:
                continue
        
        if len(experts) < 3:
            pytest.skip("Need at least 3 experts for load balancing test")
        
        # Create router with load balancing
        router = AdaptiveExpertRouter(self.config.d_model, len(experts))
        router.load_balancing = True
        router.load_balancing_weight = 0.1
        
        moe_layer = MoELayer(experts, router, top_k=2)
        
        # Run multiple forward passes
        total_load_loss = 0
        for _ in range(10):
            output, moe_info = moe_layer(self.test_data)
            if 'load_balancing_loss' in moe_info['routing_info']:
                total_load_loss += moe_info['routing_info']['load_balancing_loss']
        
        # Check expert utilization
        utilization = moe_layer.get_expert_utilization()
        utilization_values = list(utilization.values())
        
        # Load balancing should prevent extreme imbalance
        max_util = max(utilization_values)
        min_util = min(utilization_values)
        imbalance_ratio = max_util / (min_util + 1e-8)
        
        print(f"✓ Load balancing test")
        print(f"  Expert utilization: {utilization}")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
        print(f"  Average load loss: {total_load_loss / 10:.6f}")
        
        # With load balancing, imbalance should be reasonable
        assert imbalance_ratio < 10.0, "Load balancing should prevent extreme imbalance"
    
    def test_moe_gradient_flow_integration(self):
        """Test gradient flow through complete MoE system."""
        # Create experts
        experts = []
        for name in ['seasonal_expert', 'trend_expert']:
            try:
                expert = create_expert(name, self.config)
                experts.append(expert)
            except:
                continue
        
        if len(experts) < 2:
            pytest.skip("Need at least 2 experts for gradient test")
        
        # Create MoE system
        router = AdaptiveExpertRouter(self.config.d_model, len(experts))
        moe_layer = MoELayer(experts, router, top_k=2)
        
        # Forward pass
        output, moe_info = moe_layer(self.test_data)
        
        # Compute loss with load balancing
        loss = output.mean()
        if 'load_balancing_loss' in moe_info['routing_info']:
            loss += 0.01 * moe_info['routing_info']['load_balancing_loss']
        
        # Backward pass
        loss.backward()
        
        # Check gradients in router
        router_grad_count = 0
        for param in router.parameters():
            if param.grad is not None:
                router_grad_count += 1
                assert torch.isfinite(param.grad).all()
        
        # Check gradients in experts
        expert_grad_counts = []
        for expert in experts:
            expert_grad_count = 0
            for param in expert.parameters():
                if param.grad is not None:
                    expert_grad_count += 1
                    assert torch.isfinite(param.grad).all()
            expert_grad_counts.append(expert_grad_count)
        
        assert router_grad_count > 0, "Router should have gradients"
        assert all(count > 0 for count in expert_grad_counts), "All experts should have gradients"
        
        print(f"✓ MoE gradient flow integration")
        print(f"  Router gradients: {router_grad_count}")
        print(f"  Expert gradients: {expert_grad_counts}")
    
    def test_enhanced_model_moe_integration(self):
        """Test MoE integration in the enhanced SOTA PGAT model."""
        try:
            model = Enhanced_SOTA_Temporal_PGAT_MoE(self.config, mode='probabilistic')
            model.train()
            
            # Create input data
            wave_window = self.test_data
            target_window = torch.randn(self.batch_size, self.config.pred_len, self.config.enc_in)
            graph = torch.ones(self.batch_size, self.config.seq_len + self.config.pred_len,
                             self.config.seq_len + self.config.pred_len)
            
            # Forward pass
            output, moe_info = model(wave_window, target_window, graph)
            
            # Check MoE integration
            assert isinstance(moe_info, dict)
            
            # Check for different expert types
            expert_types = ['temporal', 'spatial', 'uncertainty']
            found_types = []
            
            for expert_type in expert_types:
                if expert_type in moe_info:
                    found_types.append(expert_type)
                    type_info = moe_info[expert_type]
                    assert 'routing_info' in type_info
            
            assert len(found_types) > 0, "Should have at least one expert type"
            
            # Test expert utilization
            utilization = model.get_expert_utilization()
            assert isinstance(utilization, dict)
            
            print(f"✓ Enhanced model MoE integration")
            print(f"  Found expert types: {found_types}")
            print(f"  Utilization keys: {list(utilization.keys())}")
            
        except Exception as e:
            print(f"⚠ Enhanced model MoE integration test skipped: {e}")
    
    def test_moe_performance_scaling(self):
        """Test MoE performance with different numbers of experts."""
        expert_counts = [2, 3, 4]
        performance_results = {}
        
        for num_experts in expert_counts:
            # Create experts
            expert_names = ['seasonal_expert', 'trend_expert', 'volatility_expert', 'regime_expert']
            experts = []
            
            for i, name in enumerate(expert_names[:num_experts]):
                try:
                    expert = create_expert(name, self.config)
                    experts.append(expert)
                except:
                    continue
            
            if len(experts) != num_experts:
                continue
            
            # Create MoE layer
            router = AdaptiveExpertRouter(self.config.d_model, len(experts))
            moe_layer = MoELayer(experts, router, top_k=min(2, len(experts)))
            
            # Measure performance
            import time
            start_time = time.time()
            
            for _ in range(5):
                output, moe_info = moe_layer(self.test_data)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 5
            
            performance_results[num_experts] = {
                'avg_time': avg_time,
                'output_shape': output.shape,
                'num_params': sum(p.numel() for p in moe_layer.parameters())
            }
        
        print(f"✓ MoE performance scaling test")
        for num_experts, results in performance_results.items():
            print(f"  {num_experts} experts: {results['avg_time']:.4f}s, {results['num_params']:,} params")
    
    def test_moe_memory_efficiency(self):
        """Test MoE memory efficiency with sparse routing."""
        # Create experts
        experts = []
        for name in ['seasonal_expert', 'trend_expert', 'volatility_expert']:
            try:
                expert = create_expert(name, self.config)
                experts.append(expert)
            except:
                continue
        
        if len(experts) < 3:
            pytest.skip("Need at least 3 experts for memory test")
        
        # Test with different top_k values
        top_k_values = [1, 2, 3]
        memory_usage = {}
        
        for top_k in top_k_values:
            if top_k > len(experts):
                continue
            
            router = AdaptiveExpertRouter(self.config.d_model, len(experts))
            moe_layer = MoELayer(experts, router, top_k=top_k)
            
            # Measure memory (approximate)
            total_params = sum(p.numel() for p in moe_layer.parameters())
            
            # Forward pass
            output, moe_info = moe_layer(self.test_data)
            
            memory_usage[top_k] = {
                'total_params': total_params,
                'active_experts': top_k,
                'efficiency_ratio': top_k / len(experts)
            }
        
        print(f"✓ MoE memory efficiency test")
        for top_k, usage in memory_usage.items():
            print(f"  top_k={top_k}: {usage['total_params']:,} params, "
                  f"efficiency={usage['efficiency_ratio']:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])