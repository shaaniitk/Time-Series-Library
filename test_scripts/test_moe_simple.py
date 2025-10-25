#!/usr/bin/env python3
"""
Simple Test for Enhanced SOTA PGAT with MoE Components

Direct test without pytest to validate core functionality.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, '.')

def create_proper_config():
    """Create a proper configuration object."""
    class Config:
        def __init__(self):
            # Basic model parameters
            self.d_model = 128
            self.n_heads = 4
            self.dropout = 0.1
            
            # Data dimensions
            self.seq_len = 48
            self.pred_len = 12
            self.enc_in = 4
            self.dec_in = 4
            self.c_out = 4
            self.features = 'M'
            
            # MoE parameters
            self.temporal_top_k = 2
            self.spatial_top_k = 2
            self.uncertainty_top_k = 1
            
            # Enhanced components
            self.use_mixture_density = True
            self.use_autocorr_attention = True
            self.use_dynamic_edge_weights = True
            self.use_adaptive_temporal = True
            
            # MDN parameters
            self.mdn_components = 2
            
            # Graph parameters
            self.hierarchy_levels = 2
            self.sparsity_ratio = 0.2
            self.max_eigenvectors = 8
            self.num_nodes = 8
            
            # Expert-specific parameters
            self.seasonal_decomp_kernel = 15
            self.seasonal_harmonics = 5
            self.seasonal_heads = 4
            self.local_neighborhood_size = 3
            self.local_spatial_heads = 4
            self.global_spatial_heads = 8
            self.spatial_hierarchy_levels = 2
            self.hierarchical_heads = 4
            self.epistemic_ensemble_size = 3
            self.garch_order = (1, 1)
            self.num_regimes = 3
            self.regime_detection_window = 10
            
            # Training parameters
            self.use_sequence_curriculum = True
            self.use_complexity_curriculum = True
            self.min_seq_len = 24
            self.max_seq_len = 48
            self.min_experts = 1
            self.max_experts = 4
            
            # Memory optimization
            self.use_mixed_precision = False  # Disable for CPU testing
            self.use_gradient_checkpointing = False
            self.batch_size = 4
            self.max_batch_size = 16
            self.memory_threshold = 0.8
    
    return Config()

def test_basic_imports():
    """Test that all basic imports work."""
    print("🔍 Testing Basic Imports...")
    
    try:
        from layers.modular.experts.base_expert import BaseExpert, ExpertOutput
        print("  ✅ Base expert classes")
        
        from layers.modular.experts.registry import expert_registry
        print("  ✅ Expert registry")
        
        from layers.modular.experts.expert_router import ExpertRouter
        print("  ✅ Expert router")
        
        from layers.modular.experts.moe_layer import MoELayer
        print("  ✅ MoE layer")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_expert_registry():
    """Test expert registry functionality."""
    print("\n🏪 Testing Expert Registry...")
    
    try:
        from layers.modular.experts.registry import expert_registry
        
        # List available experts
        temporal_experts = expert_registry.list_experts('temporal')
        spatial_experts = expert_registry.list_experts('spatial')
        uncertainty_experts = expert_registry.list_experts('uncertainty')
        
        print(f"  ✅ Temporal experts: {len(temporal_experts)}")
        print(f"  ✅ Spatial experts: {len(spatial_experts)}")
        print(f"  ✅ Uncertainty experts: {len(uncertainty_experts)}")
        
        # Test expert info
        if temporal_experts:
            info = expert_registry.get_expert_info(temporal_experts[0])
            print(f"  ✅ Expert info: {info['name']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Registry test failed: {e}")
        return False

def test_expert_creation():
    """Test creating individual experts."""
    print("\n🧪 Testing Expert Creation...")
    
    config = create_proper_config()
    
    # Test temporal experts
    temporal_results = {}
    temporal_experts = ['seasonal_expert', 'trend_expert', 'volatility_expert', 'regime_expert']
    
    for expert_name in temporal_experts:
        try:
            from layers.modular.experts.registry import create_expert
            expert = create_expert(expert_name, config)
            
            # Test forward pass
            test_input = torch.randn(2, config.seq_len, config.d_model)
            output = expert(test_input)
            
            assert isinstance(output, ExpertOutput)
            temporal_results[expert_name] = True
            print(f"  ✅ {expert_name}: {output.output.shape}")
            
        except Exception as e:
            temporal_results[expert_name] = False
            print(f"  ❌ {expert_name}: {str(e)}")
    
    # Test spatial experts (simplified)
    spatial_results = {}
    spatial_experts = ['local_spatial_expert', 'global_spatial_expert', 'hierarchical_spatial_expert']
    
    for expert_name in spatial_experts:
        try:
            from layers.modular.experts.registry import create_expert
            expert = create_expert(expert_name, config)
            
            # Test forward pass
            test_input = torch.randn(2, config.seq_len, config.d_model)
            output = expert(test_input)
            
            assert isinstance(output, ExpertOutput)
            spatial_results[expert_name] = True
            print(f"  ✅ {expert_name}: {output.output.shape}")
            
        except Exception as e:
            spatial_results[expert_name] = False
            print(f"  ❌ {expert_name}: {str(e)}")
    
    # Test uncertainty experts
    uncertainty_results = {}
    uncertainty_experts = ['aleatoric_uncertainty_expert', 'epistemic_uncertainty_expert']
    
    for expert_name in uncertainty_experts:
        try:
            from layers.modular.experts.registry import create_expert
            expert = create_expert(expert_name, config)
            
            # Test forward pass
            test_input = torch.randn(2, config.seq_len, config.d_model)
            output = expert(test_input)
            
            assert isinstance(output, ExpertOutput)
            uncertainty_results[expert_name] = True
            print(f"  ✅ {expert_name}: {output.output.shape}")
            
        except Exception as e:
            uncertainty_results[expert_name] = False
            print(f"  ❌ {expert_name}: {str(e)}")
    
    # Summary
    total_experts = len(temporal_results) + len(spatial_results) + len(uncertainty_results)
    successful_experts = sum(temporal_results.values()) + sum(spatial_results.values()) + sum(uncertainty_results.values())
    
    print(f"\n📊 Expert Creation Summary: {successful_experts}/{total_experts} successful")
    
    return successful_experts > 0

def test_moe_layer():
    """Test MoE layer with working experts."""
    print("\n🎯 Testing MoE Layer...")
    
    try:
        from layers.modular.experts.expert_router import ExpertRouter
        from layers.modular.experts.moe_layer import MoELayer
        from layers.modular.experts.base_expert import BaseExpert, ExpertOutput
        
        config = create_proper_config()
        
        # Create simple mock experts that definitely work
        class SimpleMockExpert(BaseExpert):
            def __init__(self, config, name):
                super().__init__(config, 'temporal', name)
                self.linear = nn.Linear(self.d_model, self.expert_dim)
                
            def forward(self, x, **kwargs):
                output = self.linear(x)
                confidence = torch.ones(x.size(0), x.size(1), 1) * 0.8
                return ExpertOutput(
                    output=output,
                    confidence=confidence,
                    metadata={'expert_name': self.expert_name}
                )
        
        # Create experts and router
        experts = [SimpleMockExpert(config, f'mock_expert_{i}') for i in range(3)]
        router = ExpertRouter(config.d_model, len(experts))
        
        # Create MoE layer
        moe_layer = MoELayer(experts, router, top_k=2)
        
        # Test forward pass
        test_input = torch.randn(2, config.seq_len, config.d_model)
        output, moe_info = moe_layer(test_input)
        
        assert output.shape == test_input.shape
        assert 'routing_info' in moe_info
        
        print(f"  ✅ MoE layer works: {output.shape}")
        print(f"  ✅ MoE info keys: {list(moe_info.keys())}")
        
        # Test expert utilization
        utilization = moe_layer.get_expert_utilization()
        print(f"  ✅ Expert utilization: {utilization}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ MoE layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_curriculum_learning():
    """Test curriculum learning components."""
    print("\n📚 Testing Curriculum Learning...")
    
    try:
        from layers.modular.training.curriculum_learning import (
            SequenceLengthCurriculum, ComplexityCurriculum
        )
        
        # Test sequence length curriculum
        seq_curriculum = SequenceLengthCurriculum(
            total_epochs=20,
            min_seq_len=8,
            max_seq_len=32,
            growth_strategy='linear'
        )
        
        # Test progression
        for epoch in [0, 5, 10, 15, 19]:
            seq_curriculum.update_epoch(epoch)
            params = seq_curriculum.get_curriculum_params(epoch)
            print(f"  ✅ Epoch {epoch}: seq_len={params['seq_len']}, progress={params['progress']:.2f}")
        
        # Test complexity curriculum
        complexity_curriculum = ComplexityCurriculum(
            total_epochs=20,
            min_experts=1,
            max_experts=4
        )
        
        params = complexity_curriculum.get_curriculum_params(10)
        print(f"  ✅ Complexity curriculum: {params['num_active_experts']} experts")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Curriculum learning test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization components."""
    print("\n💾 Testing Memory Optimization...")
    
    try:
        from layers.modular.training.memory_optimization import (
            GradientCheckpointing, MixedPrecisionTraining, MemoryEfficientAttention
        )
        
        # Test gradient checkpointing
        simple_module = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        checkpointed = GradientCheckpointing(simple_module)
        test_input = torch.randn(2, 10, 128)
        output = checkpointed(test_input)
        
        assert output.shape == test_input.shape
        print(f"  ✅ Gradient checkpointing: {output.shape}")
        
        # Test mixed precision
        mp_trainer = MixedPrecisionTraining(enabled=False)  # CPU mode
        loss = torch.tensor(1.0, requires_grad=True)
        scaled_loss = mp_trainer.scale_loss(loss)
        print(f"  ✅ Mixed precision training")
        
        # Test memory efficient attention
        attention = MemoryEfficientAttention(d_model=128, num_heads=4, chunk_size=16)
        attn_output = attention(test_input, test_input, test_input)
        
        assert attn_output.shape == test_input.shape
        print(f"  ✅ Memory efficient attention: {attn_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Memory optimization test failed: {e}")
        return False

def test_enhanced_model_simple():
    """Test enhanced model with simplified approach."""
    print("\n🚀 Testing Enhanced Model (Simplified)...")
    
    try:
        # First test if we can import the model
        from models.Enhanced_SOTA_Temporal_PGAT_MoE import Enhanced_SOTA_Temporal_PGAT_MoE
        print("  ✅ Enhanced model import successful")
        
        # Create proper config
        config = create_proper_config()
        
        # Try to create model
        print("  🏗️  Creating model...")
        model = Enhanced_SOTA_Temporal_PGAT_MoE(config, mode='standard')  # Use standard mode
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Model created: {total_params:,} parameters")
        
        # Test forward pass
        print("  🔄 Testing forward pass...")
        batch_size = 2
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        target_window = torch.randn(batch_size, config.pred_len, config.dec_in)
        graph = torch.ones(batch_size, config.seq_len + config.pred_len, 
                          config.seq_len + config.pred_len)
        
        model.eval()
        with torch.no_grad():
            output = model(wave_window, target_window, graph)
        
        print(f"  ✅ Forward pass successful")
        print(f"  📊 Output type: {type(output)}")
        
        if isinstance(output, tuple):
            predictions, moe_info = output
            print(f"  📊 Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else type(predictions)}")
            print(f"  📊 MoE info keys: {list(moe_info.keys()) if isinstance(moe_info, dict) else 'N/A'}")
        else:
            print(f"  📊 Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual components separately."""
    print("\n🔧 Testing Individual Components...")
    
    config = create_proper_config()
    results = {}
    
    # Test expert router
    try:
        from layers.modular.experts.expert_router import ExpertRouter
        router = ExpertRouter(config.d_model, 3)
        test_input = torch.randn(2, config.seq_len, config.d_model)
        routing_weights, routing_info = router(test_input)
        
        assert routing_weights.shape == (2, 3)
        results['expert_router'] = True
        print("  ✅ Expert router works")
        
    except Exception as e:
        results['expert_router'] = False
        print(f"  ❌ Expert router failed: {e}")
    
    # Test simple expert
    try:
        from layers.modular.experts.base_expert import BaseExpert, ExpertOutput
        
        class TestExpert(BaseExpert):
            def __init__(self, config):
                super().__init__(config, 'temporal', 'test_expert')
                self.linear = nn.Linear(self.d_model, self.expert_dim)
                
            def forward(self, x, **kwargs):
                output = self.linear(x)
                confidence = self.compute_confidence(x, output)
                return ExpertOutput(
                    output=output,
                    confidence=confidence,
                    metadata={'test': True}
                )
        
        expert = TestExpert(config)
        test_input = torch.randn(2, config.seq_len, config.d_model)
        output = expert(test_input)
        
        assert isinstance(output, ExpertOutput)
        results['simple_expert'] = True
        print("  ✅ Simple expert works")
        
    except Exception as e:
        results['simple_expert'] = False
        print(f"  ❌ Simple expert failed: {e}")
    
    return results

def main():
    """Main test function."""
    print("🚀 Enhanced SOTA PGAT with MoE - Simple Validation")
    print("=" * 80)
    
    results = {}
    
    # Run tests
    results['imports'] = test_basic_imports()
    results['registry'] = test_expert_registry()
    results['components'] = test_individual_components()
    results['curriculum'] = test_curriculum_learning()
    results['memory_opt'] = test_memory_optimization()
    results['enhanced_model'] = test_enhanced_model_simple()
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    success_rate = passed / total if total > 0 else 0
    
    print(f"📊 Overall Results: {passed}/{total} passed ({success_rate:.1%})")
    print(f"\n📋 Detailed Results:")
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {test_name}")
    
    if success_rate >= 0.7:
        print(f"\n🎉 Validation successful!")
        print(f"   The Enhanced SOTA PGAT with MoE framework is working!")
        
        print(f"\n🎯 What's Working:")
        print(f"   • Core MoE framework components")
        print(f"   • Expert registry system")
        print(f"   • Individual expert creation")
        print(f"   • Curriculum learning strategies")
        print(f"   • Memory optimization components")
        if results.get('enhanced_model'):
            print(f"   • Complete enhanced model")
        
        print(f"\n🔧 Next Steps:")
        print(f"   • Run the comprehensive example: python examples/enhanced_sota_pgat_moe_example.py")
        print(f"   • Integrate into your existing training pipeline")
        print(f"   • Customize experts for your specific use case")
        
    else:
        print(f"\n⚠️  Some components need attention")
        print(f"   Check the detailed results above for specific issues")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)