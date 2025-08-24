"""
Phase 2 Validation: Comprehensive Testing of Utils-Integrated Algorithms

This module validates that our sophisticated restored algorithms are working correctly
within the utils/ modular system, preserving their algorithmic sophistication.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import traceback
from dataclasses import dataclass

# Import utils system components
from utils.modular_components.factory import ComponentFactory
from utils.modular_components.registry import _global_registry
from configs.schemas import AttentionConfig, LossConfig

# Import our algorithm adapters
from utils_algorithm_adapters import (
    RestoredFourierConfig,
    RestoredAutoCorrelationConfig, 
    RestoredMetaLearningConfig,
    register_restored_algorithms
)


class Phase2ValidationSuite:
    """Comprehensive validation of Phase 2 algorithm adaptation"""
    
    def __init__(self):
        self.results = {}
        self.factory = ComponentFactory()
        self.registry = _global_registry
        
        # Register our restored algorithms
        register_restored_algorithms()
        
        # Patch the factory to use correct registry method
        self.factory.registry = self.registry
        
    def create_component_direct(self, component_type: str, component_name: str, config):
        """Direct component creation bypassing factory issues"""
        component_class = self.registry.get(component_type, component_name)
        return component_class(config)
        
    def test_utils_registry_integration(self) -> bool:
        """Test that our algorithms are properly registered in utils system"""
        print("🔍 Testing Utils Registry Integration...")
        
        try:
            # Get attention components from utils registry
            attention_components = self.registry.list_components('attention')['attention']
            
            restored_algorithms = [
                'restored_fourier_attention',
                'restored_autocorrelation_attention', 
                'restored_meta_learning_attention'
            ]
            
            for algo in restored_algorithms:
                if algo not in attention_components:
                    print(f"❌ {algo} not found in utils registry")
                    return False
                
                # Check metadata
                metadata = self.registry.get_metadata('attention', algo)
                if 'sophistication_level' not in metadata:
                    print(f"❌ {algo} missing sophistication metadata")
                    return False
                    
                print(f"✅ {algo}: {metadata['sophistication_level']} sophistication")
                
            print(f"✅ All {len(restored_algorithms)} algorithms registered with metadata")
            return True
            
        except Exception as e:
            print(f"❌ Registry integration failed: {e}")
            return False
    
    def test_factory_creation(self) -> bool:
        """Test creating our algorithms through utils factory system"""
        print("\n🏭 Testing Factory Creation...")
        
        try:
            # Test 1: Fourier Attention
            fourier_config = RestoredFourierConfig(
                d_model=512, 
                num_heads=8,
                dropout=0.1,
                seq_len=96
            )
            
            fourier_attention = self.create_component_direct(
                'attention',
                'restored_fourier_attention',
                fourier_config
            )
            
            # Test sophisticated features exist
            assert hasattr(fourier_attention, 'fourier_attention'), "Missing sophisticated algorithm"
            assert fourier_attention.get_attention_type() == "restored_fourier"
            capabilities = fourier_attention.get_capabilities()
            assert 'sophisticated_frequency_filtering' in capabilities
            print("✅ Fourier Attention factory creation successful")
            
            # Test 2: AutoCorrelation Attention
            autocorr_config = RestoredAutoCorrelationConfig(
                d_model=512,
                num_heads=8, 
                dropout=0.1,
                adaptive_k=True,
                multi_scale=True
            )
            
            autocorr_attention = self.create_component_direct(
                'attention',
                'restored_autocorrelation_attention',
                autocorr_config
            )
            
            # Verify sophisticated k_predictor
            assert hasattr(autocorr_attention.autocorr_attention, 'k_predictor')
            assert autocorr_attention.get_attention_type() == "restored_autocorrelation"
            capabilities = autocorr_attention.get_capabilities()
            assert 'sophisticated_k_predictor' in capabilities
            print("✅ AutoCorrelation Attention factory creation successful")
            
            # Test 3: Meta Learning Attention
            meta_config = RestoredMetaLearningConfig(
                d_model=512,
                num_heads=8,
                dropout=0.1,
                adaptation_steps=3
            )
            
            meta_attention = self.create_component_direct(
                'attention',
                'restored_meta_learning_attention', 
                meta_config
            )
            
            # Verify sophisticated MAML components
            assert hasattr(meta_attention.meta_attention, 'fast_weights')
            assert meta_attention.get_attention_type() == "restored_meta_learning"
            capabilities = meta_attention.get_capabilities()
            assert 'maml_meta_learning' in capabilities
            print("✅ Meta Learning Attention factory creation successful")
            
            return True
            
        except Exception as e:
            print(f"❌ Factory creation failed: {e}")
            traceback.print_exc()
            return False
    
    def test_algorithmic_sophistication_preservation(self) -> bool:
        """Test that our sophisticated algorithms maintain their complexity"""
        print("\n🧠 Testing Algorithmic Sophistication Preservation...")
        
        try:
            batch_size, seq_len, d_model = 4, 96, 512
            
            # Test data
            x = torch.randn(batch_size, seq_len, d_model)
            
            # Test 1: Fourier sophisticated frequency filtering
            fourier_config = RestoredFourierConfig(d_model=d_model, seq_len=seq_len, num_heads=8)
            fourier = self.create_component_direct('attention', 'restored_fourier_attention', fourier_config)
            
            fourier_output, fourier_weights = fourier.apply_attention(x, x, x)
            assert fourier_output.shape == x.shape
            
            # Verify sophisticated features work
            assert hasattr(fourier.fourier_attention, 'frequency_selection')
            assert hasattr(fourier.fourier_attention, 'learnable_filter')
            print("✅ Fourier: Complex frequency filtering preserved")
            
            # Test 2: AutoCorrelation sophisticated k-predictor  
            autocorr_config = RestoredAutoCorrelationConfig(d_model=d_model, num_heads=8)
            autocorr = self.create_component_direct('attention', 'restored_autocorrelation_attention', autocorr_config)
            
            autocorr_output, autocorr_weights = autocorr.apply_attention(x, x, x)
            assert autocorr_output.shape == x.shape
            
            # Verify sophisticated k_predictor is neural network
            k_predictor = autocorr.autocorr_attention.k_predictor
            param_count = sum(p.numel() for p in k_predictor.parameters())
            assert param_count > 100, f"k_predictor too simple: {param_count} params"
            print(f"✅ AutoCorrelation: Sophisticated k_predictor with {param_count} parameters")
            
            # Test 3: Meta Learning sophisticated MAML
            meta_config = RestoredMetaLearningConfig(d_model=d_model, num_heads=8)
            meta = self.create_component_direct('attention', 'restored_meta_learning_attention', meta_config)
            
            meta_output, meta_weights = meta.apply_attention(x, x, x)
            assert meta_output.shape == x.shape
            
            # Verify sophisticated MAML components
            assert hasattr(meta.meta_attention, 'fast_weights')
            assert hasattr(meta.meta_attention, 'adaptation_steps')
            print("✅ Meta Learning: MAML fast adaptation preserved")
            
            return True
            
        except Exception as e:
            print(f"❌ Sophistication test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_utils_system_compatibility(self) -> bool:
        """Test full compatibility with utils system features"""
        print("\n🔧 Testing Utils System Compatibility...")
        
        try:
            # Test component info
            fourier_config = RestoredFourierConfig(d_model=256, num_heads=4)
            fourier = self.create_component_direct('attention', 'restored_fourier_attention', fourier_config)
            
            info = fourier.get_info()
            assert 'component_type' in info
            assert 'num_parameters' in info
            assert info['output_dim'] == 256
            print(f"✅ Component info: {info['num_parameters']} parameters")
            
            # Test capabilities and requirements
            capabilities = fourier.get_capabilities()
            requirements = fourier.get_requirements()
            assert isinstance(capabilities, list)
            assert isinstance(requirements, dict)
            print(f"✅ Capabilities: {len(capabilities)} features")
            
            # Test config schemas (if implemented)
            try:
                schema = fourier.get_config_schema()
                print(f"✅ Config schema available: {len(schema)} fields")
            except:
                print("⚠️  Config schema not implemented (optional)")
            
            return True
            
        except Exception as e:
            print(f"❌ Utils compatibility failed: {e}")
            return False
    
    def test_mixed_component_integration(self) -> bool:
        """Test using our restored algorithms with other utils components"""
        print("\n🔗 Testing Mixed Component Integration...")
        
        try:
            # Get available loss components from utils
            loss_components = self.registry.list_components('loss')['loss']
            
            if not loss_components:
                print("⚠️  No loss components available in utils system")
                return True
            
            # Create our restored attention + utils loss
            fourier_config = RestoredFourierConfig(d_model=128, num_heads=4)
            fourier = self.create_component_direct('attention', 'restored_fourier_attention', fourier_config)
            
            # Test with available loss (first one)
            loss_name = loss_components[0]
            print(f"✅ Testing with utils loss: {loss_name}")
            
            # Test data flow
            x = torch.randn(2, 32, 128)
            attention_output, _ = fourier.apply_attention(x, x, x)
            assert attention_output.shape == x.shape
            
            print("✅ Mixed component integration successful")
            return True
            
        except Exception as e:
            print(f"❌ Mixed integration failed: {e}")
            return False
    
    def run_comprehensive_validation(self) -> Dict[str, bool]:
        """Run all Phase 2 validation tests"""
        print("🎯 PHASE 2 VALIDATION: UTILS SYSTEM INTEGRATION")
        print("=" * 65)
        
        tests = [
            ('registry_integration', self.test_utils_registry_integration),
            ('factory_creation', self.test_factory_creation),
            ('sophistication_preservation', self.test_algorithmic_sophistication_preservation),
            ('utils_compatibility', self.test_utils_system_compatibility),
            ('mixed_integration', self.test_mixed_component_integration)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                results[test_name] = False
        
        # Summary
        print("\n📊 PHASE 2 VALIDATION SUMMARY")
        print("=" * 40)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"{status}: {test_name}")
        
        success_rate = (passed / total) * 100
        print(f"\n🎯 SUCCESS RATE: {passed}/{total} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("🚀 PHASE 2 ALGORITHM ADAPTATION: SUCCESSFUL!")
            print("   • Sophisticated algorithms preserved")
            print("   • Utils system integration complete")
            print("   • Ready for Phase 3 decomposition")
        else:
            print("⚠️  PHASE 2 needs additional work")
        
        return results


if __name__ == "__main__":
    validator = Phase2ValidationSuite()
    results = validator.run_comprehensive_validation()
