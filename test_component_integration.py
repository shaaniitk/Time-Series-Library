"""
Systematic Component Testing Framework for Enhanced SOTA PGAT
Tests each component individually and in combination
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
import numpy as np
from Enhanced_SOTA_PGAT_Refactored import Enhanced_SOTA_PGAT


class ComponentTestConfig:
    """Configuration for component testing"""
    
    def __init__(self, **kwargs):
        # Basic model configuration
        self.seq_len = kwargs.get('seq_len', 24)
        self.pred_len = kwargs.get('pred_len', 6)
        self.enc_in = kwargs.get('enc_in', 118)
        self.c_out = kwargs.get('c_out', 4)
        self.d_model = kwargs.get('d_model', 64)
        self.n_heads = kwargs.get('n_heads', 4)
        self.dropout = kwargs.get('dropout', 0.1)
        
        # Component toggles - start with all disabled
        self.use_multi_scale_patching = kwargs.get('use_multi_scale_patching', False)
        self.use_hierarchical_mapper = kwargs.get('use_hierarchical_mapper', False)
        self.use_stochastic_learner = kwargs.get('use_stochastic_learner', False)
        self.use_gated_graph_combiner = kwargs.get('use_gated_graph_combiner', False)
        self.use_mixture_decoder = kwargs.get('use_mixture_decoder', False)
        
        # Additional parameters
        self.num_wave_features = kwargs.get('num_wave_features', 114)
        self.mdn_components = kwargs.get('mdn_components', 3)
        self.mixture_multivariate_mode = kwargs.get('mixture_multivariate_mode', 'independent')


class ComponentTester:
    """Tests individual components and combinations"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.test_results = {}
    
    def create_test_data(self, config: ComponentTestConfig):
        """Create synthetic test data"""
        batch_size = 2
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in).to(self.device)
        target_window = torch.randn(batch_size, config.pred_len, config.enc_in).to(self.device)
        return wave_window, target_window
    
    def test_baseline_model(self):
        """Test baseline model with all components disabled"""
        print("🔧 Testing Baseline Model (All Components Disabled)")
        
        config = ComponentTestConfig()
        model = Enhanced_SOTA_PGAT(config).to(self.device)
        
        wave_window, target_window = self.create_test_data(config)
        
        try:
            with torch.no_grad():
                output = model(wave_window, target_window)
            
            print(f"✅ Baseline model works - Output shape: {output.shape}")
            self.test_results['baseline'] = {'status': 'success', 'output_shape': output.shape}
            return True
        except Exception as e:
            print(f"❌ Baseline model failed: {e}")
            self.test_results['baseline'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_individual_components(self):
        """Test each component individually"""
        components = [
            ('multi_scale_patching', 'use_multi_scale_patching'),
            ('hierarchical_mapper', 'use_hierarchical_mapper'),
            ('stochastic_learner', 'use_stochastic_learner'),
            ('gated_graph_combiner', 'use_gated_graph_combiner'),
            ('mixture_decoder', 'use_mixture_decoder')
        ]
        
        for component_name, config_attr in components:
            print(f"\n🔧 Testing {component_name.replace('_', ' ').title()}")
            
            # Create config with only this component enabled
            config_kwargs = {config_attr: True}
            config = ComponentTestConfig(**config_kwargs)
            
            try:
                model = Enhanced_SOTA_PGAT(config).to(self.device)
                wave_window, target_window = self.create_test_data(config)
                
                with torch.no_grad():
                    output = model(wave_window, target_window)
                
                # Handle MDN outputs (tuple) vs regular outputs (tensor)
                if isinstance(output, tuple):
                    output_shape = f"MDN: {[o.shape for o in output]}"
                    print(f"✅ {component_name} works - Output shape: {output_shape}")
                else:
                    output_shape = output.shape
                    print(f"✅ {component_name} works - Output shape: {output_shape}")
                
                self.test_results[component_name] = {
                    'status': 'success', 
                    'output_shape': output_shape,
                    'is_mdn': isinstance(output, tuple)
                }
                
            except Exception as e:
                print(f"❌ {component_name} failed: {e}")
                self.test_results[component_name] = {'status': 'failed', 'error': str(e)}
    
    def test_component_combinations(self):
        """Test combinations of components"""
        print("\n🔧 Testing Component Combinations")
        
        # Test pairs
        combinations = [
            ('patching + hierarchical', {'use_multi_scale_patching': True, 'use_hierarchical_mapper': True}),
            ('hierarchical + stochastic', {'use_hierarchical_mapper': True, 'use_stochastic_learner': True}),
            ('stochastic + combiner', {'use_stochastic_learner': True, 'use_gated_graph_combiner': True}),
            ('combiner + mixture', {'use_gated_graph_combiner': True, 'use_mixture_decoder': True}),
        ]
        
        for combo_name, config_kwargs in combinations:
            print(f"\n  Testing {combo_name}")
            
            config = ComponentTestConfig(**config_kwargs)
            
            try:
                model = Enhanced_SOTA_PGAT(config).to(self.device)
                wave_window, target_window = self.create_test_data(config)
                
                with torch.no_grad():
                    output = model(wave_window, target_window)
                
                # Handle MDN outputs
                if isinstance(output, tuple):
                    output_shape = f"MDN: {[o.shape for o in output]}"
                else:
                    output_shape = output.shape
                
                print(f"  ✅ {combo_name} works - Output shape: {output_shape}")
                self.test_results[f'combo_{combo_name}'] = {
                    'status': 'success', 
                    'output_shape': output_shape
                }
                
            except Exception as e:
                print(f"  ❌ {combo_name} failed: {e}")
                self.test_results[f'combo_{combo_name}'] = {'status': 'failed', 'error': str(e)}
    
    def test_full_model(self):
        """Test full model with all components enabled"""
        print("\n🔧 Testing Full Model (All Components Enabled)")
        
        config = ComponentTestConfig(
            use_multi_scale_patching=True,
            use_hierarchical_mapper=True,
            use_stochastic_learner=True,
            use_gated_graph_combiner=True,
            use_mixture_decoder=True
        )
        
        try:
            model = Enhanced_SOTA_PGAT(config).to(self.device)
            wave_window, target_window = self.create_test_data(config)
            
            with torch.no_grad():
                output = model(wave_window, target_window)
            
            # Handle MDN outputs
            if isinstance(output, tuple):
                output_shape = f"MDN: {[o.shape for o in output]}"
                print(f"✅ Full model works - Output shape: {output_shape}")
                print(f"   MDN output: True")
                
                # Test loss computation
                targets = torch.randn(2, config.pred_len, config.c_out).to(self.device)
                loss = model.loss(output, targets)
                print(f"   Loss computation works: {loss.item():.6f}")
            else:
                output_shape = output.shape
                print(f"✅ Full model works - Output shape: {output_shape}")
                print(f"   MDN output: False")
            
            self.test_results['full_model'] = {
                'status': 'success', 
                'output_shape': output_shape,
                'is_mdn': isinstance(output, tuple)
            }
            return True
            
        except Exception as e:
            print(f"❌ Full model failed: {e}")
            self.test_results['full_model'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_gradient_flow(self):
        """Test gradient flow through the model"""
        print("\n🔧 Testing Gradient Flow")
        
        config = ComponentTestConfig(
            use_multi_scale_patching=True,
            use_hierarchical_mapper=True,
            use_stochastic_learner=True,
            use_gated_graph_combiner=True,
            use_mixture_decoder=True
        )
        
        try:
            model = Enhanced_SOTA_PGAT(config).to(self.device)
            wave_window, target_window = self.create_test_data(config)
            targets = torch.randn(2, config.pred_len, config.c_out).to(self.device)
            
            # Forward pass
            output = model(wave_window, target_window)
            loss = model.loss(output, targets)
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 0:
                        grad_norms[name] = grad_norm
            
            print(f"✅ Gradient flow works - {len(grad_norms)} parameters have gradients")
            print(f"   Average gradient norm: {np.mean(list(grad_norms.values())):.6f}")
            
            self.test_results['gradient_flow'] = {
                'status': 'success',
                'num_params_with_grads': len(grad_norms),
                'avg_grad_norm': np.mean(list(grad_norms.values()))
            }
            return True
            
        except Exception as e:
            print(f"❌ Gradient flow failed: {e}")
            self.test_results['gradient_flow'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def run_all_tests(self):
        """Run all component tests"""
        print("🚀 Starting Systematic Component Testing")
        print("=" * 60)
        
        # Test baseline first
        baseline_ok = self.test_baseline_model()
        
        if baseline_ok:
            # Test individual components
            self.test_individual_components()
            
            # Test combinations
            self.test_component_combinations()
            
            # Test full model
            full_ok = self.test_full_model()
            
            if full_ok:
                # Test gradient flow
                self.test_gradient_flow()
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        success_count = 0
        total_count = 0
        
        for test_name, result in self.test_results.items():
            total_count += 1
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"{status_icon} {test_name.replace('_', ' ').title()}")
            
            if result['status'] == 'success':
                success_count += 1
                if 'output_shape' in result:
                    print(f"    Output shape: {result['output_shape']}")
            else:
                print(f"    Error: {result['error']}")
        
        print(f"\n📈 Success Rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
        
        if success_count == total_count:
            print("🎉 All tests passed! The refactored model is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the errors above.")


def main():
    """Main testing function"""
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run tests
    tester = ComponentTester(device=device)
    tester.run_all_tests()


if __name__ == "__main__":
    main()