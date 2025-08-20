"""
Component validation utilities for decoder implementations.
"""
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Type, Union
import inspect
from .decoder_output import DecoderOutput


class ComponentValidator:
    """
    Validates decoder components for compatibility and correctness.
    """
    
    @staticmethod
    def validate_decoder_component(component: nn.Module, 
                                 component_name: str = "decoder") -> Dict[str, Any]:
        """
        Validate a decoder component for interface compliance.
        
        Args:
            component: The decoder component to validate
            component_name: Name for error reporting
            
        Returns:
            Dict with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Check if component has forward method
        if not hasattr(component, 'forward'):
            results['valid'] = False
            results['errors'].append(f"{component_name} missing forward method")
            return results
        
        # Check forward method signature
        try:
            sig = inspect.signature(component.forward)
            params = list(sig.parameters.keys())
            
            # Expected parameters (excluding 'self')
            expected_params = ['x', 'cross']
            optional_params = ['x_mask', 'cross_mask', 'trend']
            
            # Check required parameters
            for param in expected_params:
                if param not in params:
                    results['valid'] = False
                    results['errors'].append(f"{component_name} missing required parameter: {param}")
            
            # Check optional parameters
            for param in optional_params:
                if param in params:
                    param_obj = sig.parameters[param]
                    if param_obj.default is not inspect.Parameter.empty and param_obj.default is not None:
                        results['warnings'].append(f"{component_name} parameter {param} has non-None default")
            
            results['info']['signature'] = str(sig)
            results['info']['parameters'] = params
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Error inspecting {component_name} signature: {str(e)}")
        
        return results
    
    @staticmethod
    def validate_decoder_output(output: Any, 
                              component_name: str = "decoder") -> Dict[str, Any]:
        """
        Validate decoder output format.
        
        Args:
            output: The decoder output to validate
            component_name: Name for error reporting
            
        Returns:
            Dict with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Check output type
        if isinstance(output, DecoderOutput):
            results['info']['format'] = 'DecoderOutput'
            results['info']['aux_loss'] = output.aux_loss
        elif isinstance(output, (tuple, list)):
            results['info']['format'] = f"{len(output)}-tuple"
            
            if len(output) == 2:
                seasonal, trend = output
                if not isinstance(seasonal, torch.Tensor) or not isinstance(trend, torch.Tensor):
                    results['valid'] = False
                    results['errors'].append(f"{component_name} output elements must be torch.Tensor")
            elif len(output) == 3:
                seasonal, trend, aux_loss = output
                if not isinstance(seasonal, torch.Tensor) or not isinstance(trend, torch.Tensor):
                    results['valid'] = False
                    results['errors'].append(f"{component_name} seasonal and trend must be torch.Tensor")
                if not isinstance(aux_loss, (int, float, torch.Tensor)):
                    results['valid'] = False
                    results['errors'].append(f"{component_name} aux_loss must be numeric")
            else:
                results['valid'] = False
                results['errors'].append(f"{component_name} output must be 2 or 3 elements, got {len(output)}")
        else:
            results['valid'] = False
            results['errors'].append(f"{component_name} output must be tuple, list, or DecoderOutput")
        
        return results
    
    @staticmethod
    def validate_tensor_compatibility(x: torch.Tensor, 
                                    cross: torch.Tensor,
                                    trend: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Validate tensor compatibility for decoder inputs.
        
        Args:
            x: Input tensor
            cross: Cross-attention tensor
            trend: Optional trend tensor
            
        Returns:
            Dict with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Check tensor types
        if not isinstance(x, torch.Tensor):
            results['valid'] = False
            results['errors'].append(f"x must be torch.Tensor, got {type(x)}")
        
        if not isinstance(cross, torch.Tensor):
            results['valid'] = False
            results['errors'].append(f"cross must be torch.Tensor, got {type(cross)}")
        
        if trend is not None and not isinstance(trend, torch.Tensor):
            results['valid'] = False
            results['errors'].append(f"trend must be torch.Tensor or None, got {type(trend)}")
        
        if not results['valid']:
            return results
        
        # Check dimensions
        if x.dim() != 3:
            results['valid'] = False
            results['errors'].append(f"x must be 3D tensor, got {x.dim()}D")
        
        if cross.dim() != 3:
            results['valid'] = False
            results['errors'].append(f"cross must be 3D tensor, got {cross.dim()}D")
        
        if not results['valid']:
            return results
        
        # Check shape compatibility
        if x.size(0) != cross.size(0):
            results['valid'] = False
            results['errors'].append(f"Batch size mismatch: x={x.size(0)}, cross={cross.size(0)}")
        
        if x.size(-1) != cross.size(-1):
            results['valid'] = False
            results['errors'].append(f"Feature dimension mismatch: x={x.size(-1)}, cross={cross.size(-1)}")
        
        if trend is not None:
            if trend.dim() != 3:
                results['warnings'].append(f"trend should be 3D tensor, got {trend.dim()}D")
            elif trend.size(0) != x.size(0):
                results['valid'] = False
                results['errors'].append(f"Trend batch size mismatch: x={x.size(0)}, trend={trend.size(0)}")
        
        # Store shape info
        results['info']['shapes'] = {
            'x': list(x.shape),
            'cross': list(cross.shape),
            'trend': list(trend.shape) if trend is not None else None
        }
        
        return results


class DecoderTestRunner:
    """
    Runs comprehensive tests on decoder components.
    """
    
    def __init__(self, decoder: nn.Module, device: str = 'cpu'):
        self.decoder = decoder
        self.device = device
        self.validator = ComponentValidator()
    
    def run_smoke_test(self, batch_size: int = 2, seq_len: int = 10, 
                      d_model: int = 8) -> Dict[str, Any]:
        """
        Run a smoke test on the decoder.
        
        Args:
            batch_size: Batch size for test tensors
            seq_len: Sequence length for test tensors
            d_model: Model dimension for test tensors
            
        Returns:
            Dict with test results
        """
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'performance': {}
        }
        
        try:
            # Create test tensors
            x = torch.randn(batch_size, seq_len, d_model, device=self.device)
            cross = torch.randn(batch_size, seq_len + 2, d_model, device=self.device)
            trend = torch.randn(batch_size, seq_len, d_model, device=self.device)
            
            # Validate inputs
            input_validation = self.validator.validate_tensor_compatibility(x, cross, trend)
            if not input_validation['valid']:
                results['passed'] = False
                results['errors'].extend(input_validation['errors'])
                return results
            
            # Run forward pass
            import time
            start_time = time.time()
            
            with torch.no_grad():
                output = self.decoder(x, cross, trend=trend)
            
            forward_time = time.time() - start_time
            results['performance']['forward_time'] = forward_time
            
            # Validate output
            output_validation = self.validator.validate_decoder_output(output)
            if not output_validation['valid']:
                results['passed'] = False
                results['errors'].extend(output_validation['errors'])
            
            results['warnings'].extend(output_validation['warnings'])
            
            # Check output shapes
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                seasonal, trend_out = output[0], output[1]
                if seasonal.shape != x.shape:
                    results['warnings'].append(f"Seasonal output shape {seasonal.shape} != input shape {x.shape}")
                
                results['performance']['output_shapes'] = {
                    'seasonal': list(seasonal.shape),
                    'trend': list(trend_out.shape)
                }
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"Smoke test failed: {str(e)}")
        
        return results
    
    def run_gradient_test(self, batch_size: int = 2, seq_len: int = 10, 
                         d_model: int = 8) -> Dict[str, Any]:
        """
        Test gradient flow through the decoder.
        
        Args:
            batch_size: Batch size for test tensors
            seq_len: Sequence length for test tensors
            d_model: Model dimension for test tensors
            
        Returns:
            Dict with gradient test results
        """
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'gradient_info': {}
        }
        
        try:
            # Create test tensors with gradient tracking
            x = torch.randn(batch_size, seq_len, d_model, device=self.device, requires_grad=True)
            cross = torch.randn(batch_size, seq_len + 2, d_model, device=self.device, requires_grad=True)
            trend = torch.randn(batch_size, seq_len, d_model, device=self.device, requires_grad=True)
            
            # Forward pass
            output = self.decoder(x, cross, trend=trend)
            
            # Extract seasonal component for loss
            if isinstance(output, (tuple, list)):
                seasonal = output[0]
            else:
                seasonal = output.seasonal
            
            # Compute dummy loss and backward
            loss = seasonal.mean()
            loss.backward()
            
            # Check gradients
            if x.grad is None:
                results['warnings'].append("No gradient for input x")
            else:
                results['gradient_info']['x_grad_norm'] = x.grad.norm().item()
            
            if cross.grad is None:
                results['warnings'].append("No gradient for cross input")
            else:
                results['gradient_info']['cross_grad_norm'] = cross.grad.norm().item()
            
            # Check decoder parameter gradients
            param_grad_norms = []
            for name, param in self.decoder.named_parameters():
                if param.grad is not None:
                    param_grad_norms.append(param.grad.norm().item())
                else:
                    results['warnings'].append(f"No gradient for parameter {name}")
            
            if param_grad_norms:
                results['gradient_info']['param_grad_norms'] = {
                    'mean': sum(param_grad_norms) / len(param_grad_norms),
                    'max': max(param_grad_norms),
                    'min': min(param_grad_norms)
                }
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"Gradient test failed: {str(e)}")
        
        return results