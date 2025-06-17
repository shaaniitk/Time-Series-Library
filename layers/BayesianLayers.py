"""
Bayesian Neural Network Layers for Uncertainty Quantification

This module implements Bayesian versions of common neural network layers
that can be used to replace standard layers in Autoformer for uncertainty
quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.logger import logger


class BayesianLinear(nn.Module):
    """
    Bayesian Linear layer with weight uncertainty.
    
    Instead of fixed weights, this layer learns weight distributions
    and samples from them during forward passes to estimate uncertainty.
    """
    
    def __init__(self, in_features, out_features, bias=True, prior_std=1.0):
        super(BayesianLinear, self).__init__()
        logger.info(f"Initializing BayesianLinear: {in_features} -> {out_features}")
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log variance)
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), -5.0))
        
        # Bias parameters
        if bias:
            self.bias_mean = nn.Parameter(torch.randn(out_features) * 0.1)
            self.bias_logvar = nn.Parameter(torch.full((out_features,), -5.0))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_logvar', None)
    
    def forward(self, x):
        """
        Forward pass with weight sampling.
        
        Args:
            x: [batch_size, seq_len, in_features] input tensor
            
        Returns:
            output: [batch_size, seq_len, out_features] output tensor
        """
        # Sample weights from learned distributions
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight_eps = torch.randn_like(self.weight_mean)
        weight = self.weight_mean + weight_std * weight_eps
        
        # Sample bias if present
        bias = None
        if self.bias_mean is not None:
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias_eps = torch.randn_like(self.bias_mean)
            bias = self.bias_mean + bias_std * bias_eps
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """
        Compute KL divergence between learned distributions and priors.
        This is used as a regularization term in the loss function.
        """
        # KL divergence for weights
        weight_var = torch.exp(self.weight_logvar)
        kl_weight = 0.5 * torch.sum(
            self.weight_mean ** 2 / (self.prior_std ** 2) + 
            weight_var / (self.prior_std ** 2) - 
            self.weight_logvar + 
            math.log(self.prior_std ** 2) - 1
        )
        
        kl_total = kl_weight
        
        # KL divergence for bias
        if self.bias_mean is not None:
            bias_var = torch.exp(self.bias_logvar)
            kl_bias = 0.5 * torch.sum(
                self.bias_mean ** 2 / (self.prior_std ** 2) + 
                bias_var / (self.prior_std ** 2) - 
                self.bias_logvar + 
                math.log(self.prior_std ** 2) - 1
            )
            kl_total += kl_bias
        
        return kl_total


class BayesianConv1d(nn.Module):
    """
    Bayesian 1D Convolution layer with weight uncertainty.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, bias=True, prior_std=1.0):
        super(BayesianConv1d, self).__init__()
        logger.info(f"Initializing BayesianConv1d: {in_channels} -> {out_channels}, kernel={kernel_size}")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.prior_std = prior_std
        
        # Weight parameters
        self.weight_mean = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size) * 0.1)
        self.weight_logvar = nn.Parameter(torch.full((out_channels, in_channels, kernel_size), -5.0))
        
        # Bias parameters
        if bias:
            self.bias_mean = nn.Parameter(torch.randn(out_channels) * 0.1)
            self.bias_logvar = nn.Parameter(torch.full((out_channels,), -5.0))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_logvar', None)
    
    def forward(self, x):
        # Sample weights
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight_eps = torch.randn_like(self.weight_mean)
        weight = self.weight_mean + weight_std * weight_eps
        
        # Sample bias
        bias = None
        if self.bias_mean is not None:
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias_eps = torch.randn_like(self.bias_mean)
            bias = self.bias_mean + bias_std * bias_eps
        
        return F.conv1d(x, weight, bias, self.stride, self.padding)
    
    def kl_divergence(self):
        """Compute KL divergence for regularization"""
        weight_var = torch.exp(self.weight_logvar)
        kl_weight = 0.5 * torch.sum(
            self.weight_mean ** 2 / (self.prior_std ** 2) + 
            weight_var / (self.prior_std ** 2) - 
            self.weight_logvar + 
            math.log(self.prior_std ** 2) - 1
        )
        
        kl_total = kl_weight
        
        if self.bias_mean is not None:
            bias_var = torch.exp(self.bias_logvar)
            kl_bias = 0.5 * torch.sum(
                self.bias_mean ** 2 / (self.prior_std ** 2) + 
                bias_var / (self.prior_std ** 2) - 
                self.bias_logvar + 
                math.log(self.prior_std ** 2) - 1
            )
            kl_total += kl_bias
        
        return kl_total


class DropoutSampling(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    This is a simpler alternative to full Bayesian layers that uses
    dropout during inference to approximate Bayesian uncertainty.
    """
    
    def __init__(self, p=0.1):
        super(DropoutSampling, self).__init__()
        self.p = p
        
    def forward(self, x):
        # Apply dropout even during evaluation for uncertainty sampling
        return F.dropout(x, p=self.p, training=True)


def convert_to_bayesian(module, layer_types_to_convert=['Linear'], prior_std=1.0):
    """
    Convert a standard neural network module to use Bayesian layers.
    
    Args:
        module: PyTorch module to convert
        layer_types_to_convert: List of layer types to convert ['Linear', 'Conv1d']
        prior_std: Prior standard deviation for Bayesian layers
        
    Returns:
        Modified module with Bayesian layers
    """
    logger.info(f"Converting module to Bayesian with layer types: {layer_types_to_convert}")
    
    for name, child in module.named_children():
        if len(list(child.children())) > 0:
            # Recursively convert child modules
            convert_to_bayesian(child, layer_types_to_convert, prior_std)
        else:
            # Convert leaf modules
            if 'Linear' in layer_types_to_convert and isinstance(child, nn.Linear):
                bayesian_layer = BayesianLinear(
                    child.in_features, 
                    child.out_features, 
                    child.bias is not None,
                    prior_std
                )
                setattr(module, name, bayesian_layer)
                logger.info(f"Converted {name} to BayesianLinear")
                
            elif 'Conv1d' in layer_types_to_convert and isinstance(child, nn.Conv1d):
                bayesian_layer = BayesianConv1d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size[0],
                    child.stride[0],
                    child.padding[0],
                    child.bias is not None,
                    prior_std
                )
                setattr(module, name, bayesian_layer)
                logger.info(f"Converted {name} to BayesianConv1d")
    
    return module


def collect_kl_divergence(module):
    """
    Collect KL divergence from all Bayesian layers in a module.
    
    Args:
        module: PyTorch module containing Bayesian layers
        
    Returns:
        total_kl: Sum of KL divergences from all Bayesian layers
    """
    total_kl = 0.0
    
    for child in module.modules():
        if hasattr(child, 'kl_divergence'):
            total_kl += child.kl_divergence()
    
    return total_kl


# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing Bayesian layers...")
    
    # Test BayesianLinear
    bayesian_linear = BayesianLinear(64, 32)
    x = torch.randn(4, 96, 64)
    
    # Multiple forward passes should give different outputs
    out1 = bayesian_linear(x)
    out2 = bayesian_linear(x)
    
    print(f"Output difference (should be non-zero): {torch.norm(out1 - out2).item():.6f}")
    print(f"KL divergence: {bayesian_linear.kl_divergence().item():.6f}")
    
    # Test BayesianConv1d
    bayesian_conv = BayesianConv1d(64, 32, kernel_size=3, padding=1)
    x_conv = x.transpose(1, 2)  # [batch, channels, seq_len]
    
    out_conv1 = bayesian_conv(x_conv)
    out_conv2 = bayesian_conv(x_conv)
    
    print(f"Conv output difference: {torch.norm(out_conv1 - out_conv2).item():.6f}")
    print(f"Conv KL divergence: {bayesian_conv.kl_divergence().item():.6f}")
    
    logger.info("Bayesian layers test completed!")
