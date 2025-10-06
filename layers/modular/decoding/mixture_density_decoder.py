
import torch
import torch.nn as nn
import torch.distributions as D
from typing import Tuple

class MixtureDensityDecoder(nn.Module):
    """
    A decoder that outputs the parameters of a mixture of Gaussian distributions.
    This allows for probabilistic forecasting.
    """
    def __init__(self, d_model: int, output_dim: int, num_gaussians: int = 5):
        """
        Args:
            d_model (int): The input feature dimension from the model's encoder.
            output_dim (int): The dimension of the target variable to be forecast.
            num_gaussians (int): The number of Gaussian distributions in the mixture.
        """
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians

        # The projection layer will output the parameters for all Gaussians.
        # For each Gaussian, we need: a mean, a standard deviation, and a mixture weight.
        # Total parameters per Gaussian = 2 (mean, std) + 1 (weight)
        # We predict log_sigma and then exponentiate to ensure sigma is positive.
        # We predict logits for the mixture weights and use softmax.
        self.projection = nn.Linear(d_model, (2 * self.output_dim + 1) * num_gaussians)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): The input tensor from the encoder, shape [batch_size, seq_len, d_model].

        Returns:
            pi (torch.Tensor): Mixture weights (logits), shape [batch_size, seq_len, num_gaussians].
            mu (torch.Tensor): Means of the Gaussians, shape [batch_size, seq_len, num_gaussians, output_dim].
            sigma (torch.Tensor): Standard deviations of the Gaussians, shape [batch_size, seq_len, num_gaussians, output_dim].
        """
        # Project the encoder output to the mixture parameters
        params = self.projection(x)

        # Split the parameters into pi, sigma, and mu
        # The split size for each is num_gaussians * output_dim for mu and sigma, and num_gaussians for pi.
        pi_logits, log_sigma, mu = torch.split(
            params, 
            [self.num_gaussians, self.num_gaussians * self.output_dim, self.num_gaussians * self.output_dim], 
            dim=-1
        )
        
        # Reshape mu and sigma
        batch_size, seq_len, _ = x.shape
        mu = mu.view(batch_size, seq_len, self.num_gaussians, self.output_dim)
        log_sigma = log_sigma.view(batch_size, seq_len, self.num_gaussians, self.output_dim)

        # Ensure sigma is positive
        sigma = torch.exp(log_sigma)

        return pi_logits, mu, sigma

    def loss(self, pi_logits: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Negative Log-Likelihood loss for the mixture density network.

        Args:
            pi_logits (torch.Tensor): Mixture weights (logits).
            mu (torch.Tensor): Means of the Gaussians.
            sigma (torch.Tensor): Standard deviations of the Gaussians.
            targets (torch.Tensor): The ground truth values, shape [batch_size, seq_len, output_dim].

        Returns:
            torch.Tensor: The NLL loss.
        """
        # Expand targets to match the shape of mu and sigma for broadcasting
        targets = targets.unsqueeze(2).expand_as(mu)

        # Create the mixture distribution
        # The Independent wrapper treats the output_dim as a single event.
        # The MixtureSameFamily combines the Gaussians.
        component_distribution = D.Independent(D.Normal(loc=mu, scale=sigma), 1)
        mixture_distribution = D.MixtureSameFamily(
            mixture_distribution=D.Categorical(logits=pi_logits),
            component_distribution=component_distribution
        )

        # Calculate the negative log-likelihood
        nll_loss = -mixture_distribution.log_prob(targets)

        return nll_loss.mean()
