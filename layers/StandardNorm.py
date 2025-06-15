import torch
import torch.nn as nn
from utils.logger import logger


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        logger.info(f"Initializing Normalize with num_features={num_features}, eps={eps}, affine={affine}, "
                    f"subtract_last={subtract_last}, non_norm={non_norm}")
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        logger.debug(f"Normalize forward, x shape: {x.shape}, mode: {mode}")
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        logger.info("Initializing affine parameters")
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        logger.debug(f"Getting statistics, input shape: {x.shape}")
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
            logger.debug("Subtracting last timestep for normalization")
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            logger.debug(f"Computed mean: {self.mean}")
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        logger.debug(f"Computed std deviation: {self.stdev}")

    def _normalize(self, x):
        logger.debug("Normalizing the input")
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        logger.debug("Normalization complete")
        return x

    def _denormalize(self, x):
        logger.debug("Denormalizing the input")
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        logger.debug("Denormalization complete")
        return x


class StandardNorm(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        super().__init__()
        logger.info(f"Initializing StandardNorm with num_features={num_features}, eps={eps}, affine={affine}, "
                    f"subtract_last={subtract_last}, non_norm={non_norm}")
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        logger.debug(f"StandardNorm forward, x shape: {x.shape}, mode: {mode}")
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        logger.info("Initializing affine parameters")
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        logger.debug(f"Getting statistics, input shape: {x.shape}")
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
            logger.debug("Subtracting last timestep for normalization")
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            logger.debug(f"Computed mean: {self.mean}")
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        logger.debug(f"Computed std deviation: {self.stdev}")

    def _normalize(self, x):
        logger.debug("Normalizing the input")
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        logger.debug("Normalization complete")
        return x

    def _denormalize(self, x):
        logger.debug("Denormalizing the input")
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        logger.debug("Denormalization complete")
        return x
