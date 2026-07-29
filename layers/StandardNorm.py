import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # Store raw values and transform them through softplus so denormalization
        # cannot flip quantile ordering with a negative effective scale.
        init_scale = torch.ones(self.num_features)
        self.affine_weight_raw = nn.Parameter(torch.log(torch.expm1(init_scale)))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    @property
    def affine_weight(self):
        return F.softplus(self.affine_weight_raw) + self.eps

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
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
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / self.affine_weight
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        legacy_key = prefix + 'affine_weight'
        raw_key = prefix + 'affine_weight_raw'
        if legacy_key in state_dict and raw_key not in state_dict:
            legacy_value = state_dict.pop(legacy_key)
            clipped = torch.clamp(legacy_value, min=self.eps)
            state_dict[raw_key] = torch.log(torch.expm1(clipped))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
