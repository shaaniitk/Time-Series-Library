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

    def forward(self, x, mode: str, valid_mask=None):
        if mode == 'norm':
            self._get_statistics(x, valid_mask=valid_mask)
            x = self._normalize(x, valid_mask=valid_mask)
        elif mode == 'denorm':
            if valid_mask is not None:
                raise ValueError("valid_mask is only accepted for RevIN normalization.")
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

    @staticmethod
    def _canonical_valid_mask(x, valid_mask):
        if valid_mask is None:
            return None
        if not torch.is_tensor(valid_mask):
            valid_mask = torch.as_tensor(valid_mask, device=x.device)
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have boolean dtype.")
        if valid_mask.ndim != 2 or tuple(valid_mask.shape) != tuple(x.shape[:2]):
            raise ValueError(
                "valid_mask must have shape [B,T] matching the normalized input; "
                f"got {tuple(valid_mask.shape)} versus {tuple(x.shape[:2])}."
            )
        valid_mask = valid_mask.to(device=x.device)
        if torch.any(valid_mask.sum(dim=1) == 0):
            raise ValueError("Every RevIN batch row must contain a valid history token.")
        return valid_mask

    def _get_statistics(self, x, valid_mask=None):
        valid_mask = self._canonical_valid_mask(x, valid_mask)
        dim2reduce = tuple(range(1, x.ndim - 1))
        if valid_mask is None or bool(valid_mask.all()):
            if self.subtract_last:
                self.last = x[:, -1, :].unsqueeze(1)
            else:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            return

        expanded_mask = valid_mask
        while expanded_mask.ndim < x.ndim:
            expanded_mask = expanded_mask.unsqueeze(-1)
        safe_x = torch.where(expanded_mask, x, torch.zeros_like(x))
        count = valid_mask.sum(dim=1, keepdim=True).to(dtype=x.dtype)
        while count.ndim < x.ndim:
            count = count.unsqueeze(-1)
        if self.subtract_last:
            indices = torch.arange(
                x.shape[1], device=x.device, dtype=torch.long
            ).unsqueeze(0).expand_as(valid_mask)
            last_indices = torch.where(
                valid_mask, indices, torch.zeros_like(indices)
            ).amax(dim=1)
            gather_index = last_indices.view(-1, 1, *([1] * (x.ndim - 2)))
            gather_index = gather_index.expand(-1, 1, *x.shape[2:])
            self.last = torch.gather(safe_x, 1, gather_index).detach()
            centered = torch.where(
                expanded_mask, x - self.last, torch.zeros_like(x)
            )
        else:
            self.mean = (safe_x.sum(dim=dim2reduce, keepdim=True) / count).detach()
            centered = torch.where(
                expanded_mask, x - self.mean, torch.zeros_like(x)
            )
        variance = centered.square().sum(dim=dim2reduce, keepdim=True) / count
        self.stdev = torch.sqrt(variance + self.eps).detach()

    def _normalize(self, x, valid_mask=None):
        if self.non_norm:
            normalized = x
        else:
            if self.subtract_last:
                normalized = x - self.last
            else:
                normalized = x - self.mean
            normalized = normalized / self.stdev
            if self.affine:
                normalized = normalized * self.affine_weight
                normalized = normalized + self.affine_bias
        valid_mask = self._canonical_valid_mask(x, valid_mask)
        if valid_mask is not None and not bool(valid_mask.all()):
            expanded_mask = valid_mask
            while expanded_mask.ndim < normalized.ndim:
                expanded_mask = expanded_mask.unsqueeze(-1)
            normalized = torch.where(
                expanded_mask, normalized, torch.zeros_like(normalized)
            )
        return normalized

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
