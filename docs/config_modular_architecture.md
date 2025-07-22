# Modular Config Architecture: Best Practices & Extension Guide

## Overview
This document describes the principles and patterns for robust modular configuration in the Time-Series-Library. It covers:
- How each component uses its own config class
- How to safely extend components and add new ones
- Registry implementation patterns

## 1. Principles of Modular Configs
- **Each component only accesses fields defined in its config class.**
- **Global model parameters are never assumed in component configs.**
- **If a component needs global parameters, pass them explicitly as arguments, not via config.**
- **All config classes are Pydantic models or dataclasses, with clear field definitions and defaults.**

## 2. Example: Adding a New Method to an Existing Component
Suppose you want to add a `get_loss_info()` method to `MSELoss`:

```python
class MSELoss(BaseLoss):
    # ...existing code...
    def get_loss_info(self) -> dict:
        return {
            'type': 'mse',
            'reduction': self.reduction,
            'differentiable': True
        }
```
- Only use fields from `LossConfig` (e.g., `reduction`).
- Do not access unrelated fields like `seq_len` or `pred_len`.

## 3. Example: Adding a Completely New Component
Suppose you want to add a new loss function, `CustomLoss`:

```python
@dataclass
class CustomLossConfig(LossConfig):
    custom_param: float = 0.5

class CustomLoss(BaseLoss):
    def __init__(self, config: Union[CustomLossConfig, Dict[str, Any]]):
        super().__init__(config)
        if isinstance(config, dict):
            self.custom_param = config.get('custom_param', 0.5)
        else:
            self.custom_param = config.custom_param
    def compute_loss(self, pred, true):
        # Use self.custom_param safely
        return (pred - true).abs().mean() * self.custom_param
```
- Register the new component in the registry:
```python
LOSS_REGISTRY['custom'] = CustomLoss
```

## 4. Registry Implementation Pattern
- Each registry is a dictionary mapping string/type to component class.
- Example (from losses):
```python
LOSS_REGISTRY = {
    'mse': MSELoss,
    'mae': MAELoss,
    # ...other losses...
}
```
- To instantiate: `loss = LOSS_REGISTRY[loss_type](config)`

## 5. Checklist for Adding/Extending Components
- [x] Define a config class with only relevant fields
- [x] Implement the component using only its config fields
- [x] Register the component in the appropriate registry
- [x] Add tests for the new/extended component

## 6. References
- See `utils/modular_components/implementations/losses.py` for unified loss registry and robust constructor patterns.
- See `tests/components/test_registry.py` for registry/factory coverage tests.

---
**This document is maintained as part of the modular architecture best practices.**
