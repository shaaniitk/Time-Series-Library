# Encoder Implementation Improvement Plan

This document outlines a detailed step-by-step plan to improve the encoder implementation in the `layers` directory. The goal is to create a more modular, reusable, and maintainable codebase.

## Recommendation 1: Complete the Modularization

The current modular encoders have a dependency on the original, non-modular `Encoder` class. This dependency should be removed to make the modular encoders fully self-contained.

**Steps:**

1.  **Create a new `Encoder` class:** Create a new `Encoder` class in the `layers/modular/encoder` directory. This class will take a list of encoder layers and a normalization layer as input.

    ```python
    # layers/modular/encoder/base.py
    import torch.nn as nn
    from typing import List, Optional

    class ModularEncoder(nn.Module):
        def __init__(self, layers: List[nn.Module], norm_layer: Optional[nn.Module] = None):
            super(ModularEncoder, self).__init__()
            self.layers = nn.ModuleList(layers)
            self.norm = norm_layer

        def forward(self, x, attn_mask=None):
            attns = []
            for layer in self.layers:
                x, attn = layer(x, attn_mask=attn_mask)
                attns.append(attn)

            if self.norm is not None:
                x = self.norm(x)

            return x, attns
    ```

2.  **Update the modular encoders:** Update the `StandardEncoder`, `EnhancedEncoder`, and `StableEncoder` classes to use the new `ModularEncoder` class.

    ```python
    # layers/modular/encoder/standard_encoder.py
    from .base import BaseEncoder, ModularEncoder
    from ..layers.standard_layers import StandardEncoderLayer
    import torch.nn as nn

    class StandardEncoder(BaseEncoder):
        def __init__(self, num_encoder_layers, d_model, n_heads, d_ff, dropout, activation, 
                     attention_comp, decomp_comp, conv_layers=None, norm_layer=None):
            super(StandardEncoder, self).__init__()
            
            self.encoder = ModularEncoder(
                [
                    StandardEncoderLayer(
                        attention_comp,
                        decomp_comp,
                        d_model,
                        n_heads,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for _ in range(num_encoder_layers)
                ],
                norm_layer=norm_layer
            )

        def forward(self, x, attn_mask=None):
            return self.encoder(x, attn_mask)
    ```

## Recommendation 2: Refactor to Improve Code Reuse

There is a significant amount of code duplication between the `StandardEncoderLayer` and `EnhancedEncoderLayer`. This duplicated code should be extracted into a common, reusable component.

**Steps:**

1.  **Create a `FeedForward` module:** Create a new `FeedForward` module that encapsulates the shared feed-forward network logic.

    ```python
    # layers/modular/layers/common.py
    import torch.nn as nn
    import torch.nn.functional as F

    class FeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
            super(FeedForward, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.activation = F.relu if activation == "relu" else F.gelu

        def forward(self, x):
            y = x
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            return x + y
    ```

2.  **Update the encoder layers:** Update the `StandardEncoderLayer` and `EnhancedEncoderLayer` to use the new `FeedForward` module.

    ```python
    # layers/modular/layers/standard_layers.py
    from .common import FeedForward

    class StandardEncoderLayer(BaseEncoderLayer):
        def __init__(self, attention_component, decomposition_component, d_model, n_heads, d_ff, dropout=0.1, activation="relu"):
            super(StandardEncoderLayer, self).__init__()
            self.attention = attention_component
            self.decomp1 = decomposition_component
            self.decomp2 = decomposition_component
            self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        def forward(self, x, attn_mask=None):
            new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
            x = x + self.dropout(new_x)
            x, _ = self.decomp1(x)
            x = self.feed_forward(x)
            res, _ = self.decomp2(x)
            return res, attn
    ```

## Recommendation 3: Implement a Centralized Configuration System

The current configuration system is a mix of `argparse` and Pydantic schemas. The `HierarchicalEncoder` still uses a `mock_configs` namespace. A centralized configuration system should be used to manage all component parameters.

**Steps:**

1.  **Use the existing schemas:** The `configs/schemas.py` file already provides a good set of Pydantic schemas. These schemas should be used to configure all components.
2.  **Remove `mock_configs`:** The `mock_configs` namespace should be removed from the `HierarchicalEncoder`. The parameters should be passed in through the constructor, which will be populated from the `ModularAutoformerConfig`.

## Recommendation 4: Write Comprehensive Documentation

The code lacks detailed documentation. Detailed docstrings should be added to all modules, classes, and functions.

**Steps:**

1.  **Document all modules:** Add a docstring to the top of each module that explains the purpose of the module.
2.  **Document all classes:** Add a docstring to each class that explains the purpose of the class, its parameters, and its attributes.
3.  **Document all functions:** Add a docstring to each function that explains the purpose of the function, its parameters, and its return values.

## Recommendation 5: Develop a Comprehensive Test Suite

A comprehensive test suite is essential to ensure the correctness of the various encoder implementations and to prevent regressions as the code evolves.

**Steps:**

1.  **Write unit tests:** Write unit tests for each component, including the attention, decomposition, and feed-forward modules.
2.  **Write integration tests:** Write integration tests for each encoder variant, including the `StandardEncoder`, `EnhancedEncoder`, and `HierarchicalEncoder`.
3.  **Use a test runner:** Use a test runner like `pytest` to run the tests automatically.
