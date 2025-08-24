"""
Adapters/wrappers for legacy layers.modular.output_heads to register in utils registry as 'output' components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from layers.modular.core.registry import register_component
from layers.modular.base import BaseOutput


@dataclass
class OutputHeadConfig:
    d_model: int
    output_dim: int
    num_quantiles: Optional[int] = None


class StandardOutputHeadWrapper(BaseOutput):
    def __init__(self, config: OutputHeadConfig):
        from layers.modular.output_heads.standard_output_head import StandardOutputHead
        self._cfg = config
        super().__init__(config)
        self.head = StandardOutputHead(config.d_model, config.output_dim)
        self._out = config.output_dim

    def forward(self, hidden_states, **kwargs):
        # Use last hidden state like utils heads
        x = hidden_states[:, -1, :]
        return self.head(x)

    def get_output_type(self) -> str:
        return "standard_output_head"

    def get_output_dim(self) -> int:
        return self._out


class QuantileOutputHeadWrapper(BaseOutput):
    def __init__(self, config: OutputHeadConfig):
        from layers.modular.output_heads.quantile_output_head import QuantileOutputHead
        if config.num_quantiles is None:
            raise ValueError("num_quantiles must be provided for QuantileOutputHeadWrapper")
        self._cfg = config
        super().__init__(config)
        self.head = QuantileOutputHead(config.d_model, config.output_dim, config.num_quantiles)
        self._out = config.output_dim * config.num_quantiles

    def forward(self, hidden_states, **kwargs):
        x = hidden_states[:, -1, :]
        return self.head(x)

    def get_output_type(self) -> str:
        return "quantile_output_head"

    def get_output_dim(self) -> int:
        return self._out


def register_layers_output_heads() -> None:
    register_component(
        "output",
        "layers_standard_output_head",
        StandardOutputHeadWrapper,
        metadata={
            "domain": "output",
            "source": "layers.modular.output_heads.standard_output_head",
            "features": ["linear_projection"],
        },
    )
    register_component(
        "output",
        "layers_quantile_output_head",
        QuantileOutputHeadWrapper,
        metadata={
            "domain": "output",
            "source": "layers.modular.output_heads.quantile_output_head",
            "features": ["quantile_projection"],
        },
    )
