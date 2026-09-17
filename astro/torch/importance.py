"""Learned per-rule importance gates over astrology known-future channels.

Each rule owns one scalar gate that multiplies the input values of all its
channels.  Gates start at exactly zero, so an untrained model sees the same
inputs as the capacity-matched zero arm: astrology must earn any influence
through the training signal.  The known-value projection is shared with the
ungated calendar channels, which keeps gate magnitudes comparable across rules.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from astro.rules.naming import CHANNEL_PREFIX


def parse_rule_layout(
    known_feature_names: Sequence[str],
) -> tuple[list[int], list[str], list[int], list[str]]:
    """Return gated column positions, rule ids, per-column rule index, and families."""

    columns: list[int] = []
    column_rule_index: list[int] = []
    rule_ids: list[str] = []
    families: list[str] = []
    for position, name in enumerate(known_feature_names):
        parts = str(name).split(".")
        if parts[0] != CHANNEL_PREFIX:
            continue
        if len(parts) < 5:
            raise ValueError(f"Malformed astro channel name {name!r}.")
        rule_id = parts[3]
        if rule_id not in rule_ids:
            rule_ids.append(rule_id)
            families.append(parts[2])
        columns.append(position)
        column_rule_index.append(rule_ids.index(rule_id))
    return columns, rule_ids, column_rule_index, families


class AstroRuleGates(nn.Module):
    def __init__(self, known_feature_names: Sequence[str]):
        super().__init__()
        columns, rule_ids, column_rule_index, families = parse_rule_layout(known_feature_names)
        if not rule_ids:
            raise ValueError(
                "tft_astro_rule_gates requires astro.* known features; none were found."
            )
        self.known_len = len(known_feature_names)
        self.rule_ids = tuple(rule_ids)
        self.rule_families = tuple(families)
        self.gates = nn.Parameter(torch.zeros(len(rule_ids)))
        self.register_buffer("columns", torch.tensor(columns, dtype=torch.long), persistent=False)
        self.register_buffer(
            "column_rule_index", torch.tensor(column_rule_index, dtype=torch.long), persistent=False
        )

    def scale(self, reference: torch.Tensor) -> torch.Tensor:
        scale = reference.new_ones(self.known_len)
        return scale.index_put((self.columns,), self.gates[self.column_rule_index].to(reference.dtype))

    def forward(self, x_mark_enc: torch.Tensor, x_mark_dec: torch.Tensor):
        scale = self.scale(x_mark_enc)
        return x_mark_enc * scale, x_mark_dec * scale

    def importance(self) -> dict[str, float]:
        values = self.gates.detach().cpu().tolist()
        return dict(zip(self.rule_ids, values))

    def family_importance(self) -> dict[str, float]:
        """Mean absolute gate per family; the sign of a family mean is not meaningful."""

        totals: dict[str, list[float]] = {}
        for family, value in zip(self.rule_families, self.gates.detach().abs().cpu().tolist()):
            totals.setdefault(family, []).append(value)
        return {family: sum(values) / len(values) for family, values in totals.items()}
