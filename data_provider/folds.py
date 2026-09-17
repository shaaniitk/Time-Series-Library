"""Date-based walk-forward folds with a label purge.

Replaces positional 70/10/20 splits for studies whose protocol freezes calendar
boundaries.  Borders follow ``Dataset_Custom`` conventions: evaluation splits
start ``seq_len`` rows early so their first sample has full look-back, while the
training split is trimmed so at least ``purge_sessions`` rows separate its last
label from the first evaluation label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_end: str
    eval_start: str
    eval_end: str | None
    purge_sessions: int = 63
    is_locked_holdout: bool = False


# Development folds from AST_H01_FROZEN_PROTOCOL.md section 4.1 (provisional).
AST_H01_DEVELOPMENT_FOLDS = {
    "F1": FoldSpec("F1", "2007-12-31", "2008-04-01", "2010-12-31"),
    "F2": FoldSpec("F2", "2010-12-31", "2011-04-01", "2013-12-31"),
    "F3": FoldSpec("F3", "2013-12-31", "2014-04-01", "2016-12-31"),
    "F4": FoldSpec("F4", "2016-12-31", "2017-04-01", "2020-03-31"),
}

# Section 5.1: inspected exactly once per protocol version.
AST_H01_LOCKED_HOLDOUT = FoldSpec(
    "HOLDOUT", "2020-03-31", "2020-07-01", None, is_locked_holdout=True
)


class FoldContractError(ValueError):
    """Raised when fold boundaries cannot be honored on the given dates."""


def resolve_fold(name: str, unlock_holdout: bool = False) -> FoldSpec:
    if name == AST_H01_LOCKED_HOLDOUT.name:
        if not unlock_holdout:
            raise FoldContractError(
                "The locked holdout may only be opened with astro_unlock_holdout=True, "
                "and only once per frozen protocol version."
            )
        return AST_H01_LOCKED_HOLDOUT
    if name not in AST_H01_DEVELOPMENT_FOLDS:
        raise FoldContractError(
            f"Unknown fold {name!r}; expected one of "
            f"{sorted(AST_H01_DEVELOPMENT_FOLDS) + [AST_H01_LOCKED_HOLDOUT.name]}."
        )
    return AST_H01_DEVELOPMENT_FOLDS[name]


def fold_borders(
    dates: pd.Series, fold: FoldSpec, seq_len: int, pred_len: int
) -> tuple[list[int], list[int]]:
    """Return ``(border1s, border2s)`` for train/val/test in row positions.

    Validation and test share the evaluation block: selection uses validation
    only, and the test loader reports the same block for per-fold summaries.
    """

    timestamps = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"))
    if timestamps.hasnans:
        raise FoldContractError("Market dates contain NaT.")
    if len(timestamps) > 1 and np.any(np.diff(timestamps.asi8) <= 0):
        raise FoldContractError("Market dates must be strictly increasing.")
    if fold.purge_sessions < max(pred_len, 1):
        raise FoldContractError(
            f"purge_sessions={fold.purge_sessions} must be at least pred_len={pred_len}."
        )

    train_end = timestamps.searchsorted(pd.Timestamp(fold.train_end), side="right")
    eval_start = timestamps.searchsorted(pd.Timestamp(fold.eval_start), side="left")
    eval_end = (
        len(timestamps)
        if fold.eval_end is None
        else timestamps.searchsorted(pd.Timestamp(fold.eval_end), side="right")
    )

    # Trim training only as far as needed to leave the required session gap
    # between the last training label and the first evaluation label.
    purged_train_end = min(train_end, eval_start - fold.purge_sessions)
    if purged_train_end < seq_len + pred_len:
        raise FoldContractError(
            f"Fold {fold.name} leaves {purged_train_end} training rows after purging, "
            f"fewer than seq_len+pred_len={seq_len + pred_len}."
        )
    if eval_start - seq_len < 0:
        raise FoldContractError(
            f"Fold {fold.name} evaluation starts before seq_len={seq_len} rows of history exist."
        )
    if eval_end - eval_start < pred_len:
        raise FoldContractError(f"Fold {fold.name} evaluation block is empty.")

    eval_border1 = eval_start - seq_len
    border1s = [0, eval_border1, eval_border1]
    border2s = [purged_train_end, eval_end, eval_end]
    return border1s, border2s
