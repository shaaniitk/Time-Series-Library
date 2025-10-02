"""Utility helpers for lightweight memory diagnostics during training.

This module centralises CPU and GPU memory snapshot collection so that training
scripts can emit consistent, structured logs without duplicating guard logic
around optional dependencies such as CUDA or psutil. The implementation keeps
the runtime overhead low by avoiding synchronisation calls unless explicitly
requested by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableSequence, Optional, Union

try:  # pragma: no cover - psutil is optional at runtime
    import psutil
except ImportError:  # pragma: no cover - graceful degradation when psutil missing
    psutil = None  # type: ignore[assignment]

try:
    import resource
except ImportError:  # pragma: no cover - Windows compatibility path
    resource = None  # type: ignore[assignment]

import torch


def _bytes_to_megabytes(value: float) -> float:
    """Convert raw byte counts into megabytes for human-friendly logging."""

    return float(value) / (1024.0 ** 2)


@dataclass
class MemorySnapshot:
    """Structured view of memory statistics at a given point in time."""

    stage: str
    timestamp: float
    cpu_rss_mb: float
    cpu_vms_mb: Optional[float]
    cuda_allocated_mb: Optional[float]
    cuda_reserved_mb: Optional[float]
    cuda_max_allocated_mb: Optional[float]
    cuda_max_reserved_mb: Optional[float]
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable dictionary representation for logging."""

        payload: Dict[str, Any] = {
            "stage": self.stage,
            "cpu_rss_mb": round(self.cpu_rss_mb, 3),
            "cpu_vms_mb": round(self.cpu_vms_mb, 3) if self.cpu_vms_mb is not None else None,
            "cuda_allocated_mb": round(self.cuda_allocated_mb, 3) if self.cuda_allocated_mb is not None else None,
            "cuda_reserved_mb": round(self.cuda_reserved_mb, 3) if self.cuda_reserved_mb is not None else None,
            "cuda_max_allocated_mb": round(self.cuda_max_allocated_mb, 3)
            if self.cuda_max_allocated_mb is not None
            else None,
            "cuda_max_reserved_mb": round(self.cuda_max_reserved_mb, 3)
            if self.cuda_max_reserved_mb is not None
            else None,
            "timestamp": self.timestamp,
        }
        if self.extras:
            payload["extras"] = self.extras
        return payload


class MemoryDiagnostics:
    """Capture and log memory statistics across CPU and optional CUDA devices."""

    def __init__(
        self,
        logger_instance: Optional[logging.Logger] = None,
        *,
        keep_last: int = 256,
        synchronize_cuda: bool = False,
    ) -> None:
        self._log = logger_instance or logging.getLogger("memory.diagnostics")
        self._keep_last = max(1, keep_last)
        self._synchronize_cuda = synchronize_cuda
        self._history: MutableSequence[MemorySnapshot] = []

    @property
    def history(self) -> Iterable[MemorySnapshot]:
        """Expose the recent snapshot history for downstream inspection."""

        return tuple(self._history)

    def history_as_dicts(self) -> List[Dict[str, Any]]:
        """Return the stored snapshot history as serialisable dictionaries."""

        return [snapshot.to_dict() for snapshot in self._history]

    def dump_history(self, destination: Union[str, Path]) -> Optional[Path]:
        """Persist the current snapshot history to ``destination`` as JSON.

        Parameters
        ----------
        destination:
            Target file path that will receive the JSON dump. Parent directories
            are created automatically.

        Returns
        -------
        pathlib.Path | None
            Resolved path to the written file when successful, otherwise ``None``
            if an error occurred while writing.
        """

        path = Path(destination)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(self.history_as_dicts(), handle, indent=2)
            return path
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log.warning("Failed to dump memory diagnostics history to %s: %s", path, exc)
            return None

    def snapshot(self, stage: str, extras: Optional[Mapping[str, Any]] = None) -> MemorySnapshot:
        """Capture a memory snapshot and emit a structured log entry.

        Parameters
        ----------
        stage:
            Descriptive label for the current execution point (e.g. "data_load").
        extras:
            Optional metadata to enrich the log output with contextual details
            such as batch size or epoch counters.
        """

        if self._synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        cpu_rss_mb, cpu_vms_mb = self._capture_cpu_memory()
        cuda_stats = self._capture_cuda_memory()

        snapshot = MemorySnapshot(
            stage=stage,
            timestamp=time.time(),
            cpu_rss_mb=cpu_rss_mb,
            cpu_vms_mb=cpu_vms_mb,
            cuda_allocated_mb=cuda_stats.get("allocated"),
            cuda_reserved_mb=cuda_stats.get("reserved"),
            cuda_max_allocated_mb=cuda_stats.get("max_allocated"),
            cuda_max_reserved_mb=cuda_stats.get("max_reserved"),
            extras=dict(extras or {}),
        )

        self._store_snapshot(snapshot)
        self._log.info("Memory snapshot | %s", snapshot.to_dict())
        return snapshot

    def reset_cuda_peaks(self) -> None:
        """Reset CUDA peak statistics if CUDA is enabled."""

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def log_dataloader(self, name: str, dataloader: Any) -> None:
        """Log basic dataloader metadata to contextualise memory snapshots."""

        try:
            dataset = getattr(dataloader, "dataset", None)
            batch_size = getattr(dataloader, "batch_size", None)
            length = len(dataloader) if hasattr(dataloader, "__len__") else None
            message = {
                "name": name,
                "batch_size": batch_size,
                "steps": length,
                "dataset_len": len(dataset) if dataset is not None and hasattr(dataset, "__len__") else None,
                "num_workers": getattr(dataloader, "num_workers", None),
                "pin_memory": getattr(dataloader, "pin_memory", None),
            }
            self._log.info("Dataloader metadata | %s", message)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log.warning("Failed to log dataloader metadata for %s: %s", name, exc)

    def _capture_cpu_memory(self) -> tuple[float, Optional[float]]:
        """Collect CPU memory information using resource or psutil fallbacks."""

        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss_bytes = float(usage.ru_maxrss)
            if sys.platform == "darwin":
                rss_bytes *= 1024.0
            return _bytes_to_megabytes(rss_bytes), None

        if psutil is not None:  # pragma: no cover - requires psutil installation
            process = psutil.Process()
            mem_info = process.memory_info()
            rss_mb = _bytes_to_megabytes(float(mem_info.rss))
            vms_mb = _bytes_to_megabytes(float(mem_info.vms))
            return rss_mb, vms_mb

        return 0.0, None

    def _capture_cuda_memory(self) -> Dict[str, Optional[float]]:
        """Collect CUDA memory statistics when CUDA is available."""

        if not torch.cuda.is_available():
            return {"allocated": None, "reserved": None, "max_allocated": None, "max_reserved": None}

        allocated = _bytes_to_megabytes(float(torch.cuda.memory_allocated()))
        reserved = _bytes_to_megabytes(float(torch.cuda.memory_reserved()))
        max_allocated = _bytes_to_megabytes(float(torch.cuda.max_memory_allocated()))
        max_reserved = _bytes_to_megabytes(float(torch.cuda.max_memory_reserved()))
        return {
            "allocated": allocated,
            "reserved": reserved,
            "max_allocated": max_allocated,
            "max_reserved": max_reserved,
        }

    def _store_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Maintain a bounded history of snapshots for later inspection."""

        self._history.append(snapshot)
        if len(self._history) > self._keep_last:
            del self._history[0 : len(self._history) - self._keep_last]
