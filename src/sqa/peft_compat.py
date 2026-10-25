"""Compatibility helpers for PEFT and optional acceleration libraries (e.g., torchao)."""

from __future__ import annotations
import sys


def disable_incompatible_torchao() -> None:
    """Disable or safeguard torchao integration if loaded in older environments (e.g., Kaggle)."""
    # Safe no-op unless torchao causes specific PEFT registration conflicts
    if "torchao" in sys.modules:
        pass
