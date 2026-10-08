"""Model resolver helper for local paths vs Hugging Face hub IDs."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_model(model_name: str) -> str:
    """Return model_name as string. If local path exists, returns resolved path."""
    if not model_name:
        return model_name
    p = Path(model_name)
    if p.exists():
        return str(p.resolve())
    return str(model_name)
