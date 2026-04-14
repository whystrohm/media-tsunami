"""Writes voice-fingerprint.json — full raw signals for debugging + research."""
from __future__ import annotations

import json
from pathlib import Path


def _jsonable(obj):
    """Recursively convert numpy arrays + ndarrays to plain Python for JSON."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_voice_fingerprint(fingerprint: dict, output_path: str | Path) -> Path:
    """Write the full fingerprint to disk as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _jsonable(fingerprint)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return path.resolve()
