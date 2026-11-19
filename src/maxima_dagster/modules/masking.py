"""Invalid-pixel masking shared by calibration and integration."""

from __future__ import annotations

import numpy as np


# Eiger writes 2**32 - 1 into dead, hot and gap pixels; pyFAI's detector mask covers only the gaps
UINT32_SENTINEL = np.iinfo(np.uint32).max


def effective_mask(image: np.ndarray, factory_mask: np.ndarray | None = None) -> np.ndarray:
    """Combine the image's invalid counts with a detector mask supplied by the caller."""
    if image.ndim != 2 or image.size == 0 or image.dtype.kind not in "iuf":
        raise ValueError("Expected a nonempty 2D detector frame with numeric counts")
    invalid = ~np.isfinite(image) | (image < 0)
    bounds = np.finfo(image.dtype) if image.dtype.kind == "f" else np.iinfo(image.dtype)
    if bounds.max >= UINT32_SENTINEL:
        invalid |= image >= image.dtype.type(UINT32_SENTINEL)
    if factory_mask is not None:
        factory_mask = np.asarray(factory_mask, dtype=bool)
        if factory_mask.shape != image.shape:
            raise ValueError(f"Detector frame shape {image.shape} does not match factory mask shape {factory_mask.shape}")
        invalid |= factory_mask
    return invalid
