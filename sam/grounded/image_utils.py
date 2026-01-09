from __future__ import annotations

from typing import Any

import numpy as np


def extract_image_from_batch(
    batch: dict,
    image_key: str = "rgb_static",
    batch_idx: int = 0,
    time_idx: int = 0,
) -> np.ndarray:
    """
    Extract an image from a LIBERO-style batch and return HxWxC uint8.

    The notebook had to handle multiple layouts; this keeps that logic reusable.
    """
    img = batch[image_key]

    # Common shapes:
    # - (B, T, C, H, W)
    # - (B, C, H, W)
    # - (C, H, W)
    img_np = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)

    if img_np.ndim == 5:  # (B, T, C, H, W)
        img_np = img_np[batch_idx, time_idx]
    elif img_np.ndim == 4:  # (B, C, H, W)
        img_np = img_np[batch_idx]
    elif img_np.ndim == 3:  # (C, H, W) or (H, W, C)
        pass
    else:
        raise ValueError(f"Unsupported image tensor shape: {img_np.shape}")

    # Convert (C, H, W) -> (H, W, C) if needed
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3) and img_np.shape[-1] not in (1, 3, 4):
        img_np = np.transpose(img_np, (1, 2, 0))

    # Convert to uint8
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

    # The notebook rotates by 180 degrees; keep it consistent for callers that want it.
    img_np = np.rot90(img_np, k=2, axes=(0, 1))

    # Ensure contiguous (avoid negative strides from rotations/slices)
    img_np = np.ascontiguousarray(img_np)

    return img_np


def numpy_to_pil(img_np: np.ndarray):
    """Convert numpy HxWxC uint8 (or float) to a PIL Image (RGB)."""
    from PIL import Image

    import cv2

    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

    # Ensure RGB format
    if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1):
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.ndim == 3 and img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    return Image.fromarray(img_np)


def require_globals(globs: dict, *names: str) -> None:
    """Small helper for notebooks/scripts when you still rely on globals."""
    missing = [n for n in names if n not in globs]
    if missing:
        raise KeyError(f"Missing required globals: {missing}")


