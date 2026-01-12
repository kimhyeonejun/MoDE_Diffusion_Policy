"""
Offline SAM3 mask/weight-map precompute helpers.

We store per-sample binary union masks (H,W) for each view, keyed by (lang_text, idx).
This avoids running SAM3 during training (major speedup).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Optional

import torch

View = Literal["static", "gripper"]


def make_sample_key(lang_text: str, idx: int) -> str:
    """
    Build a stable key that is unique across tasks even if `idx` collides across
    concatenated task datasets.
    """
    s = f"{str(lang_text)}|{int(idx)}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def mask_path(precompute_dir: str | Path, *, view: View, key: str) -> Path:
    p = Path(precompute_dir)
    return p / view / f"{key}.pt"


@torch.no_grad()
def save_mask_u8(mask_hw: torch.Tensor, path: str | Path) -> None:
    """
    Save a single mask (H,W) as uint8 (0/1) tensor.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = mask_hw.detach().to(device="cpu")
    if m.dtype != torch.uint8:
        m = m.to(torch.uint8)
    torch.save({"mask_u8": m}, path)


_MASK_CACHE: dict[tuple[str, View, str], torch.Tensor] = {}


@torch.no_grad()
def load_mask_u8(
    precompute_dir: str | Path,
    *,
    view: View,
    key: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Load mask (H,W) as float32 tensor on `device` with values {0,1}, or None if missing.
    Uses a simple process-local cache.
    """
    cache_key = (str(precompute_dir), view, key)
    hit = _MASK_CACHE.get(cache_key)
    if hit is not None:
        return hit.to(device=device)

    p = mask_path(precompute_dir, view=view, key=key)
    if not p.exists():
        return None
    obj = torch.load(p, map_location="cpu")
    m = obj["mask_u8"]
    # store cache on CPU to reduce VRAM usage
    _MASK_CACHE[cache_key] = m
    return m.to(device=device).float()


@torch.no_grad()
def build_weight_map_from_mask(
    mask_hw: torch.Tensor,
    *,
    b: int,
    t: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Expand (H,W) mask to (B,T,1,H,W) weight map.
    """
    m = mask_hw.to(device=device, dtype=dtype)
    if m.dim() != 2:
        raise ValueError(f"mask_hw must be (H,W), got shape={tuple(m.shape)}")
    return m[None, None, None].expand(int(b), int(t), 1, m.shape[0], m.shape[1])


