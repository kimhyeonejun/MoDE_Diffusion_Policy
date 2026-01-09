from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _get_model_param_device(model: torch.nn.Module, requested_device: str | torch.device) -> torch.device:
    try:
        model_param_device = next(model.parameters()).device
    except StopIteration:
        model_param_device = torch.device(requested_device)

    if isinstance(requested_device, str):
        requested_device = torch.device(requested_device)

    # If user requests cuda but model is on cpu, honor model device (avoid mismatch)
    if requested_device.type == "cuda" and model_param_device.type != "cuda":
        return model_param_device
    return model_param_device


def compress_image_with_msillm(
    msillm_model: torch.nn.Module,
    image_tensor: torch.Tensor,
    resize_to_64_multiple: bool = True,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Any, torch.Tensor, Tuple[int, ...]]:
    """
    Compress + decompress an image tensor using MS-ILLM.

    Returns:
        compressed: opaque MS-ILLM compressed object
        recon: reconstructed image tensor in [0, 1]
        original_shape: original tensor shape tuple (for debugging)
    """
    if image_tensor is None:
        raise ValueError("image_tensor is None")

    original_shape = tuple(image_tensor.shape)

    # Normalize tensor shape to [B, T, C, H, W]
    if image_tensor.dim() == 3:  # [C, H, W]
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        squeeze_back = True
    elif image_tensor.dim() == 4:  # [B, C, H, W]
        image_tensor = image_tensor.unsqueeze(1)
        squeeze_back = False
    elif image_tensor.dim() == 5:  # [B, T, C, H, W]
        squeeze_back = False
    else:
        raise ValueError(f"Unsupported tensor shape: {image_tensor.shape}")

    effective_device = _get_model_param_device(msillm_model, device)
    image_tensor = image_tensor.clamp(0.0, 1.0).to(effective_device)

    b, t, c, h, w = image_tensor.shape
    image_bt = image_tensor.reshape(b * t, c, h, w)

    # Resize to multiple of 64 if needed (required by many HiFiC-style models)
    resize_needed = False
    if resize_to_64_multiple:
        factor = 64
        if h % factor != 0 or w % factor != 0:
            new_h = ((h + factor - 1) // factor) * factor
            new_w = ((w + factor - 1) // factor) * factor
            image_bt_resized = F.interpolate(
                image_bt, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            resize_needed = True
        else:
            image_bt_resized = image_bt
    else:
        image_bt_resized = image_bt

    force_cpu = effective_device.type == "cpu"
    with torch.no_grad():
        compressed = msillm_model.compress(image_bt_resized, force_cpu=force_cpu)
        recon_resized = msillm_model.decompress(compressed, force_cpu=force_cpu).clamp(0.0, 1.0)

    if resize_needed:
        recon = F.interpolate(recon_resized, size=(h, w), mode="bilinear", align_corners=False)
    else:
        recon = recon_resized

    recon = recon.reshape(b, t, c, h, w)
    if squeeze_back:
        recon = recon.squeeze(0).squeeze(0)  # [C, H, W]

    return compressed, recon, original_shape


def compress_numpy_image_with_msillm(
    msillm_model: torch.nn.Module,
    image_np: np.ndarray,
    resize_to_64_multiple: bool = True,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Compress + reconstruct a numpy image using MS-ILLM.

    Args:
        image_np: uint8 (or float) numpy image, shape [H, W, C] or [H, W]

    Returns:
        compressed, recon_np(uint8, HxWxC or HxW)
    """
    if image_np.dtype != np.uint8:
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

    # Avoid negative strides from rot90/slices
    image_np = np.ascontiguousarray(image_np)

    if image_np.ndim == 2 or (image_np.ndim == 3 and image_np.shape[2] == 1):
        if image_np.ndim == 3:
            image_np = image_np.squeeze(2)
        image_tensor = torch.from_numpy(image_np).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # [1, H, W]
    else:
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0  # [C, H, W]

    compressed, recon_tensor, _ = compress_image_with_msillm(
        msillm_model, image_tensor, resize_to_64_multiple=resize_to_64_multiple, device=device
    )

    if recon_tensor.dim() == 2:
        recon_np = (recon_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    else:
        recon_np = (recon_tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        if recon_np.ndim == 3 and recon_np.shape[2] == 1:
            recon_np = recon_np.squeeze(2)

    return compressed, recon_np


