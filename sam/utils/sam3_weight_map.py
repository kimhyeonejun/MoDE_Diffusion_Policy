"""SAM3 weight map computation utilities for reconstruction loss."""
from __future__ import annotations

from typing import Optional
import torch
from PIL import Image
import numpy as np

from sam.utils.sam3_segmentation import segment_with_sam3_text_prompts


# Global SAM3 processor cache for lazy initialization
_SAM3_LOSS_PROCESSOR = None


def get_sam3_processor(device: str | torch.device, logger=None):
    """
    Lazy-initialize a SAM3 processor for computing masks.
    
    Args:
        device: Device to load SAM3 model on
        logger: Optional logger for logging initialization message
    
    Returns:
        SAM3Processor instance
    """
    global _SAM3_LOSS_PROCESSOR
    if _SAM3_LOSS_PROCESSOR is not None:
        return _SAM3_LOSS_PROCESSOR

    from sam3.model_builder import build_sam3_image_model  # type: ignore[import-not-found]
    from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[import-not-found]

    # Default to a permissive threshold; we'll override per-call.
    #
    # NOTE: sam3.model_builder._setup_device_and_mode() only checks `device == "cuda"` (string).
    # If we pass a torch.device (or "cuda:0"), the model may stay on CPU -> CUDA input vs CPU weights crash.
    if isinstance(device, torch.device):
        device_str = "cuda" if device.type == "cuda" else "cpu"
    else:
        # Normalize "cuda:0" -> "cuda" for SAM3 internals
        device_str = "cuda" if str(device).startswith("cuda") else "cpu"

    sam3_model = build_sam3_image_model(device=device_str, eval_mode=True)
    _SAM3_LOSS_PROCESSOR = Sam3Processor(sam3_model, device=device_str, confidence_threshold=0.05)
    if logger is not None:
        logger.info("[SAM3 recon loss] Initialized SAM3 processor")
    return _SAM3_LOSS_PROCESSOR


def tensor01_to_pil(x01_chw: torch.Tensor) -> Image.Image:
    """
    Convert a single image tensor in [0,1], shape (C,H,W) to a PIL RGB image.
    
    Args:
        x01_chw: Image tensor in [0,1] range, shape (C, H, W)
    
    Returns:
        PIL Image in RGB format
    """
    x = x01_chw.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu()
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x_hwc = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x_hwc)


def compute_weight_map(
    gt01_btchw: torch.Tensor,
    sam3_processor,
    conf_thr: float,
    prompts: list[str],
    thresholds: list[float],
    alpha: float,
) -> torch.Tensor:
    """
    Compute weight maps for batch of images using SAM3 segmentation.
    Note: segment_with_sam3_text_prompts processes single images, so we process batch sequentially
    but use vectorized tensor operations for mask processing.
    
    Args:
        gt01_btchw: Ground truth images in [0,1] range, shape (B, T, C, H, W)
        sam3_processor: SAM3Processor instance
        conf_thr: Confidence threshold for SAM3 segmentation
        prompts: List of text prompts for segmentation
        thresholds: List of thresholds to try for each prompt
        alpha: Weight coefficient for SAM3 mask (alpha in loss formula)
    
    Returns:
        Weight maps tensor, shape (B, T, 1, H, W)
    """
    # Use first timestep only for SAM3 (B, T, C, H, W) -> list of PIL
    b, t, c, h, w = gt01_btchw.shape
    
    # Convert batch to PIL images
    img_pil_list = [tensor01_to_pil(gt01_btchw[bi, 0]) for bi in range(b)]
    
    # Process all images - segment_with_sam3_text_prompts handles single images
    sam3_processor.confidence_threshold = conf_thr
    
    # Pre-allocate weight maps tensor for batch
    weight_maps = torch.zeros((b, t, 1, h, w), dtype=torch.float32, device=gt01_btchw.device)
    
    # Process each image (SAM3 API requires per-image processing)
    # But we vectorize the mask processing and weight map creation
    # Use silent=True to suppress verbose output during training
    for bi in range(b):
        # IMPORTANT: Lightning may run training under bf16 autocast, but SAM3 internals use fp32 weights.
        # Some SAM3 ops are executed outside autocast, which can lead to:
        #   "Input type (CUDABFloat16Type) and weight type (torch.FloatTensor) should be the same"
        # To avoid dtype mismatches, run SAM3 mask inference with autocast disabled (fp32).
        if gt01_btchw.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                out = segment_with_sam3_text_prompts(
                    sam3_processor,
                    img_pil_list[bi],
                    prompts=prompts,
                    thresholds=thresholds,
                    label=f"sam3mask[b={bi}]",
                    silent=True,
                )
        else:
            out = segment_with_sam3_text_prompts(
                sam3_processor,
                img_pil_list[bi],
                prompts=prompts,
                thresholds=thresholds,
                label=f"sam3mask[b={bi}]",
                silent=True,
            )
        masks = out["masks"]
        if torch.is_tensor(masks) and masks.numel() > 0 and masks.shape[0] > 0:
            union = masks.any(dim=0)  # (1,H,W) bool
            m = union.squeeze(0).to(dtype=torch.float32, device=gt01_btchw.device)
        else:
            m = torch.zeros((h, w), dtype=torch.float32, device=gt01_btchw.device)
        # Broadcast to (T,1,H,W) and store directly in pre-allocated tensor.
        # Clamp so masked pixels are at most 1.0 (i.e., values >= 1 -> 1), while zeros stay zero.
        wm = (alpha * m).clamp(0.0, 1.0)
        weight_maps[bi] = wm.view(1, 1, h, w).expand(t, 1, h, w)
    
    # (B,T,1,H,W)
    return weight_maps
