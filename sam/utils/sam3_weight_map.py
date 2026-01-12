"""SAM3 weight map computation utilities for reconstruction loss."""
from __future__ import annotations

from typing import Optional
import torch
from PIL import Image
import numpy as np

from sam.utils.sam3_segmentation import segment_with_sam3_text_prompts
from sam.utils.prompts import build_prompt_candidates


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
    # Ensure SAM3 weights stay in fp32 even when the outer training loop uses bf16 autocast.
    sam3_model = sam3_model.to(dtype=torch.float32)
    # Explicitly freeze SAM3 model parameters (eval_mode sets model.eval() but doesn't freeze gradients)
    for param in sam3_model.parameters():
        param.requires_grad = False
    sam3_model.eval()  # Ensure eval mode is set
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
    # Convert to float32 first to handle bfloat16 inputs properly
    # This ensures all operations work correctly regardless of input dtype
    x = x01_chw.detach().to(dtype=torch.float32).clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu()
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
    
    # Convert batch to PIL images (still a loop, but cheap compared to SAM3 forward passes)
    img_pil_list = [tensor01_to_pil(gt01_btchw[bi, 0]) for bi in range(b)]

    # -------------------------------------------------------------------------
    # Fast path: batch image backbone + batch grounding
    #
    # SAM3Processor has `set_image_batch`, but its `set_text_prompt` wrapper assumes a
    # single image (uses FindStage img_ids=[0] and state["original_height"/"width"]).
    # We bypass that wrapper and call `model.forward_grounding` directly in batch.
    # -------------------------------------------------------------------------
    from sam3.model.data_misc import FindStage, interpolate  # type: ignore[import-not-found]

    device = gt01_btchw.device
    model = sam3_processor.model

    # 2) Prepare per-image IDs for batch grounding.
    img_ids = torch.arange(b, device=device, dtype=torch.long)
    # We run one text prompt at a time (single prompt embedding), and point all images to text_id=0.
    text_ids = torch.zeros(b, device=device, dtype=torch.long)

    # 3) Accumulate union masks across prompts (per-image).
    union_bt_hw = torch.zeros((b, h, w), dtype=torch.bool, device=device)

    # local import to avoid a hard dependency when running without SAM3 installed
    from contextlib import nullcontext

    # IMPORTANT: Lightning may run training under bf16 autocast, but SAM3 internals use fp32 weights.
    # Run SAM3 inference with autocast disabled to avoid dtype mismatches.
    autocast_ctx = (
        torch.autocast(device_type="cuda", enabled=False) if gt01_btchw.is_cuda else nullcontext()
    )

    with autocast_ctx:
        # 1) Precompute image features for the whole batch once (keep in fp32; Lightning may run bf16 autocast).
        state = sam3_processor.set_image_batch(img_pil_list)
        backbone_out = state["backbone_out"]

        for prompt in prompts:
            # Precompute text features once per prompt.
            # NOTE: We pass a single prompt string; `text_ids` broadcasts it to all images.
            # NOTE: Sam3Processor uses a string device internally ("cuda"/"cpu").
            text_outputs = model.backbone.forward_text([prompt], device=str(sam3_processor.device))
            backbone_out.update(text_outputs)

            # Select best threshold per image (matches notebook preference: higher thr, then more masks).
            best_thr = torch.full((b,), float("-inf"), device=device, dtype=torch.float32)
            best_num = torch.full((b,), -1, device=device, dtype=torch.long)
            best_union = torch.zeros((b, h, w), dtype=torch.bool, device=device)

            for thr in thresholds:
                # Build minimal FindStage for text-only grounding.
                find_input = FindStage(
                    img_ids=img_ids,
                    text_ids=text_ids,
                    input_boxes=None,
                    input_boxes_mask=None,
                    input_boxes_label=None,
                    input_points=None,
                    input_points_mask=None,
                )
                geometric_prompt = model._get_dummy_prompt(num_prompts=b)

                outputs = model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    geometric_prompt=geometric_prompt,
                    find_target=None,
                )

                # Compute per-image keep mask.
                logits = outputs["pred_logits"]  # (B, Q, 1) or (B, Q)
                if logits.dim() == 3:
                    logits = logits.squeeze(-1)
                probs = logits.sigmoid()
                presence = outputs.get("presence_logit_dec", None)
                if presence is not None:
                    # Some variants return presence with extra trailing dims (e.g., (B, K)).
                    # Reduce to a single scalar per image, then broadcast over masks.
                    p = presence.sigmoid()
                    if p.dim() > 1:
                        p = p.reshape(p.shape[0], -1).amax(dim=-1)
                    probs = probs * p.unsqueeze(1)
                keep = probs > float(thr)  # (B, Q) bool
                num_masks = keep.sum(dim=1)  # (B,)

                # Upsample masks to (H,W) and compute per-image union of kept masks.
                masks = outputs["pred_masks"]  # (B, Q, h', w')
                bq = masks.shape[0] * masks.shape[1]
                masks01 = interpolate(
                    masks.reshape(bq, 1, masks.shape[-2], masks.shape[-1]),
                    (h, w),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()
                masks_bin = masks01.reshape(b, masks.shape[1], h, w) > 0.5
                thr_union = (masks_bin & keep[:, :, None, None]).any(dim=1)  # (B,H,W)

                thr_f = torch.full((b,), float(thr), device=device, dtype=torch.float32)
                better = (thr_f > best_thr) | ((thr_f == best_thr) & (num_masks > best_num))
                if better.any():
                    best_thr = torch.where(better, thr_f, best_thr)
                    best_num = torch.where(better, num_masks, best_num)
                    best_union = torch.where(better[:, None, None], thr_union, best_union)

            union_bt_hw |= best_union

    # Build weight maps (B,T,1,H,W)
    weight_maps = torch.zeros((b, t, 1, h, w), dtype=torch.float32, device=device)
    wm = (alpha * union_bt_hw.to(dtype=torch.float32)).clamp(0.0, 1.0)  # (B,H,W)
    weight_maps[:] = wm[:, None, None, :, :].expand(b, t, 1, h, w)
    
    # (B,T,1,H,W)
    return weight_maps


def compute_weight_map_from_lang_text_batch(
    gt01_btchw: torch.Tensor,
    sam3_processor,
    conf_thr: float,
    lang_text_batch,
    thresholds: list[float],
    alpha: float,
) -> torch.Tensor:
    """
    Compute SAM3 weight maps using **per-sample** language instructions.

    This mirrors SAM3's own training data path:
    - Build a `find_text_batch` (unique prompt strings)
    - Build `img_ids` / `text_ids` mappings (per query)
    - Run batched grounding to get masks

    We then aggregate masks per image by OR-ing the best-per-threshold result per prompt.
    """
    b, t, c, h, w = gt01_btchw.shape
    device = gt01_btchw.device

    # Normalize lang_text_batch to List[str] length B.
    texts: list[str]
    if lang_text_batch is None or torch.is_tensor(lang_text_batch):
        texts = [""] * b
    elif isinstance(lang_text_batch, str):
        texts = [lang_text_batch] * b
    elif isinstance(lang_text_batch, (list, tuple)):
        texts = [str(x) for x in lang_text_batch]
        if len(texts) == 1 and b > 1:
            texts = texts * b
        if len(texts) != b:
            # Fallback: use the first instruction for all images if sizes mismatch.
            texts = [texts[0] if len(texts) else ""] * b
    else:
        texts = [str(lang_text_batch)] * b

    # Build prompt candidates per image, then flatten into "queries":
    # one query = (image_idx, prompt_string).
    query_img_ids: list[int] = []
    query_prompts: list[str] = []
    for i, instr in enumerate(texts):
        cands = build_prompt_candidates(instr or "")
        if not cands:
            cands = ["object"]
        for p in cands:
            query_img_ids.append(i)
            query_prompts.append(p)

    # If we somehow have no queries, return zeros.
    if not query_prompts:
        return torch.zeros((b, t, 1, h, w), dtype=torch.float32, device=device)

    # Build `find_text_batch` (unique prompts) and `text_ids` (per query).
    find_text_batch: list[str] = []
    prompt_to_id: dict[str, int] = {}
    text_ids_list: list[int] = []
    for p in query_prompts:
        pid = prompt_to_id.get(p)
        if pid is None:
            pid = len(find_text_batch)
            find_text_batch.append(p)
            prompt_to_id[p] = pid
        text_ids_list.append(pid)

    # Convert batch images to PIL once (still a loop, but cheaper than per-image SAM3 forwards).
    img_pil_list = [tensor01_to_pil(gt01_btchw[bi, 0]) for bi in range(b)]

    from sam3.model.data_misc import FindStage, interpolate  # type: ignore[import-not-found]
    from contextlib import nullcontext

    model = sam3_processor.model

    # 3) Prepare query mappings.
    q = len(query_prompts)
    img_ids_q = torch.tensor(query_img_ids, device=device, dtype=torch.long)
    text_ids_q = torch.tensor(text_ids_list, device=device, dtype=torch.long)

    # 4) For each query (image,prompt), select best threshold (prefers higher thr, then more masks),
    # then aggregate per-image by OR across that image's prompts.
    best_thr = torch.full((q,), float("-inf"), device=device, dtype=torch.float32)
    best_num = torch.full((q,), -1, device=device, dtype=torch.long)
    best_union = torch.zeros((q, h, w), dtype=torch.bool, device=device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", enabled=False) if gt01_btchw.is_cuda else nullcontext()
    )

    with torch.no_grad(), autocast_ctx:
        # 1) Precompute image features for the whole batch once (keep in fp32; Lightning may run bf16 autocast).
        state = sam3_processor.set_image_batch(img_pil_list)
        backbone_out = state["backbone_out"]

        # 2) Precompute text features for all unique prompt strings once.
        text_outputs = model.backbone.forward_text(find_text_batch, device=str(sam3_processor.device))
        backbone_out.update(text_outputs)

        for thr in thresholds:
            find_input = FindStage(
                img_ids=img_ids_q,
                text_ids=text_ids_q,
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            geometric_prompt = model._get_dummy_prompt(num_prompts=q)

            outputs = model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                geometric_prompt=geometric_prompt,
                find_target=None,
            )
            
            # Reduce logits to per-(query,mask) scores (Q, N).
            logits = outputs["pred_logits"]
            if logits.dim() == 3 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            elif logits.dim() > 2:
                logits = logits.reshape(logits.shape[0], logits.shape[1], -1).amax(dim=-1)
            
            probs = logits.sigmoid()
            presence = outputs.get("presence_logit_dec", None)
            if presence is not None:
                p = presence.sigmoid()
                if p.dim() > 1:
                    p = p.reshape(p.shape[0], -1).amax(dim=-1)
                probs = probs * p.unsqueeze(1)
                
            masks = outputs["pred_masks"]  # (Q, N, h', w')
            keep = probs > float(thr)  # (Q, N)
            num_masks = keep.sum(dim=1)  # (Q,)
            
            # Free memory: delete outputs immediately after extraction
            del outputs, logits, presence
            
            # Interpolate masks to (h, w) and compute union
            # Process in chunks to reduce peak memory if needed
            qn = masks.shape[0] * masks.shape[1]
            masks01 = interpolate(
                masks.reshape(qn, 1, masks.shape[-2], masks.shape[-1]),
                (h, w),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()
            masks_bin = masks01.reshape(q, masks.shape[1], h, w) > 0.5
            thr_union = (masks_bin & keep[:, :, None, None]).any(dim=1)  # (Q,H,W)
            
            # Free memory: delete intermediate tensors immediately
            del masks, masks01, masks_bin

            thr_f = torch.full((q,), float(thr), device=device, dtype=torch.float32)
            better = (thr_f > best_thr) | ((thr_f == best_thr) & (num_masks > best_num))
            if better.any():
                best_thr = torch.where(better, thr_f, best_thr)
                best_num = torch.where(better, num_masks, best_num)
                best_union = torch.where(better[:, None, None], thr_union, best_union)
            
            # Free memory: delete threshold-specific results
            del thr_union, thr_f, better

    # Aggregate to per-image union mask.
    union_bt_hw = torch.zeros((b, h, w), dtype=torch.bool, device=device)
    for i in range(b):
        sel = img_ids_q == i
        if sel.any():
            union_bt_hw[i] = best_union[sel].any(dim=0)

    weight_maps = torch.zeros((b, t, 1, h, w), dtype=torch.float32, device=device)
    wm = (alpha * union_bt_hw.to(dtype=torch.float32)).clamp(0.0, 1.0)
    weight_maps[:] = wm[:, None, None, :, :].expand(b, t, 1, h, w)
    return weight_maps
