"""SAM3 weight map computation utilities for reconstruction loss."""
from __future__ import annotations

from typing import Optional
from collections import OrderedDict
import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
from sam.utils.prompts import build_prompt_candidates


"""
This module runs SAM3 during training to build a per-pixel weight map.

Performance notes:
- SAM3Processor defaults to resolution=1008, which is expensive for per-step training.
- Text encoding + grounding can also be expensive when prompt candidates are many.
"""

# Global SAM3 processor cache for lazy initialization.
# Keyed by (device_str, resolution, confidence_threshold, infer_dtype).
_SAM3_LOSS_PROCESSOR: dict[tuple[str, int, float, str], object] = {}

# Text feature cache to avoid repeated expensive `forward_text` calls across training steps.
# We cache on GPU to avoid CPU->GPU transfer overhead (0.24s saved per cache hit).
# GPU memory usage: ~19 prompts * ~(256+1024) dims * float32 = ~0.1MB (negligible).
_TEXT_FEAT_CACHE: "OrderedDict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
_TEXT_FEAT_CACHE_MAX = 4096  # max unique prompt strings to keep


def _get_text_outputs_cached(model, prompts: list[str], *, device: torch.device | str):
    """
    Return SAM3 text outputs for `prompts` using a per-prompt GPU cache.

    Cached tensors are stored on GPU to avoid CPU->GPU transfer overhead.
    Returns dict matching `SAM3VLBackbone.forward_text` outputs, padded/stacked to max length.
    """
    global _TEXT_FEAT_CACHE

    if isinstance(device, str):
        device_str = device
        device_t = torch.device(device)
    else:
        device_t = device
        device_str = str(device)
    
    model_dtype = next(model.parameters()).dtype
    
    # Fill cache for missing prompts
    missing = [p for p in prompts if p not in _TEXT_FEAT_CACHE]
    if missing:
        out = model.backbone.forward_text(missing, device=device_str)
        lf = out["language_features"].detach().to(dtype=model_dtype)
        lm = out["language_mask"].detach()
        le = out["language_embeds"].detach().to(dtype=model_dtype)

        for j, p in enumerate(missing):
            _TEXT_FEAT_CACHE[p] = (
                lf[:, j : j + 1].contiguous(),
                lm[j : j + 1].contiguous(),
                le[:, j : j + 1].contiguous(),
            )
            _TEXT_FEAT_CACHE.move_to_end(p)

        while len(_TEXT_FEAT_CACHE) > _TEXT_FEAT_CACHE_MAX:
            _TEXT_FEAT_CACHE.popitem(last=False)

    # Stack cached tensors, padding to max length
    # Use cached tensors as-is (they were already converted to model_dtype when cached)
    feats, masks, embeds = [], [], []
    max_L = 0
    for p in prompts:
        lf, lm, le = _TEXT_FEAT_CACHE[p]
        max_L = max(max_L, lf.shape[0])
        feats.append(lf)
        masks.append(lm)
        embeds.append(le)
    
    feats_pad, masks_pad, embeds_pad = [], [], []
    for lf, lm, le in zip(feats, masks, embeds):
        if lf.shape[0] < max_L:
            pad_L = max_L - lf.shape[0]
            lf = F.pad(lf, (0, 0, 0, 0, 0, pad_L), value=0.0)
            le = F.pad(le, (0, 0, 0, 0, 0, pad_L), value=0.0)
            lm = F.pad(lm, (0, pad_L), value=False)
        feats_pad.append(lf)
        masks_pad.append(lm)
        embeds_pad.append(le)

    language_features = torch.cat(feats_pad, dim=1)
    language_mask = torch.cat(masks_pad, dim=0)
    language_embeds = torch.cat(embeds_pad, dim=1)
    
    # Ensure correct device
    if language_features.device != device_t:
        language_features = language_features.to(device=device_t)
    if language_mask.device != device_t:
        language_mask = language_mask.to(device=device_t)
    if language_embeds.device != device_t:
        language_embeds = language_embeds.to(device=device_t)
    
    return {
        "language_features": language_features,
        "language_mask": language_mask,
        "language_embeds": language_embeds,
    }


def get_sam3_processor(
    device: str | torch.device,
    logger=None,
    *,
    resolution: int = 1008,
    confidence_threshold: float = 0.05,
    infer_dtype: str = "fp32",
):
    """
    Lazy-initialize a SAM3 processor for computing masks.
    
    Args:
        device: Device to load SAM3 model on
        logger: Optional logger for logging initialization message
    
    Returns:
        SAM3Processor instance
    """
    global _SAM3_LOSS_PROCESSOR

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

    # Normalize dtype option.
    infer_dtype = str(infer_dtype).lower().strip()
    if infer_dtype in ("bf16", "bfloat16"):
        infer_dtype = "bf16"
    elif infer_dtype in ("fp32", "float32", "32"):
        infer_dtype = "fp32"
    else:
        if logger is not None:
            logger.warning(
                f"[SAM3 recon loss] Unknown sam3_infer_dtype={infer_dtype!r}. Using fp32."
            )
        infer_dtype = "fp32"

    key = (device_str, int(resolution), float(confidence_threshold), infer_dtype)
    cached = _SAM3_LOSS_PROCESSOR.get(key, None)
    if cached is not None:
        return cached  # type: ignore[return-value]

    sam3_model = build_sam3_image_model(device=device_str, eval_mode=True)
    
    # Convert model to target dtype
    target_dtype = torch.bfloat16 if infer_dtype == "bf16" else torch.float32
    sam3_model = sam3_model.to(dtype=target_dtype)
    
    # Ensure all parameters and buffers are in target dtype
    if infer_dtype == "bf16":
        for param in sam3_model.parameters():
            if param.is_floating_point():
                param.data = param.data.to(dtype=torch.bfloat16)
        for buffer in sam3_model.buffers():
            if buffer.is_floating_point():
                buffer.data = buffer.data.to(dtype=torch.bfloat16)
    
    # Freeze parameters and set eval mode
    for param in sam3_model.parameters():
        param.requires_grad = False
    sam3_model.eval()
    
    # Verify model dtype
    if logger is not None:
        first_param_dtype = next(sam3_model.parameters()).dtype
        first_buffer_dtype = None
        for buf in sam3_model.buffers():
            if buf.is_floating_point():
                first_buffer_dtype = buf.dtype
                break
        logger.info(
            f"[SAM3 recon loss] Model dtype: param={first_param_dtype}, buffer={first_buffer_dtype} (expected: {target_dtype})"
        )
    proc = Sam3Processor(
        sam3_model,
        device=device_str,
        resolution=int(resolution),
        confidence_threshold=float(confidence_threshold),
    )
    _SAM3_LOSS_PROCESSOR[key] = proc
    if logger is not None:
        logger.info(
            f"[SAM3 recon loss] Initialized SAM3 processor (device={device_str}, resolution={int(resolution)}, conf_thr={float(confidence_threshold)}, dtype={infer_dtype})"
        )
    return proc


def _set_image_batch_backbone_out(img_pil_list: list[Image.Image], sam3_processor):
    """
    Equivalent to Sam3Processor.set_image_batch, but ensures image tensor dtype
    matches the SAM3 model's parameter dtype (fp32/bf16) to avoid dtype mismatch.
    """
    from torchvision.transforms import v2  # type: ignore

    model = sam3_processor.model
    model_dtype = next(model.parameters()).dtype
    device = torch.device(str(sam3_processor.device))

    images = [
        sam3_processor.transform(v2.functional.to_image(image).to(device))
        for image in img_pil_list
    ]
    images = torch.stack(images, dim=0).to(dtype=model_dtype)

    state = {
        "original_heights": [im.height for im in img_pil_list],
        "original_widths": [im.width for im in img_pil_list],
        "backbone_out": model.backbone.forward_image(images),
    }

    # Process SAM2 backbone features if inst_interactive_predictor exists
    if (
        model.inst_interactive_predictor is not None
        and "sam2_backbone_out" in state["backbone_out"]
    ):
        sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
        decoder = model.inst_interactive_predictor.model.sam_mask_decoder
        sam2_backbone_out["backbone_fpn"][0] = decoder.conv_s0(sam2_backbone_out["backbone_fpn"][0])
        sam2_backbone_out["backbone_fpn"][1] = decoder.conv_s1(sam2_backbone_out["backbone_fpn"][1])
    
    return state


def _set_image_batch_backbone_out_from_tensor01(
    img01_bchw: torch.Tensor, sam3_processor, *, orig_h: int, orig_w: int
):
    """
    Faster set_image_batch path that avoids PIL round-trips and Python per-image loops.

    Expects img01_bchw in [0,1], shape (B,C,H,W). Uses the same transform pipeline as
    Sam3Processor (Resize to resolution + Normalize(mean=0.5,std=0.5)).
    """
    model = sam3_processor.model
    model_dtype = next(model.parameters()).dtype
    device = torch.device(str(sam3_processor.device))

    # Ensure on correct device.
    # Keep float32 for resize/normalize; cast to model dtype afterwards.
    images = img01_bchw.detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    # ---- Fast path: replicate Sam3Processor.transform on GPU without torchvision v2 ----
    # Sam3Processor does:
    #   uint8(scale=True) -> Resize(res,res) -> float32(scale=True) -> Normalize(mean=0.5,std=0.5)
    # Which is equivalent (for our float [0,1] inputs) to:
    #   Resize(res,res) -> (x-0.5)/0.5  ==  x*2-1
    res = int(sam3_processor.resolution)
    if images.shape[-2:] != (res, res):
        # antialias is supported in recent torch versions; use when available.
        try:
            images = torch.nn.functional.interpolate(
                images, size=(res, res), mode="bilinear", align_corners=False, antialias=True
            )
        except TypeError:
            images = torch.nn.functional.interpolate(
                images, size=(res, res), mode="bilinear", align_corners=False
            )
    images = images.mul(2.0).sub(1.0)
    images = images.to(dtype=model_dtype)

    # Optional internal timing (CUDA events) for set_image breakdown.
    evt_pre0 = evt_pre1 = evt_fwd0 = evt_fwd1 = None
    if device.type == "cuda":
        evt_pre0 = torch.cuda.Event(enable_timing=True)
        evt_pre1 = torch.cuda.Event(enable_timing=True)
        evt_fwd0 = torch.cuda.Event(enable_timing=True)
        evt_fwd1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(device):
            evt_pre0.record()
            evt_pre1.record()

    state = {
        "original_heights": [int(orig_h)] * int(images.shape[0]),
        "original_widths": [int(orig_w)] * int(images.shape[0]),
    }
    if device.type == "cuda" and evt_fwd0 is not None:
        with torch.cuda.device(device):
            evt_fwd0.record()
    state["backbone_out"] = model.backbone.forward_image(images)
    if device.type == "cuda" and evt_fwd1 is not None:
        with torch.cuda.device(device):
            evt_fwd1.record()

    if device.type == "cuda" and evt_pre0 is not None:
        state["__set_image_evt"] = {
            "pre": (evt_pre0, evt_pre1),
            "fwd": (evt_fwd0, evt_fwd1),
        }

    # Process SAM2 backbone features if inst_interactive_predictor exists
    if (
        model.inst_interactive_predictor is not None
        and "sam2_backbone_out" in state["backbone_out"]
    ):
        sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
        decoder = model.inst_interactive_predictor.model.sam_mask_decoder
        sam2_backbone_out["backbone_fpn"][0] = decoder.conv_s0(sam2_backbone_out["backbone_fpn"][0])
        sam2_backbone_out["backbone_fpn"][1] = decoder.conv_s1(sam2_backbone_out["backbone_fpn"][1])

    return state


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


def compute_weight_map_from_lang_text_batch(
    gt01_btchw: torch.Tensor,
    sam3_processor,
    conf_thr: float,
    lang_text_batch,
    thresholds: list[float],
    alpha: float,
    *,
    max_prompts_per_image: int = 3,
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
    # LIBERO batches typically have lang_text as a list of strings (one per sample)
    if lang_text_batch is None:
        texts = [""] * b
    elif isinstance(lang_text_batch, str):
        texts = [lang_text_batch] * b
    elif isinstance(lang_text_batch, (list, tuple)):
        texts = [str(x) for x in lang_text_batch]
        if len(texts) != b:
            # Fallback: broadcast single string or use first if mismatch
            texts = (texts * b)[:b] if len(texts) == 1 else ([texts[0]] * b if texts else [""] * b)
    else:
        texts = [str(lang_text_batch)] * b

    # Build prompt candidates per image, then flatten into "queries":
    # one query = (image_idx, prompt_string).
    query_img_ids: list[int] = []
    query_prompts: list[str] = []
    max_prompts_per_image = int(max_prompts_per_image)
    for i, instr in enumerate(texts):
        cands = build_prompt_candidates(instr or "")
        if not cands:
            cands = ["object"]
        if max_prompts_per_image > 0:
            cands = cands[:max_prompts_per_image]
        for p in cands:
            query_img_ids.append(i)
            query_prompts.append(p)

    # If we somehow have no queries, return zeros.
    if not query_prompts:
        return torch.zeros((b, t, 1, h, w), dtype=torch.bfloat16, device=device)

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
    from sam3.model.data_misc import FindStage, interpolate  # type: ignore[import-not-found]
    from contextlib import nullcontext

    model = sam3_processor.model
    sam_device = torch.device(str(sam3_processor.device))
    model_dtype = next(model.parameters()).dtype

    # Prefer a tensor-only path (no PIL) for speed.
    img01_bchw = gt01_btchw[:, 0].to(device=sam_device, dtype=torch.float32)

    # 3) Prepare query mappings.
    q = len(query_prompts)
    img_ids_q = torch.tensor(query_img_ids, device=sam_device, dtype=torch.long)
    text_ids_q = torch.tensor(text_ids_list, device=sam_device, dtype=torch.long)

    # 4) For each query (image,prompt), select best threshold (prefers higher thr, then more masks),
    # then aggregate per-image by OR across that image's prompts.
    best_thr = torch.full((q,), float("-inf"), device=sam_device, dtype=torch.float32)
    best_num = torch.full((q,), -1, device=sam_device, dtype=torch.long)
    best_union = torch.zeros((q, h, w), dtype=torch.bool, device=sam_device)

    # Use autocast for bf16 models to handle dtype conversions automatically
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        if sam_device.type == "cuda" and model_dtype == torch.bfloat16
        else nullcontext()
    )

    import time

    with torch.no_grad(), autocast_ctx:
        # NOTE: time.time() around CUDA work is easily "lied to" by async execution
        # (the sync cost often gets charged to a later line). Use CUDA events when possible.
        use_cuda_events = sam_device.type == "cuda"

        def _evt():
            # Record an event on the correct device/stream.
            e = torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(sam_device):
                e.record()
            return e

        if use_cuda_events:
            e_total_start = _evt()
        else:
            t_total_start = time.time()

        if use_cuda_events:
            e_set_img_start = _evt()
        else:
            t_set_img_start = time.time()
        state = _set_image_batch_backbone_out_from_tensor01(
            img01_bchw, sam3_processor, orig_h=h, orig_w=w
        )
        backbone_out = state["backbone_out"]
        if use_cuda_events:
            e_set_img_end = _evt()
        else:
            t_set_img_elapsed = time.time() - t_set_img_start

        if use_cuda_events:
            e_text_start = _evt()
        else:
            t_text_start = time.time()
        text_outputs = _get_text_outputs_cached(model, find_text_batch, device=sam_device)
        backbone_out.update(text_outputs)
        if use_cuda_events:
            e_text_end = _evt()
        else:
            t_text_elapsed = time.time() - t_text_start
        
        # Use SAM3's built-in method to create dummy geometric prompt
        geometric_prompt = model._get_dummy_prompt(num_prompts=q)

        # forward_grounding timings
        t_forward_total = 0.0
        fwd_evt_pairs = []  # list[(evt_start, evt_end)]
        num_thresholds = len(thresholds)
        encode_times = []
        encoder_times = []
        transformer_encoder_times = []
        encoder_layer_times = []  # List of (layer_idx, time) tuples
        encoder_prepare_times = []  # Time for _prepare_multilevel_features
        prepare_detail_times = {
            'flatten_prep': [], 'loop': [], 'cat': [], 
            'spatial': [], 'spatial_device': [], 'spatial_tensor': [], 'spatial_index': [], 'spatial_valid': []
        }
        reference_points_times = []  # Time for get_reference_points
        decoder_times = []
        seg_times = []
        
        for thr_idx, thr in enumerate(thresholds):
            if use_cuda_events:
                e_fwd_start = _evt()
            else:
                t_fwd_start = time.time()
            find_input = FindStage(
                img_ids=img_ids_q,
                text_ids=text_ids_q,
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            
            # Detailed timing for forward_grounding sub-steps
            # Monkey-patch to measure internal steps
            original_encode_prompt = model._encode_prompt
            original_run_encoder = model._run_encoder
            original_run_decoder = model._run_decoder
            original_run_segmentation_heads = model._run_segmentation_heads
            
            def timed_encode_prompt(*args, **kwargs):
                if use_cuda_events:
                    e0 = _evt()
                    result = original_encode_prompt(*args, **kwargs)
                    e1 = _evt()
                    encode_times.append((e0, e1))
                else:
                    t0 = time.time()
                    result = original_encode_prompt(*args, **kwargs)
                    encode_times.append(time.time() - t0)
                return result
            
            def timed_run_encoder(*args, **kwargs):
                if use_cuda_events:
                    e0 = _evt()
                else:
                    t0 = time.time()
                # Also measure transformer.encoder.forward time and each layer
                original_transformer_encoder_forward = model.transformer.encoder.forward
                original_prepare_multilevel = model.transformer.encoder._prepare_multilevel_features
                local_transformer_times = []
                layer_times = []
                prepare_times = []
                
                # Patch _prepare_multilevel_features to measure time and internal steps
                def timed_prepare_multilevel(srcs, masks, pos_embeds):
                    t_prep0 = time.time()
                    # Measure individual operations
                    t_flatten_start = time.time()
                    src_flatten = []
                    mask_flatten = []
                    lvl_pos_embed_flatten = []
                    spatial_shapes = []
                    has_mask = masks is not None and masks[0] is not None
                    t_flatten_prep = time.time() - t_flatten_start
                    
                    t_loop_start = time.time()
                    for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
                        bs, c, h, w = src.shape
                        spatial_shape = (h, w)
                        spatial_shapes.append(spatial_shape)
                        
                        src = src.flatten(2).transpose(1, 2)  # bs, hw, c
                        if has_mask:
                            mask = mask.flatten(1)
                        pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
                        
                        if model.transformer.encoder.level_embed is not None:
                            lvl_pos_embed = pos_embed + model.transformer.encoder.level_embed[lvl].view(1, 1, -1)
                        else:
                            lvl_pos_embed = pos_embed
                        lvl_pos_embed_flatten.append(lvl_pos_embed)
                        src_flatten.append(src)
                        if has_mask:
                            mask_flatten.append(mask)
                    
                    t_loop_elapsed = time.time() - t_loop_start
                    t_cat_start = time.time()
                    src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
                    mask_flatten = torch.cat(mask_flatten, 1) if has_mask else None  # bs, \sum{hxw}
                    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
                    t_cat_elapsed = time.time() - t_cat_start
                    
                    t_spatial_start = time.time()
                    t_device_start = time.time()
                    # Check if we need to sync (original code uses src_flatten.device which may cause sync)
                    # We use pre-extracted device to avoid sync, but let's verify
                    actual_device = device  # Already extracted from srcs[0] at function start
                    t_device_elapsed = time.time() - t_device_start
                    
                    t_tensor_start = time.time()
                    # NOTE: spatial_shapes is used for indexing/shape math inside SAM3 encoder.
                    # It must be integer type and on the same device as the vision features.
                    spatial_shapes = torch.tensor(
                        spatial_shapes, dtype=torch.long, device=actual_device
                    )
                    t_tensor_elapsed = time.time() - t_tensor_start
                    
                    t_index_start = time.time()
                    level_start_index = torch.cat(
                        (
                            spatial_shapes.new_zeros((1,)),
                            spatial_shapes.prod(1).cumsum(0)[:-1],
                        )
                    )
                    t_index_elapsed = time.time() - t_index_start
                    
                    t_valid_start = time.time()
                    if has_mask:
                        from sam3.model.model_misc import get_valid_ratio  # type: ignore[import-not-found]
                        valid_ratios = torch.stack([get_valid_ratio(m) for m in masks], 1)
                    else:
                        valid_ratios = torch.ones(
                            (src_flatten.shape[0], model.transformer.encoder.num_feature_levels, 2),
                            device=src_flatten.device,
                        )
                    t_valid_elapsed = time.time() - t_valid_start
                    t_spatial_elapsed = time.time() - t_spatial_start
                    
                    prepare_times.append(time.time() - t_prep0)
                    
                    # Store detailed timing for logging (use outer scope variable)
                    prepare_detail_times['flatten_prep'].append(t_flatten_prep)
                    prepare_detail_times['loop'].append(t_loop_elapsed)
                    prepare_detail_times['cat'].append(t_cat_elapsed)
                    prepare_detail_times['spatial'].append(t_spatial_elapsed)
                    prepare_detail_times['spatial_device'].append(t_device_elapsed)
                    prepare_detail_times['spatial_tensor'].append(t_tensor_elapsed)
                    prepare_detail_times['spatial_index'].append(t_index_elapsed)
                    prepare_detail_times['spatial_valid'].append(t_valid_elapsed)
                    
                    return (
                        src_flatten,
                        mask_flatten,
                        lvl_pos_embed_flatten,
                        level_start_index,
                        valid_ratios,
                        spatial_shapes,
                    )
                
                # Patch each encoder layer to measure time
                original_layer_forwards = []
                for layer_idx, layer in enumerate(model.transformer.encoder.layers):
                    original_layer_forward = layer.forward
                    def make_timed_layer_forward(layer_idx, orig_forward):
                        def timed_layer_forward(*layer_args, **layer_kwargs):
                            t_layer0 = time.time()
                            result = orig_forward(*layer_args, **layer_kwargs)
                            layer_times.append((layer_idx, time.time() - t_layer0))
                            return result
                        return timed_layer_forward
                    layer.forward = make_timed_layer_forward(layer_idx, original_layer_forward)
                    original_layer_forwards.append((layer_idx, original_layer_forward))
                
                # Also measure get_reference_points time
                original_get_reference_points = model.transformer.encoder.get_reference_points
                reference_points_times_local = []
                
                def timed_get_reference_points(*args, **kwargs):
                    t_ref0 = time.time()
                    result = original_get_reference_points(*args, **kwargs)
                    reference_points_times_local.append(time.time() - t_ref0)
                    return result
                
                def timed_transformer_encoder_forward(*enc_args, **enc_kwargs):
                    t_enc0 = time.time()
                    result = original_transformer_encoder_forward(*enc_args, **enc_kwargs)
                    local_transformer_times.append(time.time() - t_enc0)
                    return result
                
                model.transformer.encoder.forward = timed_transformer_encoder_forward
                model.transformer.encoder._prepare_multilevel_features = timed_prepare_multilevel
                model.transformer.encoder.get_reference_points = timed_get_reference_points
                try:
                    result = original_run_encoder(*args, **kwargs)
                finally:
                    model.transformer.encoder.forward = original_transformer_encoder_forward
                    model.transformer.encoder._prepare_multilevel_features = original_prepare_multilevel
                    model.transformer.encoder.get_reference_points = original_get_reference_points
                    # Restore original layer forwards
                    for layer_idx, orig_forward in original_layer_forwards:
                        model.transformer.encoder.layers[layer_idx].forward = orig_forward
                
                if use_cuda_events:
                    e1 = _evt()
                    encoder_times.append((e0, e1))
                else:
                    encoder_times.append(time.time() - t0)
                if local_transformer_times:
                    transformer_encoder_times.extend(local_transformer_times)
                if layer_times:
                    encoder_layer_times.extend(layer_times)
                if prepare_times:
                    encoder_prepare_times.extend(prepare_times)
                if reference_points_times_local:
                    reference_points_times.extend(reference_points_times_local)
                return result
            
            def timed_run_decoder(*args, **kwargs):
                if use_cuda_events:
                    e0 = _evt()
                    result = original_run_decoder(*args, **kwargs)
                    e1 = _evt()
                    decoder_times.append((e0, e1))
                else:
                    t0 = time.time()
                    result = original_run_decoder(*args, **kwargs)
                    decoder_times.append(time.time() - t0)
                return result
            
            def timed_run_segmentation_heads(*args, **kwargs):
                if use_cuda_events:
                    e0 = _evt()
                    result = original_run_segmentation_heads(*args, **kwargs)
                    e1 = _evt()
                    seg_times.append((e0, e1))
                else:
                    t0 = time.time()
                    result = original_run_segmentation_heads(*args, **kwargs)
                    seg_times.append(time.time() - t0)
                return result
            
            model._encode_prompt = timed_encode_prompt
            model._run_encoder = timed_run_encoder
            model._run_decoder = timed_run_decoder
            model._run_segmentation_heads = timed_run_segmentation_heads
            
            try:
                outputs = model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    geometric_prompt=geometric_prompt,
                    find_target=None,
                )
            finally:
                # Restore original methods
                model._encode_prompt = original_encode_prompt
                model._run_encoder = original_run_encoder
                model._run_decoder = original_run_decoder
                model._run_segmentation_heads = original_run_segmentation_heads
            
            # Process logits and presence scores
            logits = outputs["pred_logits"]
            if logits.dim() == 3 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            elif logits.dim() > 2:
                logits = logits.reshape(logits.shape[0], logits.shape[1], -1).amax(dim=-1)
            
            probs = logits.sigmoid()
            presence = outputs.get("presence_logit_dec")
            if presence is not None:
                p = presence.sigmoid()
                if p.dim() > 1:
                    p = p.reshape(p.shape[0], -1).amax(dim=-1)
                probs = probs * p.unsqueeze(1)
            
            masks = outputs["pred_masks"]
            keep = probs > float(thr)
            num_masks = keep.sum(dim=1)
            
            # Interpolate masks and compute union
            qn = masks.shape[0] * masks.shape[1]
            masks01 = interpolate(
                masks.reshape(qn, 1, masks.shape[-2], masks.shape[-1]),
                (h, w),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()
            masks_bin = masks01.reshape(q, masks.shape[1], h, w) > 0.5
            thr_union = (masks_bin & keep[:, :, None, None]).any(dim=1)

            # Update best threshold/masks
            thr_f = torch.full((q,), float(thr), device=sam_device, dtype=torch.float32)
            better = (thr_f > best_thr) | ((thr_f == best_thr) & (num_masks > best_num))
            if better.any():
                best_thr = torch.where(better, thr_f, best_thr)
                best_num = torch.where(better, num_masks, best_num)
                best_union = torch.where(better[:, None, None], thr_union, best_union)
            
            # Cleanup
            del outputs, logits, presence, masks, masks01, masks_bin, thr_union, thr_f, better
            if use_cuda_events:
                e_fwd_end = _evt()
                fwd_evt_pairs.append((e_fwd_start, e_fwd_end))
            else:
                t_fwd_elapsed = time.time() - t_fwd_start
                t_forward_total += t_fwd_elapsed
        
        if use_cuda_events:
            e_total_end = _evt()
            # One sync at the end to materialize all event timings.
            torch.cuda.synchronize(sam_device)
            t_total_elapsed = e_total_start.elapsed_time(e_total_end) / 1000.0
            t_set_img_elapsed = e_set_img_start.elapsed_time(e_set_img_end) / 1000.0
            t_text_elapsed = e_text_start.elapsed_time(e_text_end) / 1000.0
            t_forward_total = sum(e0.elapsed_time(e1) for (e0, e1) in fwd_evt_pairs) / 1000.0

            # If available, report set_image internal breakdown (preprocess vs backbone forward).
            set_img_pre = set_img_fwd = None
            evts = state.get("__set_image_evt")
            if isinstance(evts, dict):
                pre = evts.get("pre")
                fwd = evts.get("fwd")
                if isinstance(pre, tuple) and len(pre) == 2:
                    set_img_pre = pre[0].elapsed_time(pre[1]) / 1000.0
                if isinstance(fwd, tuple) and len(fwd) == 2:
                    set_img_fwd = fwd[0].elapsed_time(fwd[1]) / 1000.0
        else:
            t_total_elapsed = time.time() - t_total_start
        
        # Log detailed timing breakdown
        import logging
        logger = logging.getLogger(__name__)
        
        # Calculate average times for forward_grounding sub-steps
        if use_cuda_events:
            avg_encode = (
                sum(e0.elapsed_time(e1) for (e0, e1) in encode_times) / len(encode_times) / 1000.0
                if encode_times
                else 0.0
            )
            avg_encoder = (
                sum(e0.elapsed_time(e1) for (e0, e1) in encoder_times) / len(encoder_times) / 1000.0
                if encoder_times
                else 0.0
            )
            avg_decoder = (
                sum(e0.elapsed_time(e1) for (e0, e1) in decoder_times) / len(decoder_times) / 1000.0
                if decoder_times
                else 0.0
            )
            avg_seg = (
                sum(e0.elapsed_time(e1) for (e0, e1) in seg_times) / len(seg_times) / 1000.0
                if seg_times
                else 0.0
            )
        else:
            avg_encode = sum(encode_times) / len(encode_times) if encode_times else 0.0
            avg_encoder = sum(encoder_times) / len(encoder_times) if encoder_times else 0.0
            avg_decoder = sum(decoder_times) / len(decoder_times) if decoder_times else 0.0
            avg_seg = sum(seg_times) / len(seg_times) if seg_times else 0.0
        
        # Get transformer encoder times
        avg_transformer_encoder = sum(transformer_encoder_times) / len(transformer_encoder_times) if transformer_encoder_times else 0.0
        
        # Calculate per-layer average times
        layer_avg_times = {}
        if encoder_layer_times:
            from collections import defaultdict
            layer_time_sums = defaultdict(float)
            layer_counts = defaultdict(int)
            for layer_idx, layer_time in encoder_layer_times:
                layer_time_sums[layer_idx] += layer_time
                layer_counts[layer_idx] += 1
            layer_avg_times = {idx: layer_time_sums[idx] / layer_counts[idx] 
                              for idx in layer_time_sums}
        
        logger.info(
            f"[SAM3 Latency] Total={t_total_elapsed:.3f}s | "
            f"set_image={t_set_img_elapsed:.3f}s | "
            f"forward_text={t_text_elapsed:.3f}s | "
            f"forward_grounding(x{num_thresholds})={t_forward_total:.3f}s "
            f"(avg={t_forward_total/num_thresholds:.3f}s/thr) | "
            f"other={t_total_elapsed - t_set_img_elapsed - t_text_elapsed - t_forward_total:.3f}s"
        )
        if use_cuda_events and (set_img_pre is not None or set_img_fwd is not None):
            logger.info(
                f"[SAM3 set_image breakdown] preprocess={float(set_img_pre or 0.0):.3f}s | "
                f"backbone_forward={float(set_img_fwd or 0.0):.3f}s"
            )
        # Note: avg_transformer_encoder is computed from time.time() monkeypatch timings and can be
        # misleading under CUDA async execution. When using CUDA-event timing, omit it.
        if use_cuda_events:
            logger.info(
                f"[SAM3 forward_grounding breakdown] "
                f"encode_prompt={avg_encode:.3f}s | "
                f"run_encoder={avg_encoder:.3f}s | "
                f"run_decoder={avg_decoder:.3f}s | "
                f"run_segmentation_heads={avg_seg:.3f}s"
            )
        else:
            logger.info(
                f"[SAM3 forward_grounding breakdown] "
                f"encode_prompt={avg_encode:.3f}s | "
                f"run_encoder={avg_encoder:.3f}s "
                f"(transformer.encoder={avg_transformer_encoder:.3f}s) | "
                f"run_decoder={avg_decoder:.3f}s | "
                f"run_segmentation_heads={avg_seg:.3f}s"
            )
        avg_prepare = sum(encoder_prepare_times) / len(encoder_prepare_times) if encoder_prepare_times else 0.0
        avg_reference_points = sum(reference_points_times) / len(reference_points_times) if reference_points_times else 0.0
        total_layer_time = sum(layer_avg_times.values()) if layer_avg_times else 0.0
        other_encoder_time = avg_transformer_encoder - total_layer_time - avg_prepare - avg_reference_points

        # Detailed encoder breakdown relies on time.time-based monkeypatch timings.
        if (not use_cuda_events) and (layer_avg_times or encoder_prepare_times or reference_points_times):
            parts = []
            if avg_prepare > 0:
                parts.append(f"prepare={avg_prepare:.3f}s")
            if avg_reference_points > 0:
                parts.append(f"get_ref_points={avg_reference_points:.3f}s")
            if layer_avg_times:
                layer_times_str = " | ".join([f"L{i}={t:.3f}s" for i, t in sorted(layer_avg_times.items())])
                parts.append(layer_times_str)
            if other_encoder_time > 0.001:
                parts.append(f"other={other_encoder_time:.3f}s")
            logger.info(f"[SAM3 encoder breakdown] {' | '.join(parts)}")
            
            # Log detailed prepare breakdown if prepare takes significant time
            if avg_prepare > 0.01:
                avg_prepare_flatten_prep = sum(prepare_detail_times.get('flatten_prep', [])) / len(prepare_detail_times.get('flatten_prep', [])) if prepare_detail_times.get('flatten_prep') else 0.0
                avg_prepare_loop = sum(prepare_detail_times.get('loop', [])) / len(prepare_detail_times.get('loop', [])) if prepare_detail_times.get('loop', []) else 0.0
                avg_prepare_cat = sum(prepare_detail_times.get('cat', [])) / len(prepare_detail_times.get('cat', [])) if prepare_detail_times.get('cat', []) else 0.0
                avg_prepare_spatial = sum(prepare_detail_times.get('spatial', [])) / len(prepare_detail_times.get('spatial', [])) if prepare_detail_times.get('spatial', []) else 0.0
                avg_spatial_device = sum(prepare_detail_times.get('spatial_device', [])) / len(prepare_detail_times.get('spatial_device', [])) if prepare_detail_times.get('spatial_device', []) else 0.0
                avg_spatial_tensor = sum(prepare_detail_times.get('spatial_tensor', [])) / len(prepare_detail_times.get('spatial_tensor', [])) if prepare_detail_times.get('spatial_tensor', []) else 0.0
                avg_spatial_index = sum(prepare_detail_times.get('spatial_index', [])) / len(prepare_detail_times.get('spatial_index', [])) if prepare_detail_times.get('spatial_index', []) else 0.0
                avg_spatial_valid = sum(prepare_detail_times.get('spatial_valid', [])) / len(prepare_detail_times.get('spatial_valid', [])) if prepare_detail_times.get('spatial_valid', []) else 0.0
                logger.info(
                    f"[SAM3 prepare breakdown] "
                    f"flatten_prep={avg_prepare_flatten_prep:.3f}s | "
                    f"loop={avg_prepare_loop:.3f}s | "
                    f"cat={avg_prepare_cat:.3f}s | "
                    f"spatial={avg_prepare_spatial:.3f}s"
                )
                logger.info(
                    f"[SAM3 spatial breakdown] "
                    f"device={avg_spatial_device:.3f}s | "
                    f"tensor={avg_spatial_tensor:.3f}s | "
                    f"index={avg_spatial_index:.3f}s | "
                    f"valid={avg_spatial_valid:.3f}s"
                )

    # Aggregate to per-image union mask.
    union_bt_hw = torch.zeros((b, h, w), dtype=torch.bool, device=sam_device)
    for i in range(b):
        sel = img_ids_q == i
        if sel.any():
            union_bt_hw[i] = best_union[sel].any(dim=0)

    weight_maps = torch.zeros((b, t, 1, h, w), dtype=torch.float32, device=device)
    wm = union_bt_hw.to(device=device, dtype=torch.float32)
    weight_maps[:] = wm[:, None, None, :, :].expand(b, t, 1, h, w)
    
    return weight_maps
