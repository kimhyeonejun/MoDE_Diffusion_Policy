from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


@dataclass
class Sam3SegmentationResult:
    masks: Any
    boxes: Any
    scores: Any
    prompt_results: List[dict]


def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float(x.item())


def segment_with_sam3_text_prompts(
    sam3_processor,
    image_pil,
    prompts: Sequence[str],
    thresholds: Sequence[float],
    label: str = "image",
) -> Dict[str, Any]:
    """
    Run SAM3 text-prompt segmentation for multiple prompts and thresholds.

    Behavior matches the notebook logic:
    - For each prompt, try multiple thresholds; keep the best result per prompt
      (prefers higher threshold then more masks).
    - Combine masks/boxes/scores across prompts.
    """
    state = sam3_processor.set_image(image_pil)
    print(f"✓ {label}: image set")

    all_results: List[dict] = []

    for pi, prompt in enumerate(prompts):
        print(f"\n  [{pi+1}/{len(prompts)}] Trying prompt: {prompt!r}")
        prompt_results: List[dict] = []

        for thr in thresholds:
            sam3_processor.confidence_threshold = _as_float(thr)
            out = sam3_processor.set_text_prompt(prompt=prompt, state=state)
            n = len(out["scores"])

            if n > 0:
                scores = out["scores"]
                smin = scores.min().item() if isinstance(scores, torch.Tensor) else float(np.min(scores))
                smax = scores.max().item() if isinstance(scores, torch.Tensor) else float(np.max(scores))
                print(
                    f"    ✓ threshold={sam3_processor.confidence_threshold:.2f}: {n} masks "
                    f"(score range: [{smin:.3f}, {smax:.3f}])"
                )
                prompt_results.append(
                    {
                        "threshold": sam3_processor.confidence_threshold,
                        "masks": out["masks"],
                        "boxes": out["boxes"],
                        "scores": out["scores"],
                        "num_masks": n,
                    }
                )
            else:
                print(f"    ✗ threshold={sam3_processor.confidence_threshold:.2f}: 0 masks")

        if prompt_results:
            best_for_prompt = max(prompt_results, key=lambda x: (x["threshold"], x["num_masks"]))
            best_for_prompt["prompt"] = prompt
            all_results.append(best_for_prompt)
            print(
                f"    → Selected: threshold={best_for_prompt['threshold']:.2f}, "
                f"{best_for_prompt['num_masks']} masks\n"
            )
        else:
            print("    → No detections for this prompt\n")

    if not all_results:
        print("  Summary: No detections for any prompt")
        device = torch.device(getattr(sam3_processor, "device", "cpu"))
        return {
            "masks": torch.empty((0, 1, 1, 1), device=device, dtype=torch.bool),
            "boxes": torch.empty((0, 4), device=device, dtype=torch.float32),
            "scores": torch.empty((0,), device=device, dtype=torch.float32),
            "prompt_results": [],
        }

    print(f"  Summary: Found results for {len(all_results)}/{len(prompts)} prompts")

    combined_masks = []
    combined_boxes = []
    combined_scores = []

    for result in all_results:
        masks = result["masks"]
        boxes = result["boxes"]
        scores = result["scores"]

        for i in range(len(scores)):
            combined_masks.append(masks[i].clone() if isinstance(masks, torch.Tensor) else masks[i].copy())
            combined_boxes.append(boxes[i].clone() if isinstance(boxes, torch.Tensor) else boxes[i].copy())
            combined_scores.append(scores[i].clone() if isinstance(scores, torch.Tensor) else scores[i].copy())

    if isinstance(all_results[0]["masks"], torch.Tensor):
        return {
            "masks": torch.stack(combined_masks),
            "boxes": torch.stack(combined_boxes),
            "scores": torch.stack(combined_scores),
            "prompt_results": all_results,
        }

    # numpy fallback
    try:
        masks_arr = np.stack(combined_masks)
    except Exception:
        masks_arr = combined_masks
    return {
        "masks": masks_arr,
        "boxes": np.stack(combined_boxes),
        "scores": np.stack(combined_scores),
        "prompt_results": all_results,
    }


