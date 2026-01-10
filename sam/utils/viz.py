from __future__ import annotations

from typing import Any

import numpy as np


def visualize_sam3_results(image_pil, masks: Any, boxes: Any, scores: Any, title: str = "SAM3 Segmentation"):
    """
    Visualize SAM3 results similarly to the notebook:
    - left: original
    - right: overlays masks + boxes
    """
    import matplotlib.pyplot as plt
    import torch

    img_np = np.array(image_pil)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(img_np)

    if masks is not None and len(masks) > 0:
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        boxes_np = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
        scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores

        overlay = img_np.copy()
        colors = plt.cm.tab20(np.linspace(0, 1, len(masks_np)))

        for i, (mask, score) in enumerate(zip(masks_np, scores_np)):
            color = (colors[i][:3] * 255).astype(np.uint8)
            mask = np.squeeze(mask)
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (overlay[mask_bool] * 0.5 + color * 0.5).astype(np.uint8)

        axes[1].imshow(overlay)

        if boxes_np is not None and len(boxes_np) > 0:
            for i, (box, score) in enumerate(zip(boxes_np, scores_np)):
                if len(box) != 4:
                    continue
                x1, y1, x2, y2 = box
                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=colors[i],
                    linewidth=2,
                )
                axes[1].add_patch(rect)
                axes[1].text(
                    x1,
                    y1 - 5,
                    f"{float(score):.2f}",
                    color=colors[i],
                    fontsize=10,
                    weight="bold",
                )

    axes[1].set_title(f"{title} ({len(masks) if masks is not None else 0} masks)")
    axes[1].axis("off")

    plt.tight_layout()
    return fig


