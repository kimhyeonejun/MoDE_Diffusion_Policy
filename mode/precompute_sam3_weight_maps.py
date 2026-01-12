"""
Precompute SAM3 union masks offline for LIBERO training samples.

This iterates the same LiberoDataModule dataset and saves per-sample binary masks
so training can load them without running SAM3 (major speedup).

Usage (example):
  python -m mode.precompute_sam3_weight_maps \
    precompute.out_dir=/path/to/sam3_masks \
    custom_loss.sam3_view=both \
    custom_loss.sam3_confidence_threshold=0.05 \
    custom_loss.sam3_thresholds=[0.05] \
    custom_loss.sam3_resolution=1008 \
    custom_loss.sam3_infer_dtype=bf16 \
    batch_size=8 num_workers=0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from sam.utils.sam3_weight_map import get_sam3_processor, compute_weight_map_from_lang_text_batch
from sam.utils.sam3_precomputed import make_sample_key, save_mask_u8

log = logging.getLogger(__name__)


def _cfg_get(cfg: DictConfig, key: str, default: Any):
    try:
        return cfg.get(key, default)
    except Exception:
        return default


@hydra.main(config_path="../conf", config_name="config_libero_msillm_sam.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    # Config section for this script
    pre = cfg.get("precompute", OmegaConf.create({}))
    out_dir = Path(str(_cfg_get(pre, "out_dir", "./sam3_precomputed")))
    limit = _cfg_get(pre, "limit", -1)
    start = int(_cfg_get(pre, "start", 0))
    batch_size = int(_cfg_get(pre, "batch_size", 8))
    save_every = int(_cfg_get(pre, "save_every", 100))

    # SAM3 params are read from the same custom_loss section used by training
    view = str(_cfg_get(cfg.custom_loss, "sam3_view", "both"))
    conf_thr = float(_cfg_get(cfg.custom_loss, "sam3_confidence_threshold", 0.05))
    thresholds = _cfg_get(cfg.custom_loss, "sam3_thresholds", [conf_thr])
    if isinstance(thresholds, (float, int)):
        thresholds = [float(thresholds)]
    thresholds = [float(x) for x in thresholds]
    sam3_resolution = int(_cfg_get(cfg.custom_loss, "sam3_resolution", 1008))
    sam3_infer_dtype = str(_cfg_get(cfg.custom_loss, "sam3_infer_dtype", "fp32"))
    max_prompts_per_image = int(_cfg_get(cfg.custom_loss, "sam3_max_prompts_per_image", 3))
    alpha = float(_cfg_get(cfg.custom_loss, "sam3_alpha", 1.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        f"[precompute] out_dir={out_dir} view={view} device={device} "
        f"conf_thr={conf_thr} thresholds={thresholds} res={sam3_resolution} dtype={sam3_infer_dtype} "
        f"max_prompts_per_image={max_prompts_per_image}"
    )

    # Build datamodule from config defaults (same as training)
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")
    dataset = dm.train_datasets["lang"]  # ConcatDataset of tasks
    n = len(dataset)
    if limit is not None and int(limit) > 0:
        n = min(n, start + int(limit))

    # Init SAM3 once
    sam3 = get_sam3_processor(
        device=device,
        logger=log,
        resolution=sam3_resolution,
        infer_dtype=sam3_infer_dtype,
        confidence_threshold=conf_thr,
    )

    # Iterate dataset in simple batches (index-based; avoids DataLoader collation variability)
    num_done = 0
    for base in range(start, n, batch_size):
        idxs = list(range(base, min(base + batch_size, n)))
        samples = [dataset[i] for i in idxs]

        # lang_text + local idx (may collide across tasks) -> stable hash key
        lang_texts = [str(s.get("lang_text", "")) for s in samples]
        local_idxs = [int(s.get("idx", i)) for s, i in zip(samples, idxs)]
        keys = [make_sample_key(t, j) for t, j in zip(lang_texts, local_idxs)]

        # Static/gripper GT: (B, T, C, H, W) in [0,1]
        if view in ("static", "both"):
            rgb_static = [s["rgb_obs"]["rgb_static"] for s in samples]  # (T,C,H,W)
            gt_static = torch.stack(rgb_static, dim=0).to(device=device, dtype=torch.float32)
            # only first frame is used; keep T=1 to save compute/memory
            gt_static = gt_static[:, :1]
            wm_static = compute_weight_map_from_lang_text_batch(
                gt_static,
                sam3,
                conf_thr,
                lang_texts,
                thresholds,
                alpha,
                max_prompts_per_image=max_prompts_per_image,
            )  # (B,1,1,H,W) float
            masks = (wm_static[:, 0, 0] > 0.5).to(torch.uint8).cpu()
            for k, m in zip(keys, masks):
                save_mask_u8(m, out_dir / "static" / f"{k}.pt")

        if view in ("gripper", "both"):
            rgb_gripper = [s["rgb_obs"]["rgb_gripper"] for s in samples]
            gt_grip = torch.stack(rgb_gripper, dim=0).to(device=device, dtype=torch.float32)
            gt_grip = gt_grip[:, :1]
            wm_grip = compute_weight_map_from_lang_text_batch(
                gt_grip,
                sam3,
                conf_thr,
                lang_texts,
                thresholds,
                alpha,
                max_prompts_per_image=max_prompts_per_image,
            )
            masks = (wm_grip[:, 0, 0] > 0.5).to(torch.uint8).cpu()
            for k, m in zip(keys, masks):
                save_mask_u8(m, out_dir / "gripper" / f"{k}.pt")

        num_done += len(idxs)
        if num_done % save_every == 0:
            log.info(f"[precompute] processed {num_done}/{n-start} samples")

    log.info(f"[precompute] done. wrote masks to {out_dir}")


if __name__ == "__main__":
    main()


