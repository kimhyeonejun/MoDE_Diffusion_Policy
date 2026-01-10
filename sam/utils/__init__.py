"""Utilities extracted from `sam/grounded_sam.ipynb`.

This package is intended to keep notebook logic reusable from scripts/tests.
It is *not* part of Meta SAM3 code; it's project glue around:
- SAM3 segmentation via text prompts
- LIBERO batch image extraction and visualization helpers
"""

from .image_utils import extract_image_from_batch, numpy_to_pil
from .prompts import build_prompt_candidates
from .sam3_segmentation import segment_with_sam3_text_prompts
from .sam3_weight_map import get_sam3_processor, tensor01_to_pil, compute_weight_map
from .viz import visualize_sam3_results

__all__ = [
    "extract_image_from_batch",
    "numpy_to_pil",
    "build_prompt_candidates",
    "segment_with_sam3_text_prompts",
    "get_sam3_processor",
    "tensor01_to_pil",
    "compute_weight_map",
    "visualize_sam3_results",
]


