"""Utilities extracted from `sam/grounded_sam.ipynb`.

This package is intended to keep notebook logic reusable from scripts/tests.
It is *not* part of Meta SAM3 code; it's project glue around:
- MS-ILLM (NeuralCompression) compression/decompression
- SAM3 segmentation via text prompts
- LIBERO batch image extraction and visualization helpers
"""

from .image_utils import extract_image_from_batch, numpy_to_pil
from .msillm import compress_image_with_msillm, compress_numpy_image_with_msillm
from .prompts import build_prompt_candidates
from .sam3_segmentation import segment_with_sam3_text_prompts
from .viz import visualize_sam3_results

__all__ = [
    "extract_image_from_batch",
    "numpy_to_pil",
    "compress_image_with_msillm",
    "compress_numpy_image_with_msillm",
    "build_prompt_candidates",
    "segment_with_sam3_text_prompts",
    "visualize_sam3_results",
]


