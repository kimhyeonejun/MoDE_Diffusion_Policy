### `sam/utils/` (from `grounded_sam.ipynb`)

This folder contains small reusable modules extracted from `sam/grounded_sam.ipynb`:

- `image_utils.py`: LIBERO batch → numpy image, numpy → PIL
- `prompts.py`: heuristics to convert full instructions to object prompts
- `sam3_segmentation.py`: run SAM3 text-prompt segmentation (multi-prompt + threshold sweep)
- `msillm.py`: MS-ILLM compress/decompress helpers (handles negative strides + cpu/gpu mismatch)
- `viz.py`: visualization helpers

Example usage (in a notebook where repo root is on `sys.path`):

```python
from sam.grounded import (
  extract_image_from_batch, numpy_to_pil,
  build_prompt_candidates, segment_with_sam3_text_prompts,
)
```


