from __future__ import annotations

import re
from typing import List


def build_prompt_candidates(instruction: str) -> List[str]:
    """
    Heuristic prompt extraction from a full language instruction.

    Example:
        "put both the alphabet soup and the tomato sauce in the basket"
      -> ["alphabet soup", "tomato sauce", "basket"]
    """
    if not isinstance(instruction, str):
        instruction = str(instruction)
    base = instruction.strip()
    if not base:
        return ["object"]

    low = base.lower()
    # Extract phrases like "the X" up to a conjunction/prep/punct.
    phrases = re.findall(
        r"\bthe ([a-z0-9\- ]+?)(?=\s+(?:and|in|on|into|onto|to)\b|\s*[\.,;:]|$)",
        low,
    )

    cands: List[str] = []
    for p in phrases:
        p = p.strip()
        if p and p not in cands:
            cands.append(p)

    return cands or [base]


