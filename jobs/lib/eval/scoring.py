"""Pure-Python helpers for benchmark scoring (no third-party imports, so they can be unit-tested anywhere).

Math answers are graded with slime's `grade_answer_verl` in run_bench.py (the same grader as the GRPO `math` reward);
this module covers multiple-choice extraction and the aggregate statistics.
"""

from __future__ import annotations

import math
import re

LETTERS = "ABCDEFGHIJ"


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}) in `text`, with nested braces handled; None if absent."""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    # "\boxed 5" (no braces) is rare; only the braced form is accepted.
    if text[start:i].strip() not in ("\\boxed", "\\fbox"):
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


_TEXT_WRAP = re.compile(r"\\(?:text|textbf|mathrm|mathbf)\{([^{}]*)\}")
_ANSWER_IS = re.compile(r"answer is\s*:?\s*\(?\s*\\?(?:text\{)?([A-J])\b")


def mc_letter(text: str, n_choices: int) -> str | None:
    """The option letter a response chose: last \\boxed{} first, else the last 'answer is (X)'. None if neither."""
    valid = LETTERS[:n_choices]
    box = last_boxed(text)
    if box is not None:
        s = box.strip()
        for _ in range(3):  # \boxed{\text{(B)}} etc.
            s = _TEXT_WRAP.sub(r"\1", s).strip()
        s = s.strip("()[]{}. :$").strip()
        if len(s) >= 1 and s[0] in valid and (len(s) == 1 or not s[1].isalpha()):
            return s[0]
    hits = [m.group(1) for m in _ANSWER_IS.finditer(text) if m.group(1) in valid]
    return hits[-1] if hits else None


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021) from n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def mean_ci95(values: list[float]) -> tuple[float, float]:
    """Mean and 95% normal-approximation half-width over independent units (here: questions)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, 1.96 * math.sqrt(var / n)
