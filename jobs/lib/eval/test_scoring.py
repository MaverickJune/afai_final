"""Unit tests for scoring.py (stdlib only): python jobs/lib/eval/test_scoring.py"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import last_boxed, mc_letter, mean_ci95, pass_at_k  # noqa: E402

assert last_boxed(r"so \boxed{\frac{1}{2}} and finally \boxed{x^{2}+1}.") == "x^{2}+1"
assert last_boxed(r"\fbox{7}") == "7"
assert last_boxed("no answer") is None
assert last_boxed(r"\boxed{unclosed") is None
assert last_boxed(r"\boxed {12}") == "12"

assert mc_letter(r"... thus \boxed{B}", 4) == "B"
assert mc_letter(r"\boxed{(C)}", 4) == "C"
assert mc_letter(r"\boxed{\text{D}}", 4) == "D"
assert mc_letter(r"\boxed{\textbf{(A)}}", 4) == "A"
assert mc_letter(r"\boxed{C) 42}", 4) == "C"
assert mc_letter(r"\boxed{E}", 4) is None              # out of range for 4 options
assert mc_letter(r"\boxed{E}", 10) == "E"
assert mc_letter(r"\boxed{Apple}", 4) is None          # a word, not a letter
assert mc_letter("The answer is (B).", 4) == "B"
assert mc_letter("first the answer is A, then the answer is: C", 4) == "C"
assert mc_letter("I think B", 4) is None

assert pass_at_k(4, 0, 1) == 0.0 and pass_at_k(4, 4, 1) == 1.0
assert abs(pass_at_k(4, 1, 1) - 0.25) < 1e-12
assert abs(pass_at_k(4, 1, 4) - 1.0) < 1e-12
assert abs(pass_at_k(16, 2, 4) - (1 - (14 * 13 * 12 * 11) / (16 * 15 * 14 * 13))) < 1e-12

m, h = mean_ci95([1, 0, 1, 0])
assert m == 0.5 and abs(h - 1.96 * math.sqrt((1 / 3) / 4)) < 1e-12
print("test_scoring: all passed")
