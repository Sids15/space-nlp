"""Offline checks for the n-gram models. Run: python tests/test_language_model.py"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from language_model import EOS, UNK, NGramLM

TRAIN = [s.split() for s in [
    "the storm degraded gps positioning", "the storm disrupted hf radio",
    "the flare disrupted hf radio", "gps positioning degraded during the storm",
    "a rare word appears once",
]]


def test_distributions_sum_to_one():
    for order in (2, 3):
        for smoothing in ("laplace", "kn"):
            lm = NGramLM(order, smoothing).fit(TRAIN)
            for ctx in [("the", "storm")[-(order - 1):], ("unseen", "ctx")[-(order - 1):]]:
                total = sum(lm.prob(ctx, w) for w in lm.vocab)
                assert math.isclose(total, 1.0, abs_tol=1e-9), (order, smoothing, ctx, total)


def test_rare_words_become_unk():
    lm = NGramLM(2, "kn").fit(TRAIN)
    assert "rare" not in lm.vocab and UNK in lm.vocab and EOS in lm.vocab


def test_mle_gives_inf_on_unseen_and_smoothing_fixes_it():
    test = ["the flare degraded gps positioning".split()]
    assert NGramLM(2, "mle").fit(TRAIN).perplexity(test)[0] == math.inf
    for smoothing in ("laplace", "kn"):
        assert math.isfinite(NGramLM(2, smoothing).fit(TRAIN).perplexity(test)[0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
