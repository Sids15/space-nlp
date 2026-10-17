"""Stage 3: domain language model. Bigram/trigram models with no smoothing (MLE), Laplace, and
interpolated Kneser-Ney smoothing. Perplexity on space-weather vs general astrophysics text
shows how different the domain's language is.

Usage:  python src/language_model.py [--version v1]

Each model is trained on one domain and evaluated on held-out text from both domains. The two
training sets are cut to the same token count so neither model wins on data volume alone.
Writes data/lm/perplexity_<version>.json.
"""

import argparse
import json
import logging
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "lm"
SEED = 42
TEST_FRACTION = 0.1
MIN_COUNT = 2  # words seen once in training become <unk>
BOS, EOS, UNK, NUM = "<s>", "</s>", "<unk>", "<num>"
NUM_RE = re.compile(r"^[\d.,+\-−–/]*\d[\d.,+\-−–/%]*$")
log = logging.getLogger("language_model")


class NGramLM:
    """smoothing: 'mle' (none), 'laplace' (add-one) or 'kn' (interpolated Kneser-Ney)."""

    def __init__(self, order, smoothing, discount=0.75):
        self.n, self.smoothing, self.d = order, smoothing, discount

    def fit(self, sentences):
        counts = Counter(w for s in sentences for w in s)
        self.vocab = {w for w, c in counts.items() if c >= MIN_COUNT} | {EOS, UNK}
        grams = [Counter() for _ in range(self.n + 1)]  # grams[k]: k-gram counts
        for sent in sentences:
            padded = self._pad(sent)
            for k in range(1, self.n + 1):
                for i in range(max(0, self.n - 1 - k + 1), len(padded) - k + 1):  # never end on <s>
                    grams[k][tuple(padded[i:i + k])] += 1

        # Level n uses raw counts. For KN, lower levels use continuation counts:
        # how many distinct words precede the k-gram (N1+(. w)).
        self.count = [None] * (self.n + 1)
        self.count[self.n] = grams[self.n]
        for k in range(self.n - 1, 0, -1):
            self.count[k] = Counter(g[1:] for g in grams[k + 1]) if self.smoothing == "kn" else grams[k]
        self.ctx_total = [defaultdict(int) for _ in range(self.n + 1)]
        self.ctx_types = [defaultdict(int) for _ in range(self.n + 1)]
        for k in range(1, self.n + 1):
            for g, c in self.count[k].items():
                self.ctx_total[k][g[:-1]] += c
                self.ctx_types[k][g[:-1]] += 1
        return self

    def _pad(self, sent):
        return [BOS] * (self.n - 1) + [w if w in self.vocab else UNK for w in sent] + [EOS]

    def prob(self, ctx, w):
        if self.smoothing == "mle":
            total = self.ctx_total[self.n][ctx]
            return self.count[self.n][ctx + (w,)] / total if total else 0.0
        if self.smoothing == "laplace":
            return (self.count[self.n][ctx + (w,)] + 1) / (self.ctx_total[self.n][ctx] + len(self.vocab))
        return self._kn(self.n, ctx, w)

    def _kn(self, k, ctx, w):
        if k == 0:
            return 1 / len(self.vocab)
        lower = self._kn(k - 1, ctx[1:], w)
        total = self.ctx_total[k][ctx]
        if not total:  # unseen context: all mass goes to the lower order
            return lower
        return (max(self.count[k][ctx + (w,)] - self.d, 0) + self.d * self.ctx_types[k][ctx] * lower) / total

    def perplexity(self, sentences):
        """Returns (perplexity, share of tokens given zero probability)."""
        log_sum, n, zeros = 0.0, 0, 0
        for sent in sentences:
            padded = self._pad(sent)
            for i in range(self.n - 1, len(padded)):
                p = self.prob(tuple(padded[i - self.n + 1:i]), padded[i])
                n += 1
                if p == 0:
                    zeros += 1
                else:
                    log_sum += math.log(p)
        return (math.inf if zeros else math.exp(-log_sum / n)), zeros / n


def load_docs(path):
    if not path.exists():
        sys.exit(f"{path} missing. Run the earlier stages first.")
    docs = []
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        docs.append([[NUM if NUM_RE.match(t) else t for t in s["tokens"]] for s in rec["sentences"] if s["tokens"]])
    return docs


def split(docs, rng):
    """Split by abstract, not by sentence, so no abstract leaks into both sides."""
    docs = docs[:]
    rng.shuffle(docs)
    cut = int(len(docs) * TEST_FRACTION)
    return [s for d in docs[cut:] for s in d], [s for d in docs[:cut] for s in d]


def truncate(sentences, max_tokens):
    out, total = [], 0
    for s in sentences:
        if total + len(s) > max_tokens:
            break
        out.append(s)
        total += len(s)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rng = random.Random(SEED)
    data = {
        "space_weather": split(load_docs(ROOT / "data" / "relevance" / f"filtered_{args.version}.jsonl"), rng),
        "astrophysics": split(load_docs(ROOT / "data" / "processed" / f"reference_{args.version}.jsonl"), rng),
    }
    budget = min(sum(map(len, train)) for train, _ in data.values())
    log.info("Training budget per domain: %d tokens", budget)

    models = [("bigram", 2, "mle"), ("bigram", 2, "laplace"), ("trigram", 3, "laplace"),
              ("bigram", 2, "kn"), ("trigram", 3, "kn")]
    results = []
    for train_domain, (train, _) in data.items():
        train = truncate(train, budget)
        for name, order, smoothing in models:
            lm = NGramLM(order, smoothing).fit(train)
            for test_domain, (_, test) in data.items():
                ppl, zero_rate = lm.perplexity(test)
                oov = sum(w not in lm.vocab for s in test for w in s) / sum(map(len, test))
                results.append({"train": train_domain, "test": test_domain, "model": f"{name}-{smoothing}",
                                "perplexity": None if math.isinf(ppl) else round(ppl, 1),
                                "zero_prob_rate": round(zero_rate, 4), "oov_rate": round(oov, 4),
                                "vocab": len(lm.vocab)})
                log.info("train=%-13s test=%-13s %-16s ppl=%-8s oov=%.3f zero=%.3f", train_domain,
                         test_domain, f"{name}-{smoothing}", results[-1]["perplexity"] or "inf", oov, zero_rate)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"perplexity_{args.version}.json"
    out.write_text(json.dumps({"seed": SEED, "test_fraction": TEST_FRACTION, "min_count": MIN_COUNT,
                               "train_tokens_per_domain": budget, "results": results}, indent=2), encoding="utf-8")
    log.info("Wrote %s", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
