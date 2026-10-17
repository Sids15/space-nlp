"""Stage 9: evidence matrix. Combines the triple store into an event x system matrix.

Usage:  python src/matrix.py [--version v1] [--run full]

Per (event, system) cell:
  papers      distinct abstracts with at least one triple for the pair
  by_effect   papers reporting each effect label (a paper counts once per label)
  agreement   share of the cell's paper-votes that go to its majority label (1.0 = everyone agrees)
  status      gap (no papers) | single (1 paper) | agree (agreement >= 0.8) | mixed
  evidence    supporting sentences, highest-confidence first, each traceable by bibcode + sentence_id

Writes data/matrix/matrix_<run>_<version>.json.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from domain import EFFECTS, EVENTS, SYSTEMS

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "matrix"
AGREE = 0.8


def load_triples(run, version):
    path = ROOT / "data" / "triples" / f"{run}_{version}.jsonl"
    if not path.exists():
        sys.exit(f"{path} missing. Run effects.py first.")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def build(triples):
    votes = defaultdict(lambda: defaultdict(set))  # (event, system) -> effect -> {bibcode}
    evidence = defaultdict(list)
    for t in triples:
        key = (t["event"]["canonical"], t["system"]["canonical"])
        votes[key][t["effect"]].add(t["bibcode"])
        evidence[key].append({k: t[k] for k in ("bibcode", "sentence_id", "sentence", "effect", "method", "confidence")}
                             | {"event_text": t["event"]["text"], "system_text": t["system"]["text"]})
    cells = []
    for event in EVENTS:
        for system in SYSTEMS:
            v = votes.get((event, system), {})
            by_effect = {e: len(v.get(e, ())) for e in EFFECTS}
            papers = len(set().union(*v.values())) if v else 0
            total = sum(by_effect.values())
            agreement = round(max(by_effect.values()) / total, 3) if total else None
            status = ("gap" if papers == 0 else "single" if papers == 1 else
                      "agree" if agreement >= AGREE else "mixed")
            ev = sorted(evidence.get((event, system), []), key=lambda e: (-e["confidence"], e["bibcode"], e["sentence_id"]))
            # one sentence can hold several mention pairs for the same cell; show it once
            seen, unique = set(), []
            for e in ev:
                if (e["bibcode"], e["sentence_id"], e["effect"]) not in seen:
                    seen.add((e["bibcode"], e["sentence_id"], e["effect"]))
                    unique.append(e)
            cells.append({"event": event, "system": system, "papers": papers, "by_effect": by_effect,
                          "majority": max(by_effect, key=by_effect.get) if total else None,
                          "agreement": agreement, "status": status, "evidence": unique})
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--run", default="full", help="triple run name, e.g. full, full_nowsd, cooccurrence_full")
    args = parser.parse_args()
    cells = build(load_triples(args.run, args.version))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"matrix_{args.run}_{args.version}.json"
    out.write_text(json.dumps(cells, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{'event':18s} {'system':11s} papers  " + "  ".join(f"{e[:6]:>6s}" for e in EFFECTS) + "  agree  status")
    for c in cells:
        print(f"{c['event']:18s} {c['system']:11s} {c['papers']:6d}  " +
              "  ".join(f"{c['by_effect'][e]:6d}" for e in EFFECTS) +
              f"  {c['agreement'] if c['agreement'] is not None else '-':>5}  {c['status']}")
    print("->", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
