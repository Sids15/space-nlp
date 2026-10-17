"""Stage 10: evaluation against the human gold set.

Usage:  python src/evaluate.py [--gold data/gold/exports/test_gold.json] [--version v1]

Scores every run that exists on disk against the gold sentences:
  entities  data/entities/<run>_<v>.jsonl   (full, gazetteer, spacy_ner, full_nowsd, full_nosyn)
            exact span match (start, end, label) and partial match (overlap + same label), per slot
  triples   data/triples/<run>_<v>.jsonl    (full, cooccurrence_full, full_noneg, full_nowsd, full_nosyn)
            labelled (event, system, effect) and unlabelled (event, system) per sentence,
            with a 95% bootstrap interval over sentences and a paired bootstrap against the baseline

Writes data/eval/results_<gold>_<v>.json and data/eval/errors_<gold>_<v>.csv (false positives / negatives of the
full run, with an empty `category` column for the error analysis).
"""

import argparse
import json
import random
import sys
from pathlib import Path

from domain import EVENTS, SYSTEMS
from gold import load_export
from relevance import write_csv

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "eval"
ENTITY_RUNS = ["full", "gazetteer", "spacy_ner", "full_nowsd", "full_nosyn"]
TRIPLE_RUNS = ["full", "cooccurrence_full", "full_noneg", "full_nowsd", "full_nosyn"]
BASELINE = "cooccurrence_full"
BOOTSTRAP = 1000
SEED = 42


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": round(p, 3), "r": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def partial_matches(pred, gold):
    """One-to-one greedy matching of overlapping spans with the same label."""
    used, tp = set(), 0
    for ps, pe, pl in sorted(pred):
        for i, (gs, ge, gl) in enumerate(sorted(gold)):
            if i not in used and gl == pl and ps < ge and gs < pe:
                used.add(i)
                tp += 1
                break
    return tp


def score_entities(gold, pred):
    out = {}
    for slot, allowed in (("event", set(EVENTS)), ("system", set(SYSTEMS)), ("all", set(EVENTS) | set(SYSTEMS))):
        exact = [0, 0, 0]
        part = [0, 0, 0]
        for sid, g in gold.items():
            gs = {e for e in g["entities"] if e[2] in allowed}
            ps = {e for e in pred.get(sid, ()) if e[2] in allowed}
            hit = len(gs & ps)
            exact = [exact[0] + hit, exact[1] + len(ps) - hit, exact[2] + len(gs) - hit]
            hit = partial_matches(ps, gs)
            part = [part[0] + hit, part[1] + len(ps) - hit, part[2] + len(gs) - hit]
        out[slot] = {"exact": prf(*exact), "partial": prf(*part)}
    return out


def gold_triples(g, labelled):
    out = set()
    for (event, system), effects in g["triples"].items():
        for effect in effects.split("|"):  # an annotator may give one pair two labels
            out.add((event, system, effect) if labelled else (event, system))
    return out


def counts(gold, pred, labelled, ids):
    tp = fp = fn = 0
    for sid in ids:
        gs = gold_triples(gold[sid], labelled)
        ps = {t if labelled else t[:2] for t in pred.get(sid, ())}
        tp, fp, fn = tp + len(gs & ps), fp + len(ps - gs), fn + len(gs - ps)
    return tp, fp, fn


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def score_triples(gold, runs, labelled):
    ids = sorted(gold)
    rng = random.Random(SEED)
    samples = [[rng.choice(ids) for _ in ids] for _ in range(BOOTSTRAP)]
    out = {}
    for run, pred in runs.items():
        res = prf(*counts(gold, pred, labelled, ids))
        boot = sorted(f1(*counts(gold, pred, labelled, s)) for s in samples)
        res["f1_ci95"] = [round(boot[int(0.025 * BOOTSTRAP)], 3), round(boot[int(0.975 * BOOTSTRAP) - 1], 3)]
        if BASELINE in runs and run != BASELINE:
            wins = sum(f1(*counts(gold, pred, labelled, s)) > f1(*counts(gold, runs[BASELINE], labelled, s))
                       for s in samples)
            res["p_beats_baseline"] = round(wins / BOOTSTRAP, 3)  # share of resamples where run F1 > baseline F1
        out[run] = res
    return out


def load_entities(run, version, ids):
    path = ROOT / "data" / "entities" / f"{run}_{version}.jsonl"
    if not path.exists():
        return None
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        key = f"{r['bibcode']}|{r['sid']}"
        if key in ids:
            out[key] = {(e["start"], e["end"], e["canonical"]) for e in r["entities"]}
    return out


def load_triples(run, version, ids):
    path = ROOT / "data" / "triples" / f"{run}_{version}.jsonl"
    if not path.exists():
        return None
    out = {}
    for line in path.open(encoding="utf-8"):
        t = json.loads(line)
        key = f"{t['bibcode']}|{t['sentence_id']}"
        if key in ids:
            out.setdefault(key, set()).add((t["event"]["canonical"], t["system"]["canonical"], t["effect"]))
    return out


def errors(gold, ents, triples):
    rows = []
    for sid, g in sorted(gold.items()):
        text = g["text"]
        gs, ps = set(g["entities"]), ents.get(sid, set())
        for kind, spans in (("entity_fp", ps - gs), ("entity_fn", gs - ps)):
            for s, e, label in sorted(spans):
                rows.append({"kind": kind, "id": sid, "item": f"'{text[s:e]}' {label}", "sentence": text})
        gt, pt = gold_triples(g, True), triples.get(sid, set())
        for kind, items in (("triple_fp", pt - gt), ("triple_fn", gt - pt)):
            for t in sorted(items):
                rows.append({"kind": kind, "id": sid, "item": " -> ".join(t), "sentence": text})
    for r in rows:
        r["category"] = ""  # filled in by hand: boundary / wrong label / missed term / cue / scope / gold error ...
    return rows


def table(title, rows):
    print(f"\n{title}")
    for name, r in rows:
        extra = "".join(f"  {k}={r[k]}" for k in ("f1_ci95", "p_beats_baseline") if k in r)
        print(f"  {name:24s} P={r['p']:.3f}  R={r['r']:.3f}  F1={r['f1']:.3f}  (tp {r['tp']}, fp {r['fp']}, fn {r['fn']}){extra}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, default=ROOT / "data" / "gold" / "exports" / "test_gold.json")
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    if not args.gold.exists():
        sys.exit(f"{args.gold} not found. Build it with `gold.py merge`.")
    sys.stdout.reconfigure(encoding="utf-8")

    gold = load_export(args.gold)
    ids = set(gold)
    ent_runs = {r: e for r in ENTITY_RUNS if (e := load_entities(r, args.version, ids)) is not None}
    tri_runs = {r: t for r in TRIPLE_RUNS if (t := load_triples(r, args.version, ids)) is not None}
    if "full" not in ent_runs or "full" not in tri_runs:
        sys.exit("Run entities.py and effects.py first (data/entities/full_*.jsonl, data/triples/full_*.jsonl).")

    results = {"gold": args.gold.resolve().relative_to(ROOT).as_posix(), "sentences": len(gold),
               "gold_entities": sum(len(g["entities"]) for g in gold.values()),
               "gold_triples": sum(len(gold_triples(g, True)) for g in gold.values()),
               "entities": {r: score_entities(gold, e) for r, e in ent_runs.items()},
               "triples_labelled": score_triples(gold, tri_runs, True),
               "triples_unlabelled": score_triples(gold, tri_runs, False)}
    full_p = results["entities"]["full"]["all"]["exact"]["p"]
    t_full, t_base = results["triples_labelled"]["full"]["f1"], results["triples_labelled"].get(BASELINE, {}).get("f1")
    results["phase1_gate"] = {"entity_precision>=0.75": full_p >= 0.75,
                              "triple_f1_beats_cooccurrence": t_base is not None and t_full > t_base}

    print(f"gold: {results['sentences']} sentences, {results['gold_entities']} entities, {results['gold_triples']} triples")
    for mode in ("exact", "partial"):
        table(f"Entities ({mode} match, events + systems)", [(r, v["all"][mode]) for r, v in results["entities"].items()])
    table("Entities by slot (full, exact)", [(s, results["entities"]["full"][s]["exact"]) for s in ("event", "system")])
    table("Triples (event, system, effect)", list(results["triples_labelled"].items()))
    table("Pairs (event, system), effect ignored", list(results["triples_unlabelled"].items()))
    print("\nPhase 1 gate:", json.dumps(results["phase1_gate"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"results_{args.gold.stem}_{args.version}.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    err = OUT_DIR / f"errors_{args.gold.stem}_{args.version}.csv"
    rows = errors(gold, ent_runs["full"], tri_runs["full"])
    write_csv(err, rows, ["kind", "id", "item", "category", "sentence"])
    print(f"-> {out.relative_to(ROOT)}\n-> {err.relative_to(ROOT)} ({len(rows)} errors to categorise)")


if __name__ == "__main__":
    main()
