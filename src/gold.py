"""Gold set: sample sentences for annotation, compute inter-annotator agreement, load labels.

Usage:
  python src/gold.py sample [--version v1] [--force]   draw the ~300-sentence gold sample (frozen)
  python src/gold.py tasks [--version v1]              Label Studio import files: pilot (30), dev_rest (70), test (200)
  python src/gold.py kappa A.json B.json [--out name]  Cohen's kappa between two Label Studio exports
  python src/gold.py diff A.json B.json                list every disagreement, for the discussion after kappa

Sampling never looks at our tagger's output, so recall can be measured fairly. About 150 abstracts
are drawn evenly across the 8 event x system strata; from each, one sentence that mentions an event
and a system query term (from the Week 1 ADS queries) plus one random sentence. Only ~39% of abstracts
state an event and a system in one sentence, so within each stratum those abstracts are drawn first;
otherwise the test set would hold too few possible triples to measure. (Report this bias.) One third of the
abstracts (~100 sentences) form the dev split, which rules may be tuned on; the rest (~200) are the
test split, used once at the end. No abstract is in both splits. The first 30 dev sentences are the
pilot batch for calibrating the guideline.

Annotators label spans STORM, SEP, GNSS, HF_RADIO, SATELLITES, POWER_GRID, and draw a relation
from an event span to a system span when the sentence states an effect (degradation, outage,
no_effect, mitigated). See docs/annotation_guideline.md.
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from domain import EFFECTS, EVENTS, SYSTEMS
from fetch_corpus import EVENT_TERMS, SYSTEM_TERMS

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "data" / "gold"
SEED = 42
N_ABSTRACTS = 150
DEV_FRACTION = 1 / 3
PILOT = 30
LS_LABELS = {"STORM": "geomagnetic_storm", "SEP": "sep_event", "GNSS": "gnss", "HF_RADIO": "hf_radio",
             "SATELLITES": "satellites", "POWER_GRID": "power_grid"}


def _keyword_re(terms):
    words = sorted({t.strip('"').lower() for ts in terms.values() for t in ts}, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")s?\b", re.IGNORECASE)


EVENT_RE, SYSTEM_RE = _keyword_re(EVENT_TERMS), _keyword_re(SYSTEM_TERMS)


def pick_sentences(sentences, rng):
    """One sentence with an event and a system query term (or failing that, either), plus one random."""
    pair = [s for s in sentences if EVENT_RE.search(s["text"]) and SYSTEM_RE.search(s["text"])]
    one = [s for s in sentences if EVENT_RE.search(s["text"]) or SYSTEM_RE.search(s["text"])]
    first, how = ((rng.choice(pair), "keyword_pair") if pair else
                  (rng.choice(one), "keyword_one") if one else (rng.choice(sentences), "random"))
    rest = [s for s in sentences if s is not first]
    return [(first, how)] + ([(rng.choice(rest), "random")] if rest else [])


def sample(version, force):
    out = GOLD_DIR / f"sample_{version}.jsonl"
    if out.exists() and not force:
        sys.exit(f"{out.name} is frozen (annotation may have started). Use --force only if it has not.")
    rng = random.Random(SEED)
    strata = {json.loads(l)["bibcode"]: json.loads(l)["strata"]
              for l in (ROOT / "data" / "corpus" / f"corpus_{version}.jsonl").open(encoding="utf-8")}
    records = [json.loads(l) for l in (ROOT / "data" / "relevance" / f"filtered_{version}.jsonl").open(encoding="utf-8")]

    by_stratum = defaultdict(list)
    for rec in records:
        if len(rec["sentences"]) >= 2:
            by_stratum[rng.choice(sorted(strata[rec["bibcode"]]))].append(rec)
    for recs in by_stratum.values():
        rng.shuffle(recs)
        recs.sort(key=lambda r: any(EVENT_RE.search(s["text"]) and SYSTEM_RE.search(s["text"])
                                    for s in r["sentences"]))  # stable; pop() takes these first
    chosen, names = defaultdict(list), sorted(by_stratum)
    while sum(map(len, chosen.values())) < N_ABSTRACTS and any(by_stratum.values()):
        for name in names:  # round robin: small strata give what they have, the rest fill up evenly
            if by_stratum[name] and sum(map(len, chosen.values())) < N_ABSTRACTS:
                chosen[name].append(by_stratum[name].pop())

    rows = []
    for name in names:
        rng.shuffle(chosen[name])  # draw order favours explicit abstracts; the split must not
        n_dev = round(len(chosen[name]) * DEV_FRACTION)
        for i, rec in enumerate(chosen[name]):
            for sent, how in pick_sentences(rec["sentences"], rng):
                rows.append({"id": f"{rec['bibcode']}|{sent['id']}", "bibcode": rec["bibcode"], "sid": sent["id"],
                             "text": sent["text"], "stratum": name, "split": "dev" if i < n_dev else "test",
                             "selection": how})
    rng.shuffle(rows)
    dev = [r for r in rows if r["split"] == "dev"]
    for i, r in enumerate(dev):
        r["pilot"] = i < PILOT
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "pilot": r.get("pilot", False)}, ensure_ascii=False) + "\n")
    # Label Studio import files: annotators see only the id and the text, not stratum or selection.
    for split in ("dev", "test"):
        tasks = [{"data": {"id": r["id"], "text": r["text"]}} for r in rows if r["split"] == split]
        (GOLD_DIR / f"labelstudio_{split}_{version}.json").write_text(json.dumps(tasks, indent=1, ensure_ascii=False),
                                                                       encoding="utf-8")
    counts = defaultdict(int)
    for r in rows:
        counts[(r["split"], r["selection"])] += 1
    print(f"{len(rows)} sentences from {sum(map(len, chosen.values()))} abstracts ->", out.relative_to(ROOT))
    print("per stratum:", {n: len(chosen[n]) for n in names})
    print("split x selection:", dict(counts))


# ---------- loading Label Studio exports ----------

def _trim_edges(text, start, end):
    """Word-granularity highlights swallow trailing punctuation ("storms." / "(GICs),"); drop it for both annotators."""
    while start < end and (text[start].isspace() or text[start] in ".,;:"):
        start += 1
    while start < end and (text[end - 1].isspace() or text[end - 1] in ".,;:"):
        end -= 1
    if text[start] == "(" and text[end - 1] == ")" and "(" not in text[start + 1:end - 1]:
        start, end = start + 1, end - 1  # "(GICs)" -> "GICs"
    elif text[start] == "(" and ")" not in text[start:end]:
        start += 1
    elif text[end - 1] == ")" and "(" not in text[start:end]:
        end -= 1
    return start, end


def load_export(path):
    """{task id: {"text", "entities": [(start, end, canonical)], "triples": {(event, system): effect}}}"""
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for task in tasks:
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        spans, triples = {}, {}
        for r in anns[-1]["result"]:  # latest submitted annotation
            if r.get("type") == "labels":
                v = r["value"]
                spans[r["id"]] = (*_trim_edges(task["data"]["text"], v["start"], v["end"]), LS_LABELS[v["labels"][0]])
        for r in anns[-1]["result"]:
            if r.get("type") == "relation" and r.get("labels"):
                a, b = spans.get(r["from_id"]), spans.get(r["to_id"])
                if not (a and b):
                    continue
                if a[2] in SYSTEMS and b[2] in EVENTS:  # annotator drew it system -> event
                    a, b = b, a
                if a[2] in EVENTS and b[2] in SYSTEMS and r["labels"][0] in EFFECTS:
                    triples.setdefault((a[2], b[2]), set()).add(r["labels"][0])
        out[task["data"]["id"]] = {"text": task["data"]["text"], "entities": sorted(spans.values()),
                                   "triples": {k: "|".join(sorted(v)) for k, v in triples.items()}}
    return out


def write_tasks(version):
    """Import files from the frozen sample. Annotators see only the id and the text."""
    rows = [json.loads(l) for l in (GOLD_DIR / f"sample_{version}.jsonl").open(encoding="utf-8")]
    batches = {"pilot": [r for r in rows if r["pilot"]],
               "dev_rest": [r for r in rows if r["split"] == "dev" and not r["pilot"]],
               "test": [r for r in rows if r["split"] == "test"]}
    for name, batch in batches.items():
        tasks = [{"data": {"id": r["id"], "text": r["text"]}} for r in batch]
        out = GOLD_DIR / f"labelstudio_{name}_{version}.json"
        out.write_text(json.dumps(tasks, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{len(tasks):4d} tasks -> {out.relative_to(ROOT)}")


def diff(path_a, path_b):
    a, b = load_export(path_a), load_export(path_b)
    n = 0
    for t in sorted(set(a) & set(b)):
        ea, eb = set(a[t]["entities"]), set(b[t]["entities"])
        ta, tb = a[t]["triples"], b[t]["triples"]
        if ea == eb and ta == tb:
            continue
        n += 1
        text = a[t]["text"]
        print(f"\n[{t}] {text}")
        for label, mine in (("A only", ea - eb), ("B only", eb - ea)):
            for s, e, c in sorted(mine):
                print(f"   span  {label}: '{text[s:e]}' {c}")
        for pair in sorted(set(ta) | set(tb)):
            if ta.get(pair) != tb.get(pair):
                print(f"   effect {pair[0]} -> {pair[1]}: A={ta.get(pair, 'none')}  B={tb.get(pair, 'none')}")
    print(f"\n{n} sentences with disagreements")


def _token_labels(text, entities, allowed):
    labels = []
    for m in re.finditer(r"\w+|[^\w\s]", text):
        lab = next((c for s, e, c in entities if s <= m.start() < e and c in allowed), "O")
        labels.append(lab)
    return labels


def kappa(path_a, path_b, out_name):
    a, b = load_export(path_a), load_export(path_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        sys.exit("The two exports share no annotated tasks.")
    result = {"tasks": len(shared), "only_in_a": len(set(a) - set(b)), "only_in_b": len(set(b) - set(a))}
    for slot, allowed in (("event", set(EVENTS)), ("system", set(SYSTEMS))):
        la = [l for t in shared for l in _token_labels(a[t]["text"], a[t]["entities"], allowed)]
        lb = [l for t in shared for l in _token_labels(b[t]["text"], b[t]["entities"], allowed)]
        result[f"kappa_{slot}_token"] = round(cohen_kappa_score(la, lb), 3)
        sa = {(t, s, e, c) for t in shared for s, e, c in a[t]["entities"] if c in allowed}
        sb = {(t, s, e, c) for t in shared for s, e, c in b[t]["entities"] if c in allowed}
        result[f"{slot}_span_f1_exact"] = round(2 * len(sa & sb) / (len(sa) + len(sb)), 3) if sa or sb else None
    units_a, units_b = [], []
    for t in shared:
        for pair in sorted(set(a[t]["triples"]) | set(b[t]["triples"])):
            units_a.append(a[t]["triples"].get(pair, "none"))
            units_b.append(b[t]["triples"].get(pair, "none"))
    result["effect_units"] = len(units_a)
    result["kappa_effect"] = round(cohen_kappa_score(units_a, units_b), 3) if len(set(units_a + units_b)) > 1 else None
    result["gate_passed"] = all((result[k] or 0) >= 0.6 for k in ("kappa_event_token", "kappa_system_token", "kappa_effect"))
    print(json.dumps(result, indent=2))
    (GOLD_DIR / f"kappa_{out_name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--version", default="v1")
    s.add_argument("--force", action="store_true")
    sub.add_parser("tasks").add_argument("--version", default="v1")
    d = sub.add_parser("diff")
    d.add_argument("export_a", type=Path)
    d.add_argument("export_b", type=Path)
    k = sub.add_parser("kappa")
    k.add_argument("export_a", type=Path)
    k.add_argument("export_b", type=Path)
    k.add_argument("--out", default="round1", help="name for data/gold/kappa_<out>.json")
    args = parser.parse_args()
    if args.cmd == "sample":
        sample(args.version, args.force)
    elif args.cmd == "tasks":
        write_tasks(args.version)
    elif args.cmd == "diff":
        diff(args.export_a, args.export_b)
    else:
        for p in (args.export_a, args.export_b):
            if not p.exists():
                sys.exit(f"{p} not found")
        kappa(args.export_a, args.export_b, args.out)


if __name__ == "__main__":
    main()
