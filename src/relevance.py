"""Stage 2: relevance filter. Scores every abstract by its similarity to the nearest seed abstracts:
TF-IDF cosine is the filter; BoW cosine and Jaccard are reported alongside it for comparison.

Nearest-seed scoring (mean of the top-3 seed similarities) is used instead of a single seed
centroid: the centroid is dominated by the GNSS-heavy seeds and pushed power-grid papers down.

Usage:
  python src/relevance.py [--version v1] [--threshold 0.137]   score, filter, write cut-off sample
  python src/relevance.py --check                             precision of the hand-checked sample

Outputs (data/relevance/):
  scores_<v>.csv         every abstract with its three scores, ranked
  cutoff_sample_<v>.csv  50 abstracts nearest the threshold; fill the `relevant` column with y/n
  filtered_<v>.jsonl     preprocessed records that pass (seeds always pass)
  filter_<v>.meta.json   threshold, counts, seed leave-one-out scores
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
REL_DIR = ROOT / "data" / "relevance"
SEED_FILE = ROOT / "data" / "seed_bibcodes.txt"
SAMPLE_SIZE = 50
TOP_K = 3
log = logging.getLogger("relevance")


def doc_lemmas(rec):
    return rec["title_lemmas"] + [lem for s in rec["sentences"] for lem in s["lemmas"]]


def top_k_mean(sim, seed_idx):
    """sim: (docs x seeds). A seed never counts its own similarity, so seed scores are leave-one-out."""
    sim = sim.copy()
    sim[seed_idx, np.arange(len(seed_idx))] = 0.0
    return np.sort(sim, axis=1)[:, -TOP_K:].mean(axis=1)


def cosine_score(matrix, seed_idx):
    unit = normalize(matrix)
    return top_k_mean((unit @ unit[seed_idx].T).toarray(), seed_idx)


def jaccard_score(lemma_sets, seed_idx):
    seed_sets = [lemma_sets[i] for i in seed_idx]
    sim = np.array([[len(d & s) / len(d | s) if d else 0.0 for s in seed_sets] for d in lemma_sets])
    return top_k_mean(sim, seed_idx)


def score(records, seeds):
    texts = [" ".join(doc_lemmas(r)) for r in records]
    index = {r["bibcode"]: i for i, r in enumerate(records)}
    missing = [s for s in seeds if s not in index]
    if missing:
        sys.exit(f"Seed bibcodes not in corpus: {missing}")
    seed_idx = [index[s] for s in seeds]
    tfidf = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(texts)
    bow = CountVectorizer(min_df=2).fit_transform(texts)
    return {
        "tfidf": (tf := cosine_score(tfidf, seed_idx)),
        "bow": cosine_score(bow, seed_idx),
        "jaccard": jaccard_score([set(doc_lemmas(r)) for r in records], seed_idx),
        "seed_loo": [float(tf[i]) for i in seed_idx],
    }


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig opens cleanly in Excel
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(version, threshold):
    src = ROOT / "data" / "processed" / f"corpus_{version}.jsonl"
    if not src.exists():
        sys.exit(f"{src} missing. Run preprocess.py first.")
    records = [json.loads(line) for line in src.open(encoding="utf-8")]
    seeds = SEED_FILE.read_text(encoding="utf-8").split()
    s = score(records, seeds)
    seed_set = set(seeds)

    order = np.argsort(-s["tfidf"], kind="stable")
    rows = [{"rank": rank + 1, "bibcode": records[i]["bibcode"], "tfidf": round(float(s["tfidf"][i]), 4),
             "bow": round(float(s["bow"][i]), 4), "jaccard": round(float(s["jaccard"][i]), 4),
             "seed": records[i]["bibcode"] in seed_set, "title": records[i]["title"]}
            for rank, i in enumerate(order)]
    REL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(REL_DIR / f"scores_{version}.csv", rows, list(rows[0]))

    kept = [r for r in rows if r["tfidf"] >= threshold or r["seed"]]
    cut = sum(r["tfidf"] >= threshold for r in rows)  # threshold position, ignoring seeds kept below it
    half = SAMPLE_SIZE // 2
    sample_path = REL_DIR / f"cutoff_sample_{version}.csv"
    labels = ({r["bibcode"]: r for r in csv.DictReader(sample_path.open(encoding="utf-8-sig"))}
              if sample_path.exists() else {})  # never discard hand labels on a re-run
    sample = [dict(r, side="above" if r["tfidf"] >= threshold else "below",
                   relevant=labels.get(r["bibcode"], {}).get("relevant", ""),
                   note=labels.get(r["bibcode"], {}).get("note", ""),
                   abstract=" ".join(x["text"] for x in records[order[r["rank"] - 1]]["sentences"]))
              for r in rows[max(0, cut - half):cut + half] if not r["seed"]]
    write_csv(sample_path, sample, ["bibcode", "tfidf", "side", "relevant", "note", "title", "abstract"])

    keep_ids = {r["bibcode"] for r in kept}
    out = REL_DIR / f"filtered_{version}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            if rec["bibcode"] in keep_ids:
                f.write(json.dumps(dict(rec, relevance=round(float(s["tfidf"][i]), 4)), ensure_ascii=False) + "\n")

    corr = np.corrcoef([s["tfidf"], s["bow"], s["jaccard"]])
    meta = {
        "version": version, "threshold": threshold, "corpus": len(records), "kept": len(kept),
        "seeds": len(seeds),
        "seed_leave_one_out_tfidf": {"min": round(min(s["seed_loo"]), 4),
                                     "median": round(float(np.median(s["seed_loo"])), 4)},
        "pearson": {"tfidf_bow": round(corr[0, 1], 3), "tfidf_jaccard": round(corr[0, 2], 3)},
        "tfidf_quantiles": {q: round(float(np.quantile(s["tfidf"], q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
    }
    (REL_DIR / f"filter_{version}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("%s", json.dumps(meta))
    log.info("Kept %d/%d abstracts -> %s", len(kept), len(records), out.relative_to(ROOT))


def check(version):
    path = REL_DIR / f"cutoff_sample_{version}.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    labelled = [r for r in rows if r["relevant"].strip().lower() in ("y", "n")]
    if len(labelled) < len(rows):
        sys.exit(f"{len(rows) - len(labelled)} rows in {path.name} still need a y/n in `relevant`.")
    for side in ("above", "below"):
        group = [r for r in labelled if r["side"] == side]
        rel = sum(r["relevant"].strip().lower() == "y" for r in group)
        print(f"{side:5s} threshold: {rel}/{len(group)} relevant ({rel / max(len(group), 1):.0%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--threshold", type=float, default=0.137, help="minimum top-3 seed TF-IDF cosine")
    parser.add_argument("--check", action="store_true", help="report precision of the hand-checked sample")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    check(args.version) if args.check else run(args.version, args.threshold)


if __name__ == "__main__":
    main()
