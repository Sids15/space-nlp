"""Stage 8: evidence retrieval. A TF-IDF index over the corpus sentences returns the top-k sentences
for an (event, system) query or for free text. BoW (raw counts, no IDF) is the PRD baseline.

Usage:
  python src/retrieval.py query geomagnetic_storm gnss          canonical pair (query expanded with its gazetteer phrases)
  python src/retrieval.py query --text "solar flare HF blackout"  free text
  python src/retrieval.py pool [--k 10]     judgment pool for the evaluation queries -> fill `relevant` y/n
  python src/retrieval.py evaluate          P@5 and R@10 for TF-IDF and BoW from the judged pool

Recall@10 is pooled recall: relevant sentences found in the top 10, over all relevant sentences
either system returned in its top 10 (the usual approximation when the full corpus is not judged).
"""

import argparse
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from domain import EVENTS, PHRASES, SYSTEMS
from relevance import write_csv

ROOT = Path(__file__).resolve().parent.parent
RET_DIR = ROOT / "data" / "retrieval"

# Written before looking at any retrieval output. Relevant = the sentence states or reports the
# effect the query describes (not merely mentions the terms).
EVAL_QUERIES = [
    "geomagnetic storm GPS positioning error",
    "ionospheric scintillation GNSS loss of lock during storms",
    "geomagnetically induced currents transformer damage",
    "solar flare HF radio blackout",
    "polar cap absorption solar proton event HF radio",
    "single event upset solar proton event satellite",
    "satellite drag geomagnetic storm orbital decay",
    "surface charging spacecraft anomaly substorm",
    "solar radio burst GPS receiver tracking",
    "power grid blackout March 1989 storm",
    "GIC mitigation blocking device",
    "storm effects on precise point positioning",
    "SBAS WAAS EGNOS performance during geomagnetic storm",
    "solar flare GNSS positioning accuracy",
    "HF communication disruption geomagnetic storm",
    "transformer harmonics geomagnetic disturbance",
    "Starlink satellites lost geomagnetic storm drag",
    "solar energetic particles damage spacecraft electronics",
    "aviation navigation ionospheric storm",
    "low latitude GIC power network",
    "BeiDou performance geomagnetic storm",
    "shortwave fadeout solar flare",
    "geomagnetic storm no significant effect GPS",
    "satellite anomalies geomagnetic activity",
]


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner", "parser"])


def lemmas(text):
    """Same normalisation as preprocess.py: lowercased lemmas, no stopwords/punctuation/numbers."""
    return [t.lemma_.lower() for t in _nlp()(text) if not (t.is_stop or t.like_num) and t.is_alpha]


def pair_query(event, system):
    if event not in EVENTS or system not in SYSTEMS:
        raise ValueError(f"unknown pair {event}, {system}")
    return lemmas(" . ".join(PHRASES[event] + PHRASES[system]))


class Index:
    def __init__(self, version="v1", weighting="tfidf"):
        path = ROOT / "data" / "relevance" / f"filtered_{version}.jsonl"
        if not path.exists():
            sys.exit(f"{path} missing. Run relevance.py first.")
        self.rows = [(r["bibcode"], s["id"], s["text"])
                     for line in path.open(encoding="utf-8") for r in [json.loads(line)] for s in r["sentences"]]
        docs = [s["lemmas"] for line in path.open(encoding="utf-8") for s in json.loads(line)["sentences"]]
        vec_cls = TfidfVectorizer if weighting == "tfidf" else CountVectorizer
        kwargs = {"sublinear_tf": True} if weighting == "tfidf" else {}
        self.vectorizer = vec_cls(analyzer=lambda x: x, **kwargs)  # input is already a lemma list
        self.matrix = normalize(self.vectorizer.fit_transform(docs))

    def search(self, query_lemmas, k=10):
        q = normalize(self.vectorizer.transform([query_lemmas]))
        scores = (self.matrix @ q.T).toarray().ravel()
        top = np.argsort(-scores, kind="stable")[:k]
        return [{"score": round(float(scores[i]), 4), "bibcode": self.rows[i][0], "sid": self.rows[i][1],
                 "text": self.rows[i][2]} for i in top if scores[i] > 0]


def pool(version, k):
    out = RET_DIR / f"judgments_{version}.csv"
    old = {}
    if out.exists():  # keep any judgments already made
        old = {(r["query_id"], r["bibcode"], r["sid"]): r["relevant"]
               for r in csv.DictReader(out.open(encoding="utf-8-sig"))}
    indexes = {w: Index(version, w) for w in ("tfidf", "bow")}
    rows = []
    for qid, query in enumerate(EVAL_QUERIES):
        seen = {}
        for name, idx in indexes.items():
            for rank, hit in enumerate(idx.search(lemmas(query), k), 1):
                seen.setdefault((hit["bibcode"], hit["sid"]), {"text": hit["text"], "systems": []})["systems"].append(f"{name}@{rank}")
        for (bib, sid), v in seen.items():
            rows.append({"query_id": qid, "query": query, "bibcode": bib, "sid": sid,
                         "relevant": old.get((str(qid), bib, str(sid)), ""), "text": v["text"]})
    RET_DIR.mkdir(parents=True, exist_ok=True)
    # system names are hidden from the judge
    write_csv(out, rows, ["query_id", "query", "bibcode", "sid", "relevant", "text"])
    print(f"{len(rows)} pooled sentences for {len(EVAL_QUERIES)} queries -> {out.relative_to(ROOT)}")


def evaluate(version):
    path = RET_DIR / f"judgments_{version}.csv"
    if not path.exists():
        sys.exit("Run `retrieval.py pool` and fill in the judgments first.")
    judged = {(r["query_id"], r["bibcode"], r["sid"]): r["relevant"].strip().lower()
              for r in csv.DictReader(path.open(encoding="utf-8-sig"))}
    missing = sum(v not in ("y", "n") for v in judged.values())
    if missing:
        sys.exit(f"{missing} pooled sentences still need a y/n in `relevant`.")
    result = {}
    for name in ("tfidf", "bow"):
        idx = Index(version, name)
        p5, r10 = [], []
        for qid, query in enumerate(EVAL_QUERIES):
            hits = idx.search(lemmas(query), 10)
            rel = [judged.get((str(qid), h["bibcode"], str(h["sid"]))) == "y" for h in hits]
            total = sum(v == "y" for (q, _, _), v in judged.items() if q == str(qid))
            p5.append(sum(rel[:5]) / 5)
            if total:
                r10.append(sum(rel) / total)
        result[name] = {"P@5": round(float(np.mean(p5)), 3), "R@10": round(float(np.mean(r10)), 3),
                        "queries": len(EVAL_QUERIES), "queries_with_relevant": len(r10)}
    print(json.dumps(result, indent=2))
    (RET_DIR / f"eval_{version}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1")
    sub = parser.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("query")
    q.add_argument("pair", nargs="*", help="event system (canonical labels)")
    q.add_argument("--text")
    q.add_argument("--k", type=int, default=10)
    q.add_argument("--bow", action="store_true", help="use the BoW baseline")
    p = sub.add_parser("pool")
    p.add_argument("--k", type=int, default=10)
    sub.add_parser("evaluate")
    args = parser.parse_args()
    if args.cmd == "query":
        if bool(args.text) == bool(args.pair) or (args.pair and len(args.pair) != 2):
            sys.exit("Give either two canonical labels (event system) or --text.")
        try:
            query = lemmas(args.text) if args.text else pair_query(*args.pair)
        except ValueError as e:
            sys.exit(str(e))
        for h in Index(args.version, "bow" if args.bow else "tfidf").search(query, args.k):
            print(f"{h['score']:.3f}  {h['bibcode']}#{h['sid']}  {h['text'][:160]}")
    elif args.cmd == "pool":
        pool(args.version, args.k)
    else:
        evaluate(args.version)


if __name__ == "__main__":
    main()
