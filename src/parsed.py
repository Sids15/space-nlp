"""Parse every sentence of the filtered corpus once with the full spaCy pipeline and cache it.

Usage:  python src/parsed.py [--version v1]
Other stages call load_sentences(version) and get (bibcode, sentence_id, Doc) triples back.
The cache is data/processed/filtered_<version>.spacy (spaCy DocBin).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import spacy
from spacy.tokens import DocBin

ROOT = Path(__file__).resolve().parent.parent
MODEL = "en_core_web_sm"
log = logging.getLogger("parsed")
_nlp = None


def nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(MODEL)  # full pipeline: tagger, parser, lemmatizer, NER (NER = baseline only)
    return _nlp


def cache_path(version):
    return ROOT / "data" / "processed" / f"filtered_{version}.spacy"


def build(version):
    src = ROOT / "data" / "relevance" / f"filtered_{version}.jsonl"
    if not src.exists():
        sys.exit(f"{src} missing. Run relevance.py first.")
    items = [(s["text"], {"bibcode": r["bibcode"], "sid": s["id"]})
             for line in src.open(encoding="utf-8") for r in [json.loads(line)] for s in r["sentences"]]
    db = DocBin(store_user_data=True)
    for doc, ctx in nlp().pipe(items, as_tuples=True, batch_size=128):
        doc.user_data.update(ctx)
        db.add(doc)
    out = cache_path(version)
    tmp = out.with_suffix(".tmp")
    db.to_disk(tmp)
    tmp.replace(out)
    log.info("Parsed %d sentences -> %s", len(items), out.relative_to(ROOT))


def load_sentences(version):
    path = cache_path(version)
    if not path.exists():
        sys.exit(f"{path} missing. Run: python src/parsed.py --version {version}")
    return [(d.user_data["bibcode"], d.user_data["sid"], d)
            for d in DocBin().from_disk(path).get_docs(nlp().vocab)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(args.version)


if __name__ == "__main__":
    main()
