"""Stage 1: sentence/word tokenization, lowercasing, stopword removal, lemmatization (spaCy),
plus regex extraction of geomagnetic index values and abbreviation definitions.

Usage:  python src/preprocess.py data/corpus/corpus_v1.jsonl [data/corpus/reference_v1.jsonl ...]

Writes data/processed/<input stem>.jsonl with one record per abstract:
  {bibcode, title, title_lemmas, abbreviations: {short: long},
   sentences: [{id, text, tokens, lemmas, indices}]}
  tokens  = lowercased word tokens, stopwords kept   (for the n-gram language model)
  lemmas  = lowercased lemmas, stopwords/punct/numbers removed   (for TF-IDF, BoW, Jaccard)
  indices = Kp / Dst / SYM-H values found by regex
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import spacy

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed"
log = logging.getLogger("preprocess")

# ponytail: single values only; ranges ("from -6 to -295 nT") keep just the first value.
# Phase 2 condition spans replace this.
# "Kp = 8+", "Kp index reached 9-", "Dst < -100 nT", "Dst index of −422 nT", "SYM-H ≤ −518 nT"
INDEX_RE = re.compile(
    r"\b(Kp|Dst|SYM-H)\b(?:\s*index)?(?:\s*values?)?(?:\s*varied)?\s*"
    r"(<=|>=|=|<|>|≤|≥|~|∼|≈|of|reached|reaching|from|below|above|exceeding|down to|up to)?\s*"
    r"([-−–]?\s?\d+(?:\.\d+)?)\s*([+\-o](?!\d))?\s*(nT)?"
)
OPS = {"<=": "<=", "≤": "<=", ">=": ">=", "≥": ">=", "=": "=", "<": "<", ">": ">",
       "~": "~", "∼": "~", "≈": "~", "reaching": "=", "from": "from", "below": "<", "down to": "<=", "above": ">", "exceeding": ">",
       "up to": "<=", "of": "=", "reached": "="}

# Abbreviation definitions: "single event upset (SEU)", "geomagnetically induced currents (GICs)"
ABBR_RE = re.compile(r"\(\s*([A-Za-z][A-Za-z0-9\-]{1,9}?)s?\s*\)")


def extract_indices(text):
    found = []
    for m in INDEX_RE.finditer(text):
        name, op, num, suffix, unit = m.groups()
        value = float(re.sub(r"\s", "", num).replace("−", "-").replace("–", "-"))
        found.append({"index": name, "op": OPS.get(op or "=", "="), "value": value,
                      "suffix": suffix or "", "unit": unit or "", "text": m.group(0).strip()})
    return found


def _initials_match(short, words):
    """Schwartz & Hearst (2003) check: short-form letters appear in order in the long form,
    and the first letter starts the first word."""
    long_form = " ".join(words).lower()
    s, i = short.lower(), len(long_form) - 1
    for j in range(len(s) - 1, -1, -1):
        c = s[j]
        if not c.isalnum():
            continue
        while i >= 0 and (long_form[i] != c or (j == 0 and i > 0 and long_form[i - 1].isalnum())):
            i -= 1
        if i < 0:
            return False
        i -= 1
    return True


def extract_abbreviations(text):
    """Map each parenthesised short form to the shortest preceding long form that matches it."""
    abbrs = {}
    for m in ABBR_RE.finditer(text):
        short = m.group(1)
        if not any(c.isupper() for c in short):
            continue
        before = re.findall(r"[\w\-]+", text[:m.start()])
        max_words = min(len(short) + 5, len(short) * 2)
        for n in range(1, min(max_words, len(before)) + 1):
            words = before[-n:]
            if words[0][0].lower() == short[0].lower() and _initials_match(short, words):
                abbrs.setdefault(short, " ".join(words))
                break
    return abbrs


def sentence_record(i, sent):
    words = [t for t in sent if not (t.is_punct or t.is_space)]
    return {
        "id": i,
        "text": sent.text.strip(),
        "tokens": [t.lower_ for t in words],
        "lemmas": [t.lemma_.lower() for t in words if not (t.is_stop or t.like_num) and t.is_alpha],
        "indices": extract_indices(sent.text),
    }


def process_file(nlp, path):
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    out = OUT_DIR / path.name
    texts = ((r["abstract"], r) for r in records)
    titles = nlp.pipe((r["title"] for r in records), batch_size=256)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for n, ((doc, rec), title) in enumerate(zip(nlp.pipe(texts, as_tuples=True, batch_size=64), titles), 1):
            f.write(json.dumps({
                "bibcode": rec["bibcode"],
                "title": rec["title"],
                "title_lemmas": [t.lemma_.lower() for t in title if not t.is_stop and t.is_alpha],
                "abbreviations": extract_abbreviations(rec["abstract"]),
                "sentences": [sentence_record(i, s) for i, s in enumerate(doc.sents)],
            }, ensure_ascii=False) + "\n")
            if n % 500 == 0:
                log.info("%s: %d/%d", path.name, n, len(records))
    tmp.replace(out)
    log.info("Wrote %d records to %s", len(records), out.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        sys.exit(f"Input not found: {', '.join(map(str, missing))}")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])  # NER is only used later, as a baseline
    for path in args.inputs:
        process_file(nlp, path)


if __name__ == "__main__":
    main()
