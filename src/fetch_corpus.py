"""Week 1: pull space-weather abstracts from NASA ADS and freeze them as versioned JSONL.

Usage:  python src/fetch_corpus.py [--version v1] [--max-per-query 2000] [--force] [--reference]

--reference pulls the general-astrophysics comparison corpus instead (reference_<version>.jsonl).

Every query string is logged in the meta file, and every raw API response is cached
under data/raw/, so the corpus can be rebuilt exactly. Delete data/raw/ to re-fetch.
"""

import argparse
import hashlib
import html
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CORPUS_DIR = ROOT / "data" / "corpus"
ADS_URL = "https://api.adsabs.harvard.edu/v1/search/query"
FIELDS = "bibcode,title,abstract,year,doi,doctype"
PAGE_SIZE = 2000  # ADS maximum rows per request
MIN_WORDS = 50
YEARS = "[2000 TO 2026]"

# Domain restriction from the PRD: 2 event families x 4 system families.
EVENT_TERMS = {
    "geomagnetic_storm": ['"geomagnetic storm"', '"geomagnetic storms"', '"magnetic storm"',
                          '"geomagnetic disturbance"', '"coronal mass ejection"',
                          '"coronal mass ejections"'],
    # Bare "CME"/"SEP" are avoided: they match unrelated acronyms (e.g. cysteine methyl ester).
    "sep_event": ['"solar energetic particle"', '"solar energetic particles"', '"solar proton event"',
                  '"solar particle event"', '"SEP event"', '"SEP events"', '"solar flare"',
                  '"solar flares"'],
}
SYSTEM_TERMS = {
    "gnss": ["GNSS", "GPS", '"global positioning system"', '"satellite navigation"',
             "GLONASS", "BeiDou"],
    "hf_radio": ['"HF radio"', '"high frequency radio"', '"HF communication"',
                 '"radio blackout"', '"shortwave fadeout"', '"polar cap absorption"'],
    "satellites": ['"spacecraft charging"', '"surface charging"', '"single event upset"',
                   '"single event effect"', '"satellite anomaly"', '"satellite anomalies"',
                   '"satellite drag"', '"atmospheric drag"', '"orbital decay"'],
    "power_grid": ['"geomagnetically induced current"', '"geomagnetically induced currents"',
                   "GIC", '"power grid"', '"power grids"', '"power system"',
                   '"power transformer"', '"power transformers"'],  # bare "transformer" hits ML papers
}

# General-astrophysics comparison text for the Week 2 language model (no space-weather overlap).
# Sorted by citations so it spans all years instead of bunching at the start of the bibcode order.
REFERENCE_QUERY = (
    'abs:(galaxy OR galaxies OR "star formation" OR exoplanet OR "stellar population" OR '
    'cosmology OR "active galactic nuclei") AND year:[2000 TO 2026] AND doctype:article '
    'AND collection:astronomy NOT abs:(geomagnetic OR "space weather" OR ionosphere OR '
    'ionospheric OR magnetosphere OR "solar wind")'
)
REFERENCE_SORT = "citation_count desc"

# ponytail: stopword-ratio language check, swap for a langdetect model if non-English slips through
EN_STOPWORDS = frozenset(
    "the of and in to a is that for on with by are as from this at be we an which or "
    "these its was were has have it during than also between".split()
)

log = logging.getLogger("fetch_corpus")


def build_queries():
    """One ADS query per (event family, system family) pair, so every record keeps its stratum."""
    queries = {}
    for event, e_terms in EVENT_TERMS.items():
        for system, s_terms in SYSTEM_TERMS.items():
            queries[f"{event}|{system}"] = (
                f"abs:({' OR '.join(e_terms)}) AND abs:({' OR '.join(s_terms)}) "
                f"AND year:{YEARS} AND doctype:(article OR inproceedings)"
            )
    return queries


def load_token():
    """Read ADS_TOKEN from the environment, falling back to the project .env file."""
    token = os.environ.get("ADS_TOKEN")
    env_file = ROOT / ".env"
    if not token and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "ADS_TOKEN":
                token = value.strip().strip("\"'")
    if not token:
        sys.exit("ADS_TOKEN not found. Add 'ADS_TOKEN=...' to .env in the project root.")
    return token


def fetch_page(query, start, token, sort="bibcode asc", retries=4):
    """Fetch one page of results, using the on-disk cache when present."""
    params = {"q": query, "fl": FIELDS, "rows": PAGE_SIZE, "start": start, "sort": sort}
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    cache = RAW_DIR / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    request = urllib.request.Request(
        f"{ADS_URL}?{urllib.parse.urlencode(params)}",
        headers={"Authorization": f"Bearer {token}"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                remaining = resp.headers.get("X-RateLimit-Remaining")
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as err:
            if err.code == 401:
                sys.exit("ADS rejected the token (401). Check ADS_TOKEN in .env.")
            if err.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            wait = int(err.headers.get("Retry-After") or 2 ** (attempt + 2))
            log.warning("HTTP %s, retrying in %ss", err.code, wait)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as err:
            if attempt == retries - 1:
                raise
            log.warning("Network error (%s), retrying", err)
            time.sleep(2 ** (attempt + 2))

    if remaining is not None:
        log.info("ADS requests remaining today: %s", remaining)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(cache)  # atomic, so a crash never leaves a half-written cache file
    return data


def clean_text(text):
    """Strip ADS markup (<SUB>, <P />, entities) and collapse whitespace."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def is_english(text, threshold=0.15):
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and sum(w in EN_STOPWORDS for w in words) / len(words) >= threshold


def to_record(doc):
    return {
        "bibcode": doc["bibcode"],
        "title": clean_text((doc.get("title") or [""])[0]),
        "abstract": clean_text(doc.get("abstract")),
        "year": int(doc["year"]) if doc.get("year") else None,
        "doi": [d.lower() for d in doc.get("doi") or []],
        "doctype": doc.get("doctype"),
        "strata": [],
    }


def merge_and_filter(results):
    """results: iterable of (stratum, ADS doc). Dedup by bibcode and DOI, then filter."""
    by_bibcode, doi_owner, stats = {}, {}, {"raw": 0, "duplicates": 0}
    for stratum, doc in results:
        stats["raw"] += 1
        rec = to_record(doc)
        owner = by_bibcode.get(rec["bibcode"]) or next(
            (by_bibcode[doi_owner[d]] for d in rec["doi"] if d in doi_owner), None)
        if owner is not None:
            stats["duplicates"] += 1
            if stratum not in owner["strata"]:
                owner["strata"].append(stratum)
            continue
        rec["strata"].append(stratum)
        by_bibcode[rec["bibcode"]] = rec
        doi_owner.update({d: rec["bibcode"] for d in rec["doi"]})

    kept, short, non_en = [], 0, 0
    for rec in by_bibcode.values():
        if len(rec["abstract"].split()) < MIN_WORDS:
            short += 1
        elif not is_english(rec["abstract"]):
            non_en += 1
        else:
            kept.append(rec)
    stats.update(unique=len(by_bibcode), dropped_short=short, dropped_non_english=non_en, kept=len(kept))
    return sorted(kept, key=lambda r: r["bibcode"]), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1", help="corpus version tag (default v1)")
    parser.add_argument("--max-per-query", type=int, default=2000)
    parser.add_argument("--force", action="store_true", help="overwrite an existing frozen version")
    parser.add_argument("--reference", action="store_true", help="fetch the astrophysics reference corpus")
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+", args.version):
        sys.exit("--version may only contain letters, digits, '_', '-' and '.'")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    name = "reference" if args.reference else "corpus"
    out = CORPUS_DIR / f"{name}_{args.version}.jsonl"
    meta_out = CORPUS_DIR / f"{name}_{args.version}.meta.json"
    if out.exists() and not args.force:
        sys.exit(f"{out.name} is frozen. Pick a new --version or pass --force.")

    token = load_token()
    queries = {"reference": REFERENCE_QUERY} if args.reference else build_queries()
    sort = REFERENCE_SORT if args.reference else "bibcode asc"
    results, hits = [], {}
    for stratum, query in queries.items():
        start, total = 0, None
        while start < args.max_per_query and (total is None or start < total):
            body = fetch_page(query, start, token, sort)["response"]
            total = body["numFound"]
            results.extend((stratum, d) for d in body["docs"])
            start += PAGE_SIZE
        hits[stratum] = total
        log.info("%-32s %5d hits", stratum, total)

    records, stats = merge_and_filter(results)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta = {
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": ADS_URL,
        "fields": FIELDS,
        "sort": sort,
        "max_per_query": args.max_per_query,
        "min_words": MIN_WORDS,
        "queries": queries,
        "hits_per_query": hits,
        "stats": stats,
    }
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Stats: %s", stats)
    log.info("Wrote %d abstracts to %s", len(records), out.relative_to(ROOT))


if __name__ == "__main__":
    main()
