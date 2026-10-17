"""Stage 4: entity tagging for EVENT and SYSTEM mentions, with canonical labels.

Usage:  python src/entities.py [--version v1] [--mode full|gazetteer|spacy_ner] [--no-wsd] [--no-synonyms]

  full       gazetteer + acronym matches, plus noun-chunk heads (POS/chunk rules); a match is widened
             to its base noun phrase when that phrase is headed by an entity word ("GPS receivers").
             Ambiguous heads go through Lesk WSD; heads without a family go through synonym grouping.
  gazetteer  baseline: exact gazetteer/acronym lookup only, no chunk rules, no WSD, no synonyms
  spacy_ner  baseline: spaCy's default NER (EVENT -> EVENT; ORG/PRODUCT/FAC -> SYSTEM)

Writes data/entities/<run name>_<version>.jsonl, one line per sentence, with character offsets
into the sentence text, plus a .summary.json with counts and the unresolved mentions.
"""

import argparse
import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans

import parsed
import synonyms
import wsd
from domain import ACRONYMS, HEADS, LABEL_OF, PHRASES, phrase_variants

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "entities"
TRIM_POS = {"DET", "PRON", "PUNCT", "ADV"}  # leading determiners/possessives/"especially" are not part of a mention
# ponytail: Lesk margin hand-set to ~one rare content word (IDF units); tune on the dev split, never test
WSD_MARGIN = 5.0
CAUSAL_HYPHEN_HEADS = {"storm", "superstorm", "substorm", "flare"}
CAUSAL_PARTICIPLES = {"induced", "driven", "caused", "generated", "related", "associated"}
QUANTIFIERS = set("several many some all both each various numerous multiple few most one such two three four five six seven "
                  "eight nine ten".split())  # ...nor are quantity words (years such as "2003" stay)
NER_MAP = {"EVENT": "EVENT", "ORG": "SYSTEM", "PRODUCT": "SYSTEM", "FAC": "SYSTEM"}
log = logging.getLogger("entities")


@dataclass
class Trigger:
    span: object  # spacy Span
    label: str
    canonical: str | None
    kind: str  # "gaz" or "head"
    wsd_key: str | None = None


class Tagger:
    def __init__(self, nlp, mode="full", use_wsd=True, use_synonyms=True, version="v1"):
        self.mode, self.use_wsd, self.use_synonyms = mode, use_wsd, use_synonyms
        self.phrases = PhraseMatcher(nlp.vocab, attr="LOWER")
        for canonical, phrases in PHRASES.items():
            self.phrases.add(canonical, [nlp.make_doc(v) for p in phrases for v in phrase_variants(p)])
        acronyms = {c: list(a) for c, a in ACRONYMS.items()}
        self.grouper, self.wsd_extra = None, {}
        if mode == "full" and use_synonyms:
            for short, canonical in synonyms.load_acronyms(version).items():
                acronyms[canonical].append(short)
            self.grouper = synonyms.load_grouper(version)
        self.idf = None
        if mode == "full" and use_wsd:
            self.wsd_extra = wsd_domain_words(self.grouper or synonyms.load_grouper(version))
            self.idf = sentence_idf(version)
        self.acronyms = PhraseMatcher(nlp.vocab, attr="ORTH")
        for canonical, words in acronyms.items():
            self.acronyms.add(canonical, [nlp.make_doc(w) for w in words])
        self.stats = Counter()
        self.unresolved = Counter()
        self.wsd_rejected = Counter()

    def __call__(self, doc):
        if self.mode == "spacy_ner":
            return [_entity(e, NER_MAP[e.label_], synonyms.gazetteer_canonical(e.text), "spacy_ner")
                    for e in doc.ents if e.label_ in NER_MAP]
        triggers = self._gazetteer(doc)
        if self.mode == "gazetteer":
            return [_entity(t.span, t.label, t.canonical, "gazetteer") for t in triggers]
        triggers += self._heads(doc, triggers)
        return self._resolve(doc, triggers)

    def _gazetteer(self, doc):
        matches = [doc[s:e] for m in (self.phrases, self.acronyms) for _, s, e in m(doc)]
        keys = {(doc[s:e].start, doc[s:e].end): doc.vocab.strings[k]
                for m in (self.phrases, self.acronyms) for k, s, e in m(doc)}
        return [Trigger(sp, LABEL_OF[keys[(sp.start, sp.end)]], keys[(sp.start, sp.end)], "gaz")
                for sp in filter_spans(matches)]

    def _heads(self, doc, gaz):
        covered = {i for t in gaz for i in range(t.span.start, t.span.end)}
        out = []
        # causal hyphen compounds: "storm-induced GICs", "flare-driven absorption" ("storm-time" is not causal)
        for tok in doc[:-2]:
            if (tok.lemma_.lower() in CAUSAL_HYPHEN_HEADS and tok.i not in covered and doc[tok.i + 1].text == "-"
                    and doc[tok.i + 2].lower_ in CAUSAL_PARTICIPLES):
                label, canonical, key = HEADS[tok.lemma_.lower()]
                out.append(Trigger(doc[tok.i:tok.i + 1], label, canonical, "head", key))
                covered.add(tok.i)
        for chunk in doc.noun_chunks:
            root = chunk.root
            if root.lemma_.lower() in HEADS and root.i not in covered:
                label, canonical, key = HEADS[root.lemma_.lower()]
                out.append(Trigger(doc[root.i:root.i + 1], label, canonical, "head", key))
        return out

    def _resolve(self, doc, triggers):
        chunks = list(doc.noun_chunks)
        protect = {i for t in triggers for i in range(t.span.start, t.span.end)}
        groups = {}
        for t in triggers:
            span = t.span
            chunk = next((c for c in chunks if c.start <= span.start and span.end <= c.end), None)
            if chunk is not None and (chunk.root.lemma_.lower() in HEADS or span.start <= chunk.root.i < span.end):
                inside = {(x.label, x.canonical) for x in triggers if chunk.start <= x.span.start and x.span.end <= chunk.end}
                coordinated = any(tok.pos_ == "CCONJ" or tok.text == "," for tok in chunk)  # "GPS and HF", "G5 storm, S2 storm"
                if len({lab for lab, _ in inside}) == 1 and not coordinated:  # never merge an EVENT and a SYSTEM
                    span = _trim(chunk, protect)
            groups.setdefault((span.start, span.end), []).append(t)

        context = [tok.lemma_.lower() for tok in doc if tok.is_alpha]
        found = []
        for (start, end), group in groups.items():
            span = doc[start:end]
            gaz = sorted((t for t in group if t.kind == "gaz"), key=lambda t: len(t.span), reverse=True)
            if gaz:
                found.append(_entity(span, gaz[0].label, gaz[0].canonical, "gazetteer"))
                continue
            head = group[0]
            if self.use_wsd and head.wsd_key:
                sense, is_domain = wsd.disambiguate(head.wsd_key, context, self.wsd_extra.get(head.wsd_key, frozenset()),
                                                  self.idf, WSD_MARGIN)
                if not is_domain:
                    self.wsd_rejected[f"{span.text} -> {sense}"] += 1
                    continue
            canonical, source = head.canonical, "chunk_head"
            if canonical is None and self.grouper is not None:
                canonical, source = self.grouper.assign(span.text, head.label), "chunk_head+w2v"
            if canonical is None:
                self.unresolved[span.text.lower()] += 1
                continue
            found.append(_entity(span, head.label, canonical, source))
        keep = {(s.start, s.end) for s in filter_spans([e["_span"] for e in found])}
        return _merge_acronyms(doc, [e for e in found if (e["_span"].start, e["_span"].end) in keep])


def wsd_domain_words(grouper, topn=100):
    """Corpus-adapted Lesk: the domain sense of each WSD key also gets the words of the corpus
    Word2Vec neighbours of its canonical label ("geomagnetic_superstorm" -> geomagnetic, superstorm)."""
    out = {}
    for key, (_, canonical, _) in ((h[2], h) for h in HEADS.values() if h[2]):
        words = {w for n, _ in grouper.wv.similar_by_vector(grouper.prototypes[canonical], topn=topn)
                 for w in n.split("_") if w.isalpha() and len(w) > 2}
        out[key] = frozenset(wsd.normalize(words))
    return out


def sentence_idf(version):
    """IDF of each lemma over the general-astrophysics reference sentences (weights for Lesk overlaps).
    A background corpus, not our own: in-domain IDF would down-weight exactly the domain words."""
    df, n = Counter(), 0
    for line in (ROOT / "data" / "processed" / f"reference_{version}.jsonl").open(encoding="utf-8"):
        for s in json.loads(line)["sentences"]:
            df.update(set(s["lemmas"]))
            n += 1
    return {w: math.log(n / c) for w, c in df.items()}


def _trim(span, protect=frozenset()):
    """Drop leading determiners/adverbs/quantity words and trailing punctuation, but never a
    token that belongs to a gazetteer or head match ("Geomagnetically induced currents")."""
    doc, start, end = span.doc, span.start, span.end
    while start < end - 1 and start not in protect and (doc[start].pos_ in TRIM_POS or doc[start].lower_ in QUANTIFIERS
                                                                     or _is_count(doc[start])):
        start += 1
    while end > start + 1 and doc[end - 1].pos_ == "PUNCT":
        end -= 1
    return doc[start:end]


def _merge_acronyms(doc, found):
    """'Global Positioning System (GPS)' is one mention: join a name and the bracketed acronym right after it."""
    found = sorted(found, key=lambda e: e["_span"].start)
    out = []
    for e in found:
        prev = out[-1]["_span"] if out else None
        sp = e["_span"]
        if (prev is not None and out[-1]["canonical"] == e["canonical"] and sp.start == prev.end + 1
                and doc[prev.end].text == "(" and sp.end < len(doc) and doc[sp.end].text == ")"):
            out[-1] = _entity(doc[prev.start:sp.end + 1], out[-1]["label"], out[-1]["canonical"], out[-1]["source"])
        else:
            out.append(e)
    return out


def _is_count(tok):
    """'40 satellites' drops the count; '2003 Halloween storm' keeps the year."""
    return tok.like_num and not (len(tok.text) == 4 and tok.text.isdigit() and 1700 <= int(tok.text) <= 2099)


def _entity(span, label, canonical, source):
    return {"_span": span, "start": span.start_char, "end": span.end_char, "text": span.text,
            "label": label, "canonical": canonical, "source": source}


def run_name(mode, use_wsd, use_synonyms):
    return "_".join([mode] + (["nowsd"] if not use_wsd else []) + (["nosyn"] if not use_synonyms else []))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--mode", choices=("full", "gazetteer", "spacy_ner"), default="full")
    parser.add_argument("--no-wsd", action="store_true")
    parser.add_argument("--no-synonyms", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sentences = parsed.load_sentences(args.version)
    tagger = Tagger(parsed.nlp(), args.mode, not args.no_wsd, not args.no_synonyms, args.version)
    name = run_name(args.mode, tagger.use_wsd, tagger.use_synonyms)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}_{args.version}.jsonl"
    counts, sources = Counter(), Counter()
    with out.open("w", encoding="utf-8") as f:
        for bibcode, sid, doc in sentences:
            ents = tagger(doc)
            for e in ents:
                e.pop("_span")
                counts[e["canonical"]] += 1
                sources[e["source"]] += 1
            f.write(json.dumps({"bibcode": bibcode, "sid": sid, "text": doc.text, "entities": ents},
                               ensure_ascii=False) + "\n")
    summary = {"run": name, "sentences": len(sentences), "mentions": sum(counts.values()),
               "by_canonical": dict(counts.most_common()), "by_source": dict(sources.most_common()),
               "wsd_rejected": dict(tagger.wsd_rejected.most_common(30)),
               "unresolved_top": dict(tagger.unresolved.most_common(30))}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("%s", json.dumps({k: summary[k] for k in ("run", "mentions", "by_canonical", "by_source")}))
    log.info("Wrote %s", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
