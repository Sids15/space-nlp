"""Stage 5: synonym grouping. Maps surface terms to the canonical labels in domain.py.

Usage:  python src/synonyms.py [--version v1] [--glove]

Three sources, each usable on its own so they can be compared:
  acronyms  corpus abbreviation definitions ("Indian Regional Navigation Satellite System (IRNSS)")
            whose long form contains a gazetteer phrase -> the acronym joins that canonical label
  wordnet   the term's head noun is, or inherits from, a WordNet anchor synset (the "WordNet only" baseline)
  w2v       Word2Vec trained on the corpus (with collocation phrases such as geomagnetic_storm);
            a term joins the canonical label whose prototype vector is closest, if close enough
--glove compares the corpus Word2Vec against pretrained GloVe (coverage and nearest neighbours).

Writes data/lexicon/: w2v_<v>.txt (plain word2vec text format), lexicon_<v>.json
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phrases
from nltk.corpus import wordnet as wn

from domain import ACRONYMS, LABEL_OF, PHRASES

ROOT = Path(__file__).resolve().parent.parent
LEX_DIR = ROOT / "data" / "lexicon"
SEED = 42
W2V_MIN_SIM = 0.55  # cosine to the nearest prototype
W2V_MARGIN = 0.05   # ...and this much ahead of the runner-up
MIN_ACRONYM_LEN = 3  # two-letter author acronyms (SC, GS, SF) collide with unrelated uses
MIN_ACRONYM_SUPPORT = 2
ACRONYM_STOPLIST = {"THEMIS"}  # mission name whose long form ends in "substorms"
WORDNET_ANCHORS = {
    "geomagnetic_storm": ["magnetic_storm.n.01"],
    "sep_event": ["solar_flare.n.01"],
    "gnss": ["global_positioning_system.n.01", "navigation.n.01"],
    "hf_radio": ["radio.n.01", "radio.n.03", "radio_receiver.n.01"],
    "satellites": ["satellite.n.01", "spacecraft.n.01"],
    "power_grid": ["power_system.n.01", "transformer.n.01", "substation.n.01"],
}
log = logging.getLogger("synonyms")


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


# ---------- acronyms ----------

def gazetteer_canonical(text, head_only=False):
    """Canonical label whose longest gazetteer phrase occurs in text (or ends it, if head_only)."""
    t = " " + " ".join(tokenize(text)) + " "
    hits = [(len(p), c) for c, phrases in PHRASES.items() for p in phrases
            if any((t.endswith(f" {v} ") if head_only else f" {v} " in t) for v in (p, p + "s"))]
    return max(hits)[1] if hits else None


def learn_acronyms(processed_path):
    """Acronym -> canonical, by majority vote over its corpus definitions. The long form must end
    with a gazetteer phrase ("Indian Regional Navigation Satellite System"), so project and mission
    names that merely mention one ("... Geomagnetic storms and Energetic Radiation (PAGER)") are skipped."""
    votes = defaultdict(Counter)
    for line in processed_path.open(encoding="utf-8"):
        for short, long_form in json.loads(line)["abbreviations"].items():
            canonical = gazetteer_canonical(long_form, head_only=True)
            if canonical:
                votes[short][canonical] += 1
    known = {a for acr in ACRONYMS.values() for a in acr}
    return {short: c.most_common(1)[0][0] for short, c in sorted(votes.items())
            if short not in known | ACRONYM_STOPLIST and len(short) >= MIN_ACRONYM_LEN and sum(c.values()) >= MIN_ACRONYM_SUPPORT}


# ---------- WordNet ----------

@lru_cache(maxsize=None)
def _anchor_index():
    return {wn.synset(s): c for c, names in WORDNET_ANCHORS.items() for s in names}


def assign_wordnet(term):
    """Canonical label if the term's head noun is, or is a kind of, an anchor synset."""
    words = tokenize(term)
    anchors = _anchor_index()
    for n in range(min(3, len(words)), 0, -1):  # try "global positioning system" before "system"
        for synset in wn.synsets("_".join(words[-n:]), pos="n"):
            for s in [synset, *synset.closure(lambda x: x.hypernyms())]:
                if s in anchors:
                    return anchors[s]
    return None


# ---------- Word2Vec ----------

class W2VGrouper:
    def __init__(self, wv):
        self.wv = wv
        self.prototypes = {}
        for canonical, phrase_list in PHRASES.items():
            vecs = [v for p in phrase_list if (v := self.vector(p)) is not None]
            if vecs:
                self.prototypes[canonical] = _unit(np.mean(vecs, axis=0))

    def phrase(self, words):
        """Greedy longest match of multiword phrases (joined with '_') against the vocabulary."""
        out, i = [], 0
        while i < len(words):
            for n in (3, 2, 1):
                tok = "_".join(words[i:i + n])
                if n == 1 or (i + n <= len(words) and tok in self.wv):
                    out.append(tok)
                    i += n
                    break
        return out

    def vector(self, term):
        toks = [t for t in self.phrase(tokenize(term)) if t in self.wv]
        return _unit(np.mean([self.wv[t] for t in toks], axis=0)) if toks else None

    def scores(self, term):
        v = self.vector(term)
        return {} if v is None else {c: float(v @ p) for c, p in self.prototypes.items()}

    def assign(self, term, label=None):
        """Nearest canonical label (restricted to EVENT or SYSTEM labels if given), or None."""
        ranked = sorted(((s, c) for c, s in self.scores(term).items() if label is None or LABEL_OF[c] == label),
                        reverse=True)
        if not ranked or ranked[0][0] < W2V_MIN_SIM:
            return None
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < W2V_MARGIN:
            return None
        return ranked[0][1]


def _unit(v):
    return v / (np.linalg.norm(v) or 1.0)


def corpus_sentences(processed_path):
    for line in processed_path.open(encoding="utf-8"):
        for s in json.loads(line)["sentences"]:
            toks = tokenize(" ".join(s["tokens"]))
            if toks:
                yield toks


def train(version):
    processed = ROOT / "data" / "processed" / f"corpus_{version}.jsonl"
    if not processed.exists():
        sys.exit(f"{processed} missing. Run preprocess.py first.")
    sents = list(corpus_sentences(processed))  # all 3k abstracts: more text for the embeddings
    bigram = Phrases(sents, min_count=5, threshold=10, connector_words=ENGLISH_CONNECTOR_WORDS)
    trigram = Phrases(bigram[sents], min_count=5, threshold=10, connector_words=ENGLISH_CONNECTOR_WORDS)
    phrased = list(trigram[bigram[sents]])
    # workers=1 + fixed seed keeps training deterministic
    model = Word2Vec(phrased, vector_size=100, window=5, min_count=3, sg=1, epochs=20, seed=SEED, workers=1)
    return model.wv, processed


def load_grouper(version):
    path = LEX_DIR / f"w2v_{version}.txt"
    if not path.exists():
        sys.exit(f"{path} missing. Run: python src/synonyms.py --version {version}")
    return W2VGrouper(KeyedVectors.load_word2vec_format(str(path), binary=False))


def load_acronyms(version):
    path = LEX_DIR / f"lexicon_{version}.json"
    return json.loads(path.read_text(encoding="utf-8"))["acronyms"] if path.exists() else {}


def neighbours(wv, canonical_vec, n=10):
    return [w for w, _ in wv.similar_by_vector(canonical_vec, topn=n)]


def glove_comparison(grouper):
    import gensim.downloader
    glove = gensim.downloader.load("glove-wiki-gigaword-100")  # ~130 MB, cached in ~/gensim-data
    words = sorted({w for phrases in PHRASES.values() for p in phrases for w in tokenize(p)})
    out = {"glove_oov_rate": round(sum(w not in glove for w in words) / len(words), 3),
           "w2v_oov_rate": round(sum(w not in grouper.wv for w in words) / len(words), 3),
           "neighbours": {}}
    for canonical, phrases in PHRASES.items():
        gvecs = [glove[w] for p in phrases for w in tokenize(p) if w in glove]
        out["neighbours"][canonical] = {
            "w2v": neighbours(grouper.wv, grouper.prototypes[canonical]),
            "glove": neighbours(glove, _unit(np.mean(gvecs, axis=0))) if gvecs else [],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--glove", action="store_true", help="also compare against pretrained GloVe")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("gensim").setLevel(logging.WARNING)

    wv, processed = train(args.version)
    LEX_DIR.mkdir(parents=True, exist_ok=True)
    wv.save_word2vec_format(str(LEX_DIR / f"w2v_{args.version}.txt"), binary=False)
    grouper = W2VGrouper(wv)

    lexicon = {
        "acronyms": learn_acronyms(processed),
        "w2v_settings": {"min_sim": W2V_MIN_SIM, "margin": W2V_MARGIN, "vocab": len(wv)},
        "w2v_neighbours": {c: neighbours(wv, v, 15) for c, v in grouper.prototypes.items()},
    }
    if args.glove:
        lexicon["glove_comparison"] = glove_comparison(grouper)
    (LEX_DIR / f"lexicon_{args.version}.json").write_text(json.dumps(lexicon, indent=2), encoding="utf-8")
    log.info("Vocab %d; %d learned acronyms: %s", len(wv), len(lexicon["acronyms"]), lexicon["acronyms"])
    for c, ns in lexicon["w2v_neighbours"].items():
        log.info("%-18s %s", c, ", ".join(ns[:10]))


if __name__ == "__main__":
    main()
