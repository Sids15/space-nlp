"""Stage 6: Lesk-style word sense disambiguation for ambiguous chunk heads
("storm", "flare", "drag", "grid", "charging").

Extended (simplified) Lesk: each sense gets a signature built from its WordNet gloss, examples,
lemma names and the glosses of its hypernyms and hyponyms. The sense whose signature overlaps the
sentence context most wins. On a tie the domain sense wins, because the corpus has already been
filtered to space-weather abstracts (a domain most-frequent-sense backoff).

Corpus adaptation: WordNet's weather sense of "storm" has a 53-word signature against 15 for
magnetic_storm, and domain phrases share words with it ("space weather", "solar wind", "electric
currents"). So the caller can pass extra domain-signature words, which the tagger takes from the
corpus Word2Vec neighbours of the matching canonical label. Overlaps are also weighted by IDF from a
general-astrophysics background corpus, so generic words the weather sense inherits from its hyponym glosses ("strong", "condition",
"result", "scale") count for little, while rare weather words ("hurricane", "rain") still decide.

"charging" has no WordNet noun entry, so it gets short hand-written glosses instead.
("event" needs no WSD: only family-modified phrases such as "proton event" are tagged, via the gazetteer.)
"""

import re
from functools import lru_cache

from nltk.corpus import wordnet as wn
from spacy.lang.en.stop_words import STOP_WORDS

try:
    wn.ensure_loaded()
except LookupError:
    raise SystemExit("WordNet data missing. Run: python -c \"import nltk; nltk.download('wordnet')\"")

DOMAIN_SYNSET = {"storm": "magnetic_storm.n.01", "flare": "solar_flare.n.01",
                 "drag": "drag.n.01", "grid": "power_system.n.01"}
CUSTOM = {
    "charging": {
        "domain": "spacecraft satellite surface internal dielectric charging electrostatic discharge "
                  "arcing plasma energetic electron injection geosynchronous orbit anomaly",
        "other": ["battery charging electric vehicle electricity station power supply cell",
                  "accusation legal charge fee price payment cost"],
    },
}


def normalize(words):
    return {wn.morphy(w) or w for w in words if w not in STOP_WORDS and len(w) > 2}


def _words(text):
    return normalize(re.findall(r"[a-z]+", text.lower()))


def _signature(synset):
    parts = [synset.definition(), *synset.examples(), *(l.replace("_", " ") for l in synset.lemma_names())]
    for rel in synset.hypernyms() + synset.hyponyms():
        parts += [rel.definition(), *(l.replace("_", " ") for l in rel.lemma_names())]
    return _words(" ".join(parts))


@lru_cache(maxsize=None)
def senses(key):
    """[(sense name, signature, is_domain)] for an ambiguous key."""
    if key in DOMAIN_SYNSET:
        domain = DOMAIN_SYNSET[key]
        out = [(domain, _signature(wn.synset(domain)), True)]
        out += [(s.name(), _signature(s), False) for s in wn.synsets(key, pos="n") if s.name() != domain]
        return out
    custom = CUSTOM[key]
    out = [(f"{key}.domain", _words(custom["domain"]), True)]
    out += [(f"{key}.other{i}", _words(g), False) for i, g in enumerate(custom["other"])]
    out += [(s.name(), _signature(s), False) for s in wn.synsets(key, pos="n")]
    return out


def disambiguate(key, context_lemmas, domain_extra=frozenset(), idf=None, margin=0.0):
    """Returns (best sense name, is_domain). context_lemmas: lowercased lemmas of the sentence;
    domain_extra: extra words for the domain sense's signature (corpus adaptation);
    idf: {word: weight} for overlap words (unseen words get the maximum weight). None = plain counts.
    margin: an off-domain sense must beat the domain sense by this much (0 = plain Lesk)."""
    context = {w for w in context_lemmas if w not in STOP_WORDS and w != key}
    top = max(idf.values()) if idf else 1.0

    def weight(words):
        return sum(idf.get(w, top) for w in words) if idf else len(words)

    scored = [(weight((sig | domain_extra if is_domain else sig) & context), is_domain, name)
              for name, sig, is_domain in senses(key)]
    domain = max(s for s in scored if s[1])
    other = max((s for s in scored if not s[1]), default=None)
    if other is None or other[0] < domain[0] + margin:  # ties (and near-ties) go to the domain sense
        return domain[2], True
    return other[2], False
