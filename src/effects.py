"""Stage 7: effect extraction. Turns (EVENT, SYSTEM) mention pairs in one sentence into
(event, system, effect) triples using the dependency path between them, and writes the triple store.

Usage:  python src/effects.py [--version v1] [--baseline] [--no-negation] [--no-wsd] [--no-synonyms]

A pair becomes a triple only if the dependency path between the two mentions (at most MAX_PATH
edges), or the words governing that path up to the nearest verb ("[no significant effect] of the
storm on HF radio", "[mitigate] the GIC effects ... during storms"), carries an effect cue: an effect verb or noun from the word lists below, or a neutral causal
verb ("cause", "induce", "lead to") whose object is the system itself. The cue picks the label:
mitigation > outage > degradation. Negation ("did not affect", "no significant", "unaffected")
turns degradation/outage into no_effect. The pattern name records the path shape:
  dep_svo      event in the subject, system in the object of the same verb   "storm degraded GPS"
  dep_passive  system is the passive subject, event the agent                "GPS was degraded by the storm"
  dep_nominal  joined through a noun                                          "effects of storms on GPS"
  dep_prep     event attached by during/under/in                              "GPS errors increased during storms"
  dep_path     any other short path with a cue
--baseline is the PRD co-occurrence baseline: every event-system pair in a sentence is a relation
(labelled with the majority effect, degradation).

Writes data/triples/<run>_<version>.jsonl (PRD triple schema) and a .summary.json.
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from nltk.corpus import wordnet as wn

import parsed
from entities import Tagger, run_name

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "triples"
MAX_PATH = 7  # "superstorm led to loss of lock of GNSS receivers" is 7 edges
MAX_GOVERNORS = 3

MITIGATION = {"mitigate", "mitigation", "protect", "protection", "block", "blocking", "prevent", "prevention",
              "shield", "shielding", "harden", "hardening", "compensate", "compensation", "countermeasure",
              "suppress", "suppression", "safeguard"}
OUTAGE = {"fail", "failure", "outage", "blackout", "collapse", "shutdown", "shut", "trip", "breakdown",
          "disable", "destroy", "cease", "black", "loss"}  # "loss" counts only as loss of lock/signal/...
LOSS_OF = {"lock", "signal", "service", "satellite", "spacecraft", "contact", "communication", "power",
           "tracking", "mission"}
DEGRADATION = {"degrade", "degradation", "disrupt", "disruption", "affect", "impact", "influence", "perturb",
               "perturbation", "disturb", "damage", "impair", "impairment", "interrupt", "interruption",
               "interfere", "interference", "threaten", "threat", "endanger", "harm", "hamper", "corrupt",
               "distort", "distortion", "deteriorate", "deterioration", "error", "anomaly", "effect",
               "hazard", "risk", "problem", "vulnerability", "malfunction", "scintillation", "fading", "fade",
               "absorption", "delay", "upset", "heating", "saturation", "jam", "reduce", "reduction",
               "decrease", "worsen", "challenge", "jeopardize", "compromise", "drop", "outlier", "slip"}
CAUSAL = {"cause", "induce", "produce", "lead", "result", "trigger", "generate", "drive", "pose", "bring", "give"}
NO_EFFECT = {"unaffected", "immune", "negligible", "insignificant", "unchanged", "unaffecte"}
NEGATORS = {"not", "no", "never", "without", "neither", "nor", "n't", "hardly", "barely"}
CONFIDENCE = {"dep_svo": 0.9, "dep_passive": 0.9, "dep_nominal": 0.75, "dep_prep": 0.7, "dep_path": 0.6,
              "cooccurrence": 0.3}
log = logging.getLogger("effects")


def dep_path(a, b):
    """Tokens on the dependency path from a to b (inclusive), or None if not connected."""
    up = [a, *a.ancestors]
    down = [b, *b.ancestors]
    down_ids = [t.i for t in down]
    lca = next((t for t in up if t.i in down_ids), None)
    if lca is None:
        return None
    return up[:up.index(lca) + 1] + list(reversed(down[:down_ids.index(lca.i)]))


def governors(path):
    """Words above the path's top node, up to and including the nearest verb (max MAX_GOVERNORS)."""
    top = min(path, key=lambda t: len(list(t.ancestors)))
    out = []
    for tok in top.ancestors:
        if len(out) == MAX_GOVERNORS:
            break
        out.append(tok)
        if tok.pos_ in ("VERB", "AUX"):
            break
    return [] if top.pos_ in ("VERB", "AUX") else out


def _is_negated(tok):
    """A negator attached to this token ("not degrade", "no effect"), ignoring "not only"."""
    for child in tok.children:
        if child.dep_ == "neg" or child.lower_ in NEGATORS:
            nxt = child.nbor(1) if child.i + 1 < len(child.doc) else None
            if not (nxt is not None and nxt.lower_ == "only"):
                return True
    return tok.head.lower_ == "without" or (tok.head.dep_ == "prep" and tok.head.lower_ == "without")


def base_forms(tok):
    """spaCy lemma plus WordNet verb/noun bases: the tagger often reads participles as adjectives
    ("degraded" ADJ keeps lemma "degraded"), which would hide the cue "degrade"."""
    low = tok.lower_
    return {tok.lemma_.lower(), low, wn.morphy(low, wn.VERB) or low, wn.morphy(low, wn.NOUN) or low}


def _cue_label(tok):
    forms = base_forms(tok)
    if forms & NO_EFFECT:  # "unaffected": a cue in itself; classify() then marks the triple no_effect
        return "degradation"
    if forms & MITIGATION:
        return "mitigated"
    if "loss" in forms:
        of = [c for c in tok.children if c.lower_ == "of"]
        objs = [g for o in of for g in o.children if g.dep_ == "pobj"]
        return "outage" if any(g.lemma_.lower() in LOSS_OF for g in objs) else "degradation"
    if forms & OUTAGE:
        return "outage"
    if forms & DEGRADATION:
        return "degradation"
    return None


def _shape(event, system, path):
    lca = min(path, key=lambda t: len(list(t.ancestors)))
    e_rel = next((t.dep_ for t in [event, *event.ancestors] if t.head == lca), "")
    s_rel = next((t.dep_ for t in [system, *system.ancestors] if t.head == lca), "")
    if lca.pos_ in ("VERB", "AUX"):
        if e_rel in ("nsubj", "csubj") and s_rel in ("dobj", "obj", "attr", "prep", "dative", "conj", "xcomp", "ccomp"):
            return "dep_svo"
        if s_rel == "nsubjpass" and e_rel == "agent":
            return "dep_passive"
        if e_rel == "prep" and next((t for t in [event, *event.ancestors] if t.head == lca)).lower_ in (
                "during", "under", "in", "throughout", "amid"):
            return "dep_prep"
    if lca.pos_ in ("NOUN", "PROPN"):
        return "dep_nominal"
    return "dep_path"


def classify(event_span, system_span, use_negation=True):
    """(effect, method, negated) for a mention pair, or None if no effect is stated."""
    path = dep_path(event_span.root, system_span.root)
    if path is None or len(path) - 1 > MAX_PATH:
        return None
    top = min(path, key=lambda t: len(list(t.ancestors)))
    if all(t.dep_ in ("conj", "appos") for t in path if t is not top):
        return None  # items of one list ("a G5 storm, an S2 storm and an R3 blackout") are not a cause
    region = [t for t in path if t.i not in range(event_span.start, event_span.end)] + governors(path)
    cues = [(t, lab) for t in region if (lab := _cue_label(t))]
    if not cues:
        # neutral causal verb whose object is the system ("storms induce GICs", "caused a radio blackout")
        causal = [t for t in path if base_forms(t) & CAUSAL and t.pos_ in ("VERB", "ADJ")]
        if not causal or not any(system_span.root.head == v or system_span.root.head.head == v for v in causal):
            return None
        cues = [(causal[0], next((lab for t in system_span if (lab := _cue_label(t)) == "outage"), "degradation"))]
    labels = {lab for _, lab in cues}
    effect = "mitigated" if "mitigated" in labels else "outage" if "outage" in labels else "degradation"
    negated = use_negation and (
        any(_is_negated(t) for t, _ in cues) or any(_is_negated(t) for t in path if t.pos_ in ("VERB", "AUX"))
        or any(base_forms(t) & NO_EFFECT for t in region))
    if negated:
        effect = "degradation" if effect == "mitigated" else "no_effect"
    return effect, _shape(event_span.root, system_span.root, path), negated


def extract(doc, entities, baseline=False, use_negation=True):
    events = [e for e in entities if e["label"] == "EVENT"]
    systems = [e for e in entities if e["label"] == "SYSTEM"]
    out = []
    for ev in events:
        for sy in systems:
            if baseline:
                effect, method, negated = "degradation", "cooccurrence", False
            else:
                got = classify(ev["_span"], sy["_span"], use_negation)
                if got is None:
                    continue
                effect, method, negated = got
            out.append({"event": {"text": ev["text"], "canonical": ev["canonical"]},
                        "system": {"text": sy["text"], "canonical": sy["canonical"]},
                        "effect": effect, "method": method, "confidence": CONFIDENCE[method], "negated": negated})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--baseline", action="store_true", help="co-occurrence baseline")
    parser.add_argument("--no-negation", action="store_true")
    parser.add_argument("--no-wsd", action="store_true")
    parser.add_argument("--no-synonyms", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tagger = Tagger(parsed.nlp(), "full", not args.no_wsd, not args.no_synonyms, args.version)
    name = ("cooccurrence_" if args.baseline else "") + run_name("full", tagger.use_wsd, tagger.use_synonyms) + \
        ("_noneg" if args.no_negation else "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}_{args.version}.jsonl"
    effects, methods, pairs = Counter(), Counter(), Counter()
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for bibcode, sid, doc in parsed.load_sentences(args.version):
            for t in extract(doc, tagger(doc), args.baseline, not args.no_negation):
                f.write(json.dumps({"bibcode": bibcode, "sentence_id": sid, "sentence": doc.text, **t},
                                   ensure_ascii=False) + "\n")
                n += 1
                effects[t["effect"]] += 1
                methods[t["method"]] += 1
                pairs[f"{t['event']['canonical']}|{t['system']['canonical']}"] += 1
    summary = {"run": name, "triples": n, "by_effect": dict(effects.most_common()),
               "by_method": dict(methods.most_common()), "by_pair": dict(pairs.most_common())}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("%s", json.dumps(summary))


if __name__ == "__main__":
    main()
