"""Checks for effect extraction. Run: python tests/test_week4.py (needs data/lexicon from synonyms.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import parsed
from effects import extract
from entities import Tagger

NLP = parsed.nlp()
TAGGER = Tagger(NLP)


def triples(text, **kw):
    doc = NLP(text)
    return {(t["event"]["canonical"], t["system"]["canonical"], t["effect"]) for t in extract(doc, TAGGER(doc), **kw)}


CASES = {
    "Geomagnetic storms, solar radiation storms and radio blackouts are classified by NOAA.": set(),  # a list, no cause
    "The geomagnetic storm severely degraded GPS positioning accuracy.": {("geomagnetic_storm", "gnss", "degradation")},
    "GPS positioning was degraded by the geomagnetic storm.": {("geomagnetic_storm", "gnss", "degradation")},
    "The storm did not affect the power grid.": {("geomagnetic_storm", "power_grid", "no_effect")},
    "No significant effect of the geomagnetic storm on HF radio communication was observed.":
        {("geomagnetic_storm", "hf_radio", "no_effect")},
    "The power grid was unaffected by the moderate geomagnetic storm.": {("geomagnetic_storm", "power_grid", "no_effect")},
    "The solar flare caused a complete radio blackout.": {("sep_event", "hf_radio", "outage")},
    "The superstorm led to loss of lock of GNSS receivers.": {("geomagnetic_storm", "gnss", "outage")},
    "Blocking devices mitigate the GIC effects on transformers during geomagnetic storms.":
        {("geomagnetic_storm", "power_grid", "mitigated")},
    "Storm-induced GICs damaged several transformers.": {("geomagnetic_storm", "power_grid", "degradation")},
    "The storm not only degraded GPS but also disrupted HF radio.":
        {("geomagnetic_storm", "gnss", "degradation"), ("geomagnetic_storm", "hf_radio", "degradation")},
    # co-occurrence without a stated effect must give nothing
    "We studied GPS data during the geomagnetic storm of 2015.": set(),
    "The storm caused an enhancement of TEC as measured on GNSS stations.": set(),
}


def test_cases():
    for text, expected in CASES.items():
        assert triples(text) == expected, (text, triples(text))


def test_negation_ablation_flips_back():
    assert triples("The storm did not affect the power grid.", use_negation=False) == \
        {("geomagnetic_storm", "power_grid", "degradation")}


def test_cooccurrence_baseline_links_every_pair():
    assert triples("We studied GPS data during the geomagnetic storm of 2015.", baseline=True) == \
        {("geomagnetic_storm", "gnss", "degradation")}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
