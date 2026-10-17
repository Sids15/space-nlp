"""Checks for entity tagging, WSD and the gold-set loader/kappa. Run: python tests/test_week3.py
Needs the spaCy model, WordNet and data/lexicon from synonyms.py."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import gold
import parsed
import wsd
from entities import Tagger

NLP = parsed.nlp()
TAGGER = Tagger(NLP)


def tags(text, tagger=TAGGER):
    return [(e["text"], e["canonical"]) for e in tagger(NLP(text))]


def test_gazetteer_chunks_and_trimming():
    got = tags("Geomagnetically induced currents damaged several power transformers in the Hydro-Quebec grid.")
    assert ("Geomagnetically induced currents", "power_grid") in got
    assert ("power transformers", "power_grid") in got  # "several" trimmed
    assert ("Hydro-Quebec grid", "power_grid") in got   # chunk head rule


def test_event_and_system_never_merged():
    got = tags("The intense geomagnetic storm severely degraded GPS positioning accuracy.")
    assert ("intense geomagnetic storm", "geomagnetic_storm") in got and ("GPS", "gnss") in got



def test_guideline_boundaries():
    assert ("Global Positioning System (GPS)", "gnss") in tags("The storm degraded the Global Positioning System (GPS).")
    assert ("Geomagnetic Disturbances (GMDs)", "geomagnetic_storm") in tags("Geomagnetic Disturbances (GMDs) damage the power grid.")
    got = tags("The 2003 Halloween storm hit 40 satellites.")
    assert ("2003 Halloween storm", "geomagnetic_storm") in got and ("satellites", "satellites") in got  # year stays, count goes
    got = tags("The storm hit LEO satellites, GEO satellites and GPS.")
    assert not any("," in text for text, _ in got)  # never one highlight across a comma
    assert ("GPS stations", "gnss") in tags("The storm degraded data from as many as 114 GPS stations.")
    assert ("geomagnetic storms (G)", "geomagnetic_storm") in tags("NOAA scales cover geomagnetic storms (G) and more.")
    assert not any(t.count("(") != t.count(")") for t, _ in tags("Geomagnetic storms induced current (GIC) in transformers."))
    got = tags("Geomagnetically induced current (GIC) in power grids occurring during storms can cause damage.")
    assert ("Geomagnetically induced current (GIC)", "power_grid") in got


def test_wsd_rejects_off_domain_senses():
    assert tags("A tropical storm with heavy rain and hurricane winds hit the coast.") == []
    assert tags("The model was run on a 1 degree grid with fine horizontal spacing.") == []
    assert ("X9.3 flare", "sep_event") in tags("The X9.3 flare produced a radio blackout.")


def test_baseline_gazetteer_has_no_chunk_rules():
    base = Tagger(NLP, mode="gazetteer")
    assert tags("The Inmarsat satellites recorded upsets.", base) == []


def test_lesk_margin():
    assert wsd.disambiguate("storm", ["strong", "effect"], idf={"strong": 4.0}, margin=5.0)[1] is True
    assert wsd.disambiguate("storm", ["hurricane", "rain", "wind"], idf={"strong": 4.0}, margin=5.0)[1] is False


def test_gold_loader_and_kappa():
    text = "The storm degraded GPS."
    def export(effect):
        res = [{"id": "e", "type": "labels", "value": {"start": 4, "end": 9, "labels": ["STORM"]}},
               {"id": "s", "type": "labels", "value": {"start": 19, "end": 22, "labels": ["GNSS"]}}]
        if effect:  # drawn system -> event on purpose: the loader must normalise direction
            res.append({"type": "relation", "from_id": "s", "to_id": "e", "labels": [effect]})
        return [{"data": {"id": "x|0", "text": text}, "annotations": [{"result": res}]},
                {"data": {"id": "x|1", "text": "Nothing here."}, "annotations": [{"result": []}]}]
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.json", Path(d) / "b.json"
        a.write_text(json.dumps(export("degradation")))
        b.write_text(json.dumps(export(None)))
        loaded = gold.load_export(a)
        assert loaded["x|0"]["triples"] == {("geomagnetic_storm", "gnss"): "degradation"}
        old = gold.GOLD_DIR
        gold.GOLD_DIR = Path(d)
        try:
            r = gold.kappa(a, b, "test")
        finally:
            gold.GOLD_DIR = old
        assert r["kappa_event_token"] == 1.0 and r["kappa_system_token"] == 1.0
        assert r["effect_units"] == 1 and r["gate_passed"] is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
