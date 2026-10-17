"""Checks for the evaluation metrics. Run: python tests/test_eval.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import counts, errors, partial_matches, prf, score_entities  # noqa: E402


def test_prf():
    assert prf(2, 2, 0) == {"p": 0.5, "r": 1.0, "f1": 0.667, "tp": 2, "fp": 2, "fn": 0}
    assert prf(0, 0, 3)["f1"] == 0.0


def test_partial_is_one_to_one_and_label_aware():
    gold = {(0, 10, "gnss")}
    assert partial_matches({(0, 3, "gnss"), (5, 10, "gnss")}, gold) == 1  # two halves cannot both claim one gold span
    assert partial_matches({(0, 10, "hf_radio")}, gold) == 0


def test_entities_exact_vs_partial():
    gold = {"s": {"entities": [(0, 17, "geomagnetic_storm"), (30, 33, "gnss")], "triples": {}}}
    pred = {"s": {(0, 17, "geomagnetic_storm"), (26, 33, "gnss")}}
    res = score_entities(gold, pred)
    assert res["all"]["exact"]["tp"] == 1 and res["all"]["partial"]["tp"] == 2
    assert res["event"]["exact"]["f1"] == 1.0


def test_triples_split_multi_label_gold():
    gold = {"s": {"entities": [], "triples": {("sep_event", "gnss"): "degradation|outage"}}}
    pred = {"s": {("sep_event", "gnss", "outage"), ("geomagnetic_storm", "gnss", "degradation")}}
    assert counts(gold, pred, True, ["s"]) == (1, 1, 1)
    assert counts(gold, pred, False, ["s"]) == (1, 1, 0)


def test_error_categories():
    gold = {"s": {"text": "The geomagnetic storm hit GPS.", "entities": [(4, 21, "geomagnetic_storm"), (26, 29, "gnss")],
                  "triples": {("geomagnetic_storm", "gnss"): "outage"}}}
    ents = {"s": {(16, 21, "geomagnetic_storm"), (26, 29, "hf_radio")}}
    rows = errors(gold, ents, {"s": {("geomagnetic_storm", "gnss", "degradation")}})
    got = {(r["kind"], r["auto_category"]) for r in rows}
    assert {("entity_fp", "boundary"), ("entity_fp", "wrong_label"), ("triple_fp", "wrong_effect"),
            ("triple_fn", "wrong_effect")} <= got


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
