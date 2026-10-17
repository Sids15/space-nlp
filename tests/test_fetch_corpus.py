"""Offline checks for fetch_corpus. Run: python tests/test_fetch_corpus.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fetch_corpus import build_queries, clean_text, is_english, merge_and_filter

LONG_EN = "The geomagnetic storm of 2003 caused severe degradation of GPS positioning " * 8
LONG_DE = "Der geomagnetische Sturm verursachte eine starke Verschlechterung der Ortung " * 8


def doc(bibcode, abstract, doi=None):
    return {"bibcode": bibcode, "title": ["T"], "abstract": abstract, "year": "2003", "doi": doi}


def test_queries_cover_all_strata():
    q = build_queries()
    assert len(q) == 8
    assert "year:[2000 TO 2026]" in q["sep_event|power_grid"]


def test_clean_text_strips_markup():
    assert clean_text("H<SUB>2</SUB>O &amp; <P /> x") == "H 2 O & x"


def test_language_check():
    assert is_english(LONG_EN)
    assert not is_english(LONG_DE)


def test_dedup_filter_and_strata():
    results = [
        ("geomagnetic_storm|gnss", doc("A", LONG_EN, ["10.1/X"])),
        ("sep_event|gnss", doc("A", LONG_EN, ["10.1/X"])),       # same bibcode
        ("sep_event|hf_radio", doc("B", LONG_EN, ["10.1/x"])),   # same DOI, different case
        ("sep_event|hf_radio", doc("C", "too short")),
        ("sep_event|hf_radio", doc("D", LONG_DE)),
    ]
    kept, stats = merge_and_filter(results)
    assert [r["bibcode"] for r in kept] == ["A"]
    assert kept[0]["strata"] == ["geomagnetic_storm|gnss", "sep_event|gnss", "sep_event|hf_radio"]
    assert stats == {"raw": 5, "duplicates": 2, "unique": 3, "dropped_short": 1,
                     "dropped_non_english": 1, "kept": 1}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
