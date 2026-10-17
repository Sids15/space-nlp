"""Offline checks for preprocess regexes. Run: python tests/test_preprocess.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocess import extract_abbreviations, extract_indices


def test_indices():
    got = extract_indices("Storms with Dst < −100 nT and Kp = 8+ occurred; SYM-H reached -518 nT.")
    assert [(g["index"], g["op"], g["value"], g["suffix"]) for g in got] == [
        ("Dst", "<", -100.0, ""), ("Kp", "=", 8.0, "+"), ("SYM-H", "=", -518.0, "")]
    assert extract_indices("the Dst index of -422 nT")[0]["value"] == -422.0
    assert extract_indices("Kp ≥ 7")[0]["op"] == ">="
    assert extract_indices("Dst ∼ -127 nT")[0]["value"] == -127.0
    assert extract_indices("Dst index varied from -13 to -321")[0]["op"] == "from"
    assert extract_indices("No index here, only GPS 1575 MHz.") == []


def test_abbreviations():
    text = ("Geomagnetically induced currents (GICs) and single event upsets (SEU) "
            "affect the Global Navigation Satellite System (GNSS). We used the (2003) storm.")
    assert extract_abbreviations(text) == {
        "GIC": "Geomagnetically induced currents",
        "SEU": "single event upsets",
        "GNSS": "Global Navigation Satellite System",
    }
    assert extract_abbreviations("total electron content (TEC)") == {"TEC": "total electron content"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
