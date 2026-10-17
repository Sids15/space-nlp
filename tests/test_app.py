"""Headless check of the Streamlit demo. Run: python tests/test_app.py (needs the triple store)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_query_matrix_drilldown():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Space Weather Evidence Mapper"
    assert int(at.metric[0].value) > 0  # default cell (storm x GNSS) has papers

    box = {w.label: w for w in at.selectbox}
    box["Event"].set_value("SEP / solar flare")
    box["System"].set_value("Power grid")
    at.run()
    assert not at.exception, at.exception
    assert at.metric[3].value == "gap" and at.info  # the gap cell explains itself

    at.text_input[0].input("solar flare HF radio blackout").run()
    assert not at.exception, at.exception
    assert any("blackout" in m.value.lower() for m in at.markdown)


if __name__ == "__main__":
    test_query_matrix_drilldown()
    print("ok test_query_matrix_drilldown")
