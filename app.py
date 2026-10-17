"""Streamlit demo: query -> evidence matrix cell -> supporting sentences.

Run:  .venv/Scripts/streamlit run app.py
"""

import re
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from domain import EFFECTS, EVENTS, SYSTEMS  # noqa: E402
from matrix import build, load_triples  # noqa: E402
import retrieval  # noqa: E402

VERSION = "v1"
TRIPLE_DIR = Path(__file__).resolve().parent / "data" / "triples"
NAMES = {"geomagnetic_storm": "Geomagnetic storm", "sep_event": "SEP / solar flare", "gnss": "GNSS",
         "hf_radio": "HF radio", "satellites": "Satellites", "power_grid": "Power grid"}
STATUS_COLOUR = {"gap": "#f2f2f2", "single": "#fff4d6", "agree": "#d9f2e3", "mixed": "#fde2d6"}

st.set_page_config(page_title="Space Weather Evidence Mapper", layout="wide")


def md_escape(text):
    return re.sub(r"([\\`*_{}\[\]()#+\-.!$|<>~])", r"\\\1", text)


def highlight(sentence, terms):
    """Bold the first occurrence of each term; everything else is escaped (no HTML, no LaTeX)."""
    marks = []
    for term in terms:
        i = sentence.find(term)
        if i >= 0 and not any(s < i + len(term) and i < e for s, e in marks):
            marks.append((i, i + len(term)))
    out, pos = [], 0
    for s, e in sorted(marks):
        out += [md_escape(sentence[pos:s]), "**" + md_escape(sentence[s:e]) + "**"]
        pos = e
    return "".join(out + [md_escape(sentence[pos:])])


def ads_link(bibcode, sid):
    url = "https://ui.adsabs.harvard.edu/abs/" + urllib.parse.quote(bibcode, safe="") + "/abstract"
    return f"[{md_escape(bibcode)}]({url}) · sentence {sid}"


@st.cache_data
def cells_for(run):
    return build(load_triples(run, VERSION))


@st.cache_resource
def index():
    return retrieval.Index(VERSION, "tfidf")


runs = sorted(p.stem.removesuffix(f"_{VERSION}") for p in TRIPLE_DIR.glob(f"*_{VERSION}.jsonl"))
if not runs:
    st.error("No triple store found. Run src/effects.py first.")
    st.stop()

st.title("Space Weather Evidence Mapper")
st.caption("Which space-weather event affects which technological system, and how. "
           "Every cell traces back to the sentences that support it.")

with st.sidebar:
    run = st.selectbox("Extraction run", runs, index=runs.index("full") if "full" in runs else 0,
                       help="full = the Phase 1 pipeline; cooccurrence_full = baseline; others are ablations")
    st.markdown("**Status colours**")
    for status, colour in STATUS_COLOUR.items():
        st.markdown(f"<span style='background:{colour};padding:2px 8px;border-radius:4px'>{status}</span>",
                    unsafe_allow_html=True)  # fixed strings only, no user or corpus text

cells = cells_for(run)
by_key = {(c["event"], c["system"]): c for c in cells}

st.subheader("Evidence matrix")
table = pd.DataFrame(
    [[f"{by_key[(e, s)]['papers']} papers · {by_key[(e, s)]['majority'] or '—'}" for s in SYSTEMS] for e in EVENTS],
    index=[NAMES[e] for e in EVENTS], columns=[NAMES[s] for s in SYSTEMS])
colours = pd.DataFrame([[f"background-color: {STATUS_COLOUR[by_key[(e, s)]['status']]}" for s in SYSTEMS] for e in EVENTS],
                       index=table.index, columns=table.columns)
st.dataframe(table.style.apply(lambda _: colours, axis=None), width="stretch")

st.subheader("Drill down")
left, right = st.columns(2)
CANONICAL = {v: k for k, v in NAMES.items()}
event = CANONICAL[left.selectbox("Event", [NAMES[e] for e in EVENTS])]
system = CANONICAL[right.selectbox("System", [NAMES[s] for s in SYSTEMS])]
cell = by_key[(event, system)]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Papers", cell["papers"])
m2.metric("Majority effect", cell["majority"] or "—")
m3.metric("Agreement", f"{cell['agreement']:.0%}" if cell["agreement"] is not None else "—")
m4.metric("Status", cell["status"])
if cell["papers"]:
    st.bar_chart(pd.Series(cell["by_effect"], name="papers"))

effect_filter = st.multiselect("Show effects", EFFECTS, default=[e for e in EFFECTS if cell["by_effect"][e]])
evidence = [e for e in cell["evidence"] if e["effect"] in effect_filter]
st.markdown(f"**{len(evidence)} supporting sentences** (extracted triples, highest confidence first)")
for e in evidence[:50]:
    st.markdown(f"- `{e['effect']}` · {highlight(e['sentence'], [e['event_text'], e['system_text']])}  \n"
                f"  {ads_link(e['bibcode'], e['sentence_id'])} · {e['method']} ({e['confidence']})",
                unsafe_allow_html=False)
if cell["papers"] == 0:
    st.info("No extracted evidence for this pair. That is a finding too: either no paper in the corpus "
            "states this effect within one sentence, or the pipeline misses it.")

with st.expander("More related sentences (TF-IDF retrieval, not necessarily an extracted effect)"):
    for h in index().search(retrieval.pair_query(event, system), 10):
        st.markdown(f"- {md_escape(h['text'])}  \n  {ads_link(h['bibcode'], h['sid'])} · score {h['score']}")

st.subheader("Free-text search")
query = st.text_input("Search the corpus sentences", placeholder="e.g. solar flare HF radio blackout")
if query.strip():
    hits = index().search(retrieval.lemmas(query[:500]), 10)
    if not hits:
        st.write("No matching sentences.")
    for h in hits:
        st.markdown(f"- {md_escape(h['text'])}  \n  {ads_link(h['bibcode'], h['sid'])} · score {h['score']}")
