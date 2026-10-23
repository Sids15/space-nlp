"""Streamlit demo: evidence matrix -> one cell -> the sentences that support it.

Run:  .venv/Scripts/streamlit run app.py
"""

import html
import sys
import urllib.parse
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from domain import EFFECTS, EVENTS, SYSTEMS  # noqa: E402
from matrix import build, load_triples  # noqa: E402
import retrieval  # noqa: E402

VERSION = "v1"
TRIPLE_DIR = Path(__file__).resolve().parent / "data" / "triples"
NAMES = {"geomagnetic_storm": "Geomagnetic storm", "sep_event": "SEP / solar flare", "gnss": "GNSS",
         "hf_radio": "HF radio", "satellites": "Satellites", "power_grid": "Power grid"}
RUN_NAMES = {"full": "Full pipeline", "cooccurrence_full": "Co-occurrence baseline",
             "full_noneg": "Full, without negation", "full_nosyn": "Full, without synonyms",
             "full_nowsd": "Full, without WSD"}
STATUS_NOTE = {"agree": "papers agree", "mixed": "papers disagree", "single": "one paper", "gap": "no evidence"}
EFFECT_COLOUR = {"degradation": "#b5562f", "outage": "#7a2e1d", "no_effect": "#6f7f86", "mitigated": "#3f7a5a"}

# Fixed stylesheet. Every piece of corpus text that reaches unsafe_allow_html goes through html.escape first.
CSS = """
<style>
.block-container {max-width: 1100px; padding-top: 2.5rem;}
.overline {font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: #6b6f76; margin: 0;}
.lede {color: #4a4f57; max-width: 46rem; margin: .2rem 0 1.2rem;}
table.mx {border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums;}
table.mx th {font-weight: 600; font-size: .85rem; color: #4a4f57; text-align: left; padding: .4rem .6rem;
             border-bottom: 1px solid #dcd8cf;}
table.mx td {padding: .7rem .6rem; border-bottom: 1px solid #e8e5de; vertical-align: top; width: 21%;}
table.mx td.row {font-weight: 600; width: 16%;}
table.mx td.sel {outline: 2px solid #0f5c63; outline-offset: -2px;}
.n {font-size: 1.5rem; line-height: 1; font-family: Georgia, serif;}
.n small {font-size: .8rem; color: #6b6f76; font-family: sans-serif;}
.sub {font-size: .8rem; color: #4a4f57; margin-top: .25rem;}
.agree {background: #e6efe8;} .mixed {background: #f6e6d8;} .single {background: #f3eed9;}
.gap {background: repeating-linear-gradient(135deg, #fbfaf7 0 6px, #f1efe9 6px 12px); color: #8a8e94;}
.legend {font-size: .78rem; color: #6b6f76; margin-top: .4rem;}
.legend span {display: inline-block; width: .8rem; height: .8rem; margin: 0 .3rem -.1rem .9rem;
              border: 1px solid #dcd8cf;}
.bar {display: flex; height: .55rem; border-radius: 2px; overflow: hidden; margin: .3rem 0 .2rem;}
.keys {font-size: .78rem; color: #4a4f57;}
.keys i {display: inline-block; width: .6rem; height: .6rem; margin: 0 .25rem 0 .7rem; border-radius: 1px;}
.ev {border-left: 3px solid #dcd8cf; padding: .15rem 0 .15rem .9rem; margin: 0 0 1.1rem;}
.ev p {margin: 0 0 .25rem; line-height: 1.5;}
.ev mark {background: #f3e3b5; padding: 0 .1rem;}
.ev .meta {font-size: .78rem; color: #6b6f76;}
.ev .meta a {color: #0f5c63;}
.tag {font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; font-weight: 600;}
</style>
"""


def highlight(sentence, terms):
    """Escape the sentence and wrap the first occurrence of each term in <mark>."""
    marks = []
    for term in terms:
        i = sentence.find(term)
        if term and i >= 0 and not any(s < i + len(term) and i < e for s, e in marks):
            marks.append((i, i + len(term)))
    out, pos = [], 0
    for s, e in sorted(marks):
        out += [html.escape(sentence[pos:s]), "<mark>" + html.escape(sentence[s:e]) + "</mark>"]
        pos = e
    return "".join(out + [html.escape(sentence[pos:])])


def source(bibcode, sid, extra=""):
    url = "https://ui.adsabs.harvard.edu/abs/" + urllib.parse.quote(bibcode, safe="") + "/abstract"
    return (f'<div class="meta"><a href="{html.escape(url)}" target="_blank" rel="noopener">'
            f'{html.escape(bibcode)}</a> · sentence {int(sid)}{extra}</div>')


def sentence_block(body, meta, colour="#dcd8cf", tag=""):
    tag_html = f'<span class="tag" style="color:{colour}">{tag}</span> ' if tag else ""
    return f'<div class="ev" style="border-left-color:{colour}"><p>{tag_html}{body}</p>{meta}</div>'


def matrix_html(by_key, selected):
    head = "".join(f"<th>{NAMES[s]}</th>" for s in SYSTEMS)
    rows = []
    for e in EVENTS:
        tds = []
        for s in SYSTEMS:
            c = by_key[(e, s)]
            sel = " sel" if (e, s) == selected else ""
            if c["papers"]:
                sub = f"{c['majority'].replace('_', ' ')} · {c['agreement']:.0%} agree"
                tds.append(f'<td class="{c["status"]}{sel}"><div class="n">{c["papers"]} <small>papers</small>'
                           f'</div><div class="sub">{sub}</div></td>')
            else:
                tds.append(f'<td class="gap{sel}"><div class="n">0</div><div class="sub">no evidence</div></td>')
        rows.append(f'<tr><td class="row">{NAMES[e]}</td>{"".join(tds)}</tr>')
    legend = "".join(f'<span class="{k}"></span>{v}' for k, v in STATUS_NOTE.items())
    return (f'<table class="mx"><tr><th></th>{head}</tr>{"".join(rows)}</table>'
            f'<div class="legend">{legend}</div>')


def effect_bar(by_effect):
    total = sum(by_effect.values())
    parts = "".join(f'<div style="width:{100 * n / total:.1f}%;background:{EFFECT_COLOUR[k]}"></div>'
                    for k, n in by_effect.items() if n)
    keys = "".join(f'<i style="background:{EFFECT_COLOUR[k]}"></i>{k.replace("_", " ")} {n}'
                   for k, n in by_effect.items() if n)
    return f'<div class="bar">{parts}</div><div class="keys">{keys}</div>'


@st.cache_data
def cells_for(run):
    return build(load_triples(run, VERSION))


@st.cache_resource
def index():
    return retrieval.Index(VERSION, "tfidf")


st.set_page_config(page_title="Space Weather Evidence Mapper", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)

runs = sorted(p.stem.removesuffix(f"_{VERSION}") for p in TRIPLE_DIR.glob(f"*_{VERSION}.jsonl"))
if not runs:
    st.error("No triple store found. Run src/effects.py first.")
    st.stop()

with st.sidebar:
    run = st.radio("Extraction run", runs, index=runs.index("full") if "full" in runs else 0,
                   format_func=lambda r: RUN_NAMES.get(r, r),
                   help="Switch to the baseline or an ablation to see how the matrix changes.")

st.markdown('<p class="overline">NLP 702AI0C009 · Phase 1</p>', unsafe_allow_html=True)
st.title("Space Weather Evidence Mapper")
st.markdown('<p class="lede">What the literature says space weather does to technology. Each cell counts the '
            'papers in 1,002 NASA ADS abstracts where an event is stated to affect a system, and every count '
            'traces back to the sentences behind it.</p>', unsafe_allow_html=True)

cells = cells_for(run)
by_key = {(c["event"], c["system"]): c for c in cells}

tab_matrix, tab_search = st.tabs(["Evidence matrix", "Search sentences"])

with tab_matrix:
    left, right = st.columns([1, 2])
    event = left.segmented_control("Event", EVENTS, default=EVENTS[0], format_func=NAMES.get) or EVENTS[0]
    system = right.segmented_control("System", SYSTEMS, default=SYSTEMS[0], format_func=NAMES.get) or SYSTEMS[0]
    st.markdown(matrix_html(by_key, (event, system)), unsafe_allow_html=True)
    if run != "full":
        st.caption(f"Showing: {RUN_NAMES.get(run, run)}")

    cell = by_key[(event, system)]
    st.subheader(f"{NAMES[event]} → {NAMES[system]}")
    if not cell["papers"]:
        st.markdown("No paper in the corpus states this effect within one sentence, or the pipeline misses it. "
                    "A gap is a finding too. Related sentences found by search are listed below.")
    else:
        st.markdown(f"{cell['papers']} papers, mostly **{cell['majority'].replace('_', ' ')}** "
                    f"({cell['agreement']:.0%} of them agree).")
        st.markdown(effect_bar(cell["by_effect"]), unsafe_allow_html=True)

        shown = st.pills("Effects", EFFECTS, selection_mode="multi", format_func=lambda k: k.replace("_", " "),
                         default=[k for k in EFFECTS if cell["by_effect"][k]], label_visibility="collapsed")
        evidence = [e for e in cell["evidence"] if e["effect"] in shown]
        st.caption(f"{len(evidence)} extracted sentences, highest confidence first")
        with st.container(height=520, border=False):
            st.markdown("".join(
                sentence_block(highlight(e["sentence"], [e["event_text"], e["system_text"]]),
                               source(e["bibcode"], e["sentence_id"], f" · {html.escape(e['method'])} "
                                                                      f"{float(e['confidence']):.2f}"),
                               EFFECT_COLOUR[e["effect"]], e["effect"].replace("_", " "))
                for e in evidence[:50]), unsafe_allow_html=True)

    with st.expander("Related sentences from search (not necessarily an extracted effect)"):
        st.markdown("".join(sentence_block(html.escape(h["text"]), source(h["bibcode"], h["sid"],
                                                                         f" · score {float(h['score']):.3f}"))
                            for h in index().search(retrieval.pair_query(event, system), 10)),
                    unsafe_allow_html=True)

with tab_search:
    query = st.text_input("Search the corpus sentences", placeholder="solar flare HF radio blackout",
                          max_chars=500)
    if query.strip():
        hits = index().search(retrieval.lemmas(query), 10)
        if not hits:
            st.markdown("No matching sentences.")
        st.markdown("".join(sentence_block(html.escape(h["text"]), source(h["bibcode"], h["sid"],
                                                                         f" · score {float(h['score']):.3f}"))
                            for h in hits), unsafe_allow_html=True)
