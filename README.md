# Space Weather Evidence Mapper

A rule-based NLP pipeline that extracts structured evidence of space-weather effects on critical infrastructure from astrophysics abstracts. Given a corpus of peer-reviewed papers, the system identifies **event** mentions (geomagnetic storms, SEP events) and **system** mentions (GNSS, HF radio, satellites, power grids), classifies the dependency-path relationship between them into one of four effect labels (`degradation`, `outage`, `no_effect`, `mitigated`), and summarises everything into an interactive evidence matrix.

---

## Table of Contents

- [Background](#background)
- [Pipeline Architecture](#pipeline-architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Running the Demo App](#running-the-demo-app)
- [Evaluation Results](#evaluation-results)
- [Annotation and Gold Set](#annotation-and-gold-set)
- [License](#license)

---

## Background

Space-weather events — geomagnetic storms, solar energetic particle (SEP) events, solar flares — routinely degrade or interrupt civilian infrastructure. Evidence of those effects is scattered across thousands of astrophysics abstracts indexed by NASA ADS. This project builds a fully reproducible information-extraction pipeline to surface that evidence, map it to canonical (event, system) pairs, and expose it through a Streamlit interface that lets a researcher drill down from a summary cell to the exact sentences and papers that support it.

The domain is restricted to the six entity types defined in the PRD:

| Type  | Canonical labels |
|-------|-----------------|
| EVENT | `geomagnetic_storm`, `sep_event` |
| SYSTEM | `gnss`, `hf_radio`, `satellites`, `power_grid` |

Effect labels follow the same definition throughout every stage: `degradation`, `outage`, `no_effect`, `mitigated`.

---

## Pipeline Architecture

The pipeline runs in ten sequential stages. Each stage reads from `data/` and writes back to `data/`, so every intermediate artefact is inspectable and individually rerunnable.

```
Stage 1  fetch_corpus.py     NASA ADS API  ->  data/corpus/*.jsonl
Stage 2  preprocess.py       spaCy tokenise / lemmatise / abbreviation extraction
                             -> data/processed/*.jsonl
Stage 3  relevance.py        TF-IDF nearest-seed filter (threshold 0.137)
                             -> data/relevance/filtered_v1.jsonl
Stage 4  language_model.py   Bigram/trigram MLE, Laplace, Kneser-Ney
                             -> data/lm/perplexity_v1.json
Stage 5  synonyms.py         Corpus Word2Vec (100d, sg, phrases) + WordNet + abbreviation expansion
                             -> data/lexicon/w2v_v1.txt, lexicon_v1.json
Stage 6  entities.py         Gazetteer + noun-chunk head rules + WSD + W2V synonym grouping
                             -> data/entities/<run>_v1.jsonl
Stage 7  effects.py          Dependency-path effect classifier (dep_svo / dep_passive /
                             dep_nominal / dep_prep / dep_path)
                             -> data/triples/<run>_v1.jsonl
Stage 8  retrieval.py        TF-IDF sentence index, P@5 / pooled R@10
                             -> data/retrieval/eval_v1.json
Stage 9  matrix.py           (event, system) evidence matrix with agreement status
                             -> data/matrix/matrix_<run>_v1.json
Stage 10 evaluate.py         Exact and partial span P/R/F1 + bootstrap triple scoring
                             -> data/eval/results_<gold>_v1.json
```

### Stage 2 — Preprocessing

spaCy (`en_core_web_sm`) tokenises, lemmatises, and extracts geomagnetic index values (`Kp`, `Dst`, `SYM-H`) by regex. Abbreviation definitions (`IRNSS — Indian Regional Navigation Satellite System`) are extracted sentence-locally and stored per abstract for the synonym stage.

### Stage 3 — Relevance Filter

Every abstract is scored by its TF-IDF cosine to the three nearest of thirty seed abstracts (top-3-mean to avoid centroid bias toward GNSS-heavy seeds). Abstracts scoring below 0.137 are dropped, reducing 3,116 corpus abstracts to 1,002. A BoW-cosine and Jaccard score are recorded alongside TF-IDF for comparison.

### Stage 4 — Language Model

Bigram and trigram models with three smoothing regimes (MLE, Laplace, interpolated Kneser-Ney) are trained on the space-weather corpus and evaluated on a held-out 10% of both it and a 1,875-abstract general astrophysics reference corpus. Cross-domain perplexity ratios confirm that the domain language is meaningfully distinct (in-domain trigram KN perplexity: **103.9**; cross-domain: 279.9).

### Stage 5 — Synonym Grouping

Three complementary sources extend the hand-written gazetteer in `src/domain.py`:

1. **Corpus abbreviation learning** — Definitions extracted in stage 2 are matched back to the gazetteer head-word. Acronyms that appear at least twice with a consistent long form join the corresponding canonical label.
2. **WordNet closure** — The head noun of an unresolved term is looked up in WordNet; if any synset in its hypernym chain intersects a hand-written anchor synset, the term takes that canonical label.
3. **Corpus Word2Vec** — A Skip-gram model (100d, window 5, Gensim phrases for collocations such as `geomagnetic_storm`) assigns an unresolved term to the nearest prototype centroid, provided the cosine exceeds 0.55 and leads the runner-up by at least 0.05.

### Stage 6 — Entity Tagging

The `Tagger` class in `src/entities.py` runs three steps:

1. **Gazetteer + acronym scan** — spaCy `PhraseMatcher` with case-insensitive phrase matching and ORTH-sensitive acronym matching.
2. **Noun-chunk head rules** — Any noun-chunk whose root lemma is in the `HEADS` table (e.g. `storm`, `flare`, `grid`) triggers a candidate. Causal hyphen compounds (`storm-induced GICs`) are caught separately.
3. **Resolution** — Gazetteer hits take priority. Head-triggered candidates go through Lesk WSD (IDF-weighted, corpus-adapted via Word2Vec neighbours of the canonical label) and Word2Vec synonym grouping. Overlapping mentions of the same family are unioned; acronym brackets (`GPS (Global Positioning System)`) are joined into a single span.

Three ablation runs are supported via `--mode` and `--no-wsd` / `--no-synonyms` flags.

### Stage 7 — Effect Classification

`src/effects.py` finds the spaCy dependency path between every (EVENT, SYSTEM) mention pair in a sentence (maximum 7 edges). The path and its governing verb/noun cues are matched against five word sets (`MITIGATION`, `OUTAGE`, `DEGRADATION`, `CAUSAL`, `NO_EFFECT`). Effect priority: `mitigated` > `outage` > `degradation`. Negation (`not`, `no`, `n't`, `without`, plus `NO_EFFECT` lexical cues) flips `degradation`/`outage` to `no_effect` and `mitigated` to `degradation`. Five dependency patterns are distinguished:

| Pattern | Description | Example |
|---------|-------------|---------|
| `dep_svo` | Event subject, system object | *"storms degraded GPS accuracy"* |
| `dep_passive` | System passive subject, event agent | *"GPS was disrupted by the storm"* |
| `dep_nominal` | Joined through a noun | *"effects of storms on HF radio"* |
| `dep_prep` | Event attached by *during/under/in* | *"GPS errors increased during storms"* |
| `dep_path` | Any other short path with a cue | — |

The co-occurrence baseline (`--baseline`) labels every sentence-local pair `degradation` with a fixed confidence of 0.30.

### Stage 8 — Sentence Retrieval

A TF-IDF index (`sklearn TfidfVectorizer`) over 1,002 filtered abstracts supports (event, system) queries and free-text queries. Queries for canonical pairs are expanded with all gazetteer phrases for that pair. A pooled evaluation against 391 human-judged sentences measures P@5 and R@10 for TF-IDF vs. a BoW baseline.

### Stage 9 — Evidence Matrix

`src/matrix.py` collapses the triple store into an 2×4 matrix (events × systems). Each cell reports: distinct paper count per effect label, an agreement score (share of paper-votes going to the majority label), a status tag (`gap | single | agree | mixed`), and the highest-confidence supporting sentences with their bibcodes.

### Stage 10 — Evaluation

`src/evaluate.py` scores the entity tagger and effect classifier against the human gold set:

- **Entity**: exact span match `(start, end, canonical)` and partial match (overlap + same label), per slot (EVENT, SYSTEM, all).
- **Triples**: labelled `(event, system, effect)` and unlabelled `(event, system)` F1 with 95% bootstrap confidence intervals (1,000 resamples, seed 42) and a paired bootstrap test against the co-occurrence baseline.

---

## Repository Structure

```
space-nlp/
├── src/
│   ├── domain.py           canonical labels, gazetteer phrases, chunk-head rules
│   ├── fetch_corpus.py     Stage 1: NASA ADS API client
│   ├── preprocess.py       Stage 2: spaCy tokenisation / lemmatisation
│   ├── relevance.py        Stage 3: TF-IDF nearest-seed filter
│   ├── language_model.py   Stage 4: n-gram LM with three smoothing schemes
│   ├── synonyms.py         Stage 5: Word2Vec + WordNet + abbreviation grouper
│   ├── wsd.py              Stage 6a: IDF-weighted extended Lesk WSD
│   ├── entities.py         Stage 6b: entity tagger (gazetteer / heads / WSD / W2V)
│   ├── effects.py          Stage 7: dependency-path effect classifier
│   ├── retrieval.py        Stage 8: TF-IDF sentence retrieval + evaluation
│   ├── matrix.py           Stage 9: evidence matrix builder
│   ├── gold.py             gold-set sampling, Label Studio import/export, kappa, merge
│   ├── parsed.py           shared spaCy model loader and sentence iterator
│   └── evaluate.py         Stage 10: entity and triple scoring
├── tests/                  pytest suite
├── data/
│   ├── corpus/             raw and meta JSONL from NASA ADS
│   ├── processed/          spaCy-processed JSONL
│   ├── relevance/          TF-IDF scores and filtered abstracts
│   ├── lm/                 perplexity results
│   ├── lexicon/            Word2Vec model and learned lexicon
│   ├── gold/               annotation sample, Label Studio task files, kappa results
│   ├── triples/            extracted triples and summaries per run
│   ├── matrix/             evidence matrices per run
│   ├── retrieval/          retrieval evaluation
│   └── eval/               entity and triple scoring results
├── docs/
│   ├── annotation_guideline.md
│   ├── report.md           Phase 1 report
│   ├── report_results.md   all reported numbers with source file references
│   └── report.pdf          PDF build of the report
├── app.py                  Streamlit evidence-matrix demo
├── requirements.txt        pinned dependencies (Python 3.13)
└── .streamlit/config.toml  Streamlit theme settings
```

---

## Installation

> **Requires Python 3.13.** The `.venv` in the repo was built against 3.13; some pinned packages (e.g. `numpy 2.5.4`, `pandas 3.0.6`) require it.

```bash
git clone https://github.com/Sids15/space-nlp.git
cd space-nlp

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

# Download WordNet data (required by wsd.py and gold.py)
python -c "import nltk; nltk.download('wordnet')"
```

The spaCy model `en_core_web_sm` is included in `requirements.txt` as a direct URL dependency and installs automatically.

---

## Running the Pipeline

All scripts are in `src/`. Run them from the repository root with the virtual environment active.

```bash
# Stage 1: fetch corpus from NASA ADS (requires ADS_API_KEY in .env)
python src/fetch_corpus.py --version v1

# Stage 2: preprocess both corpora
python src/preprocess.py data/corpus/corpus_v1.jsonl data/corpus/reference_v1.jsonl

# Stage 3: relevance filter
python src/relevance.py --version v1

# Stage 4: language model perplexity
python src/language_model.py --version v1

# Stage 5: Word2Vec synonym grouper and learned acronyms
python src/synonyms.py --version v1

# Stage 6: entity tagging (full pipeline)
python src/entities.py --version v1 --mode full

# Stage 7: effect extraction (full pipeline)
python src/effects.py --version v1

# Stage 7: co-occurrence baseline
python src/effects.py --version v1 --baseline

# Stage 8: retrieval evaluation
python src/retrieval.py evaluate

# Stage 9: evidence matrix
python src/matrix.py --version v1 --run full
python src/matrix.py --version v1 --run cooccurrence_full

# Stage 10: evaluate against the gold set
python src/evaluate.py --gold data/gold/exports/test_gold.json --version v1
```

All stages are idempotent: re-running overwrites the previous output in the same versioned file.

---

## Running the Demo App

```bash
.venv\Scripts\streamlit run app.py     # Windows
# streamlit run app.py                 # macOS / Linux
```

The app loads `data/triples/full_v1.jsonl` and renders the 2×4 evidence matrix. Clicking a cell shows the supporting sentences with event and system spans highlighted, effect labels colour-coded, and ADS bibcode links. A run selector lets you switch between the full pipeline and the co-occurrence baseline in-session.

---

## Evaluation Results

Phase 1 numbers, all reproduced from files in this repository. Full tables are in `docs/report_results.md`; the methodology is in `docs/report.md`.

### Corpus and Filtering

| Item | Value |
|------|-------|
| Raw abstracts (NASA ADS, 2000–2026, 8 strata) | 3,116 |
| After TF-IDF relevance filter (threshold 0.137) | 1,002 |
| Astrophysics reference corpus (for LM / IDF) | 1,875 |

### Language Model (in-domain trigram Kneser-Ney)

| Train → Test | Perplexity |
|---|---|
| Space-weather → Space-weather | **103.9** |
| Space-weather → Astrophysics | 279.9 |

### Inter-Annotator Agreement (test overlap, 60 sentences)

| Metric | Value |
|---|---|
| Cohen's κ — EVENT (token) | 0.744 |
| Cohen's κ — SYSTEM (token) | 0.640 |
| Cohen's κ — effect (pair) | 0.013 |
| Span F1 — event (exact) | 0.562 |
| Span F1 — system (exact) | 0.439 |

Effect agreement did not meet the κ ≥ 0.6 gate; this is disclosed in the report.

### Entity Extraction (test gold, 200 sentences, 339 mentions)

| Run | Exact P | Exact R | Exact F1 | Partial F1 |
|---|---|---|---|---|
| **Full pipeline** | 0.534 | 0.563 | 0.548 | 0.769 |
| Gazetteer only (baseline) | 0.425 | 0.428 | 0.426 | 0.691 |
| spaCy NER (baseline) | 0.048 | 0.003 | 0.006 | 0.106 |

Entity precision ≥ 0.75 (partial) — **Phase 1 gate passed**.

### Triple Extraction (test gold, 44 labelled triples)

| Run | P | R | F1 | F1 95% CI |
|---|---|---|---|---|
| **Full pipeline** | 0.439 | 0.409 | **0.424** | 0.276–0.554 |
| Co-occurrence (baseline) | 0.242 | 0.545 | 0.336 | 0.222–0.440 |

Full pipeline beats the co-occurrence baseline in 90.3% of bootstrap resamples — **Phase 1 gate passed**.

---

## Annotation and Gold Set

The gold set consists of 300 sentences sampled from the filtered corpus across 8 event × system strata (150 abstracts, seeded random, stratified round-robin). The sample is split into:

- **Pilot / dev** — 30 sentences, annotated by both annotators for guideline calibration (κ ≥ 0.6 on all slots).
- **Dev rest** — 70 sentences, used for rule tuning.
- **Test** — 200 sentences, used once at the end.

Annotation was done in [Label Studio](https://labelstud.io/) using the configuration in `docs/label_studio_config.xml`. The guideline is `docs/annotation_guideline.md`. Annotator exports are kept out of the repository (`.gitignore: data/gold/exports/`) to preserve annotator independence until double annotation is complete; the merged gold file is referenced only from `data/eval/results_*.json`.

Inter-annotator kappa and span-F1 are computed with `python src/gold.py kappa A.json B.json`. Disagreements for the discussion round are listed with `python src/gold.py diff A.json B.json`. Final gold is produced with `python src/gold.py merge A.json B.json --out path/to/gold.json` (seeded random choice of annotator per disagreeing sentence, seed 42).

---

## License

MIT License

Copyright (c) 2026 Siddhant Bandi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
