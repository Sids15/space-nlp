# 🛰️ Space Weather Evidence Mapper — Natural Language Processing & Evidence Synthesis Pipeline

### Causal Information Extraction & Evidence Matrix Engine · NASA ADS Heliophysics Corpus

[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3130/)
[![spaCy 3.8](https://img.shields.io/badge/spaCy-3.8-09A3D5?style=flat&logo=spacy&logoColor=white)](https://spacy.io/)
[![Word2Vec Skip-gram](https://img.shields.io/badge/Word2Vec-Skip--gram%20100d-E76F51?style=flat)](https://radimrehurek.com/gensim/)
[![Lesk IDF WSD](https://img.shields.io/badge/Lesk-IDF%20Weighted%20WSD-2A9D8F?style=flat)](https://www.nltk.org/)
[![scikit-learn TF-IDF](https://img.shields.io/badge/scikit--learn-TF--IDF%201.9-F4A261?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit 1.65](https://img.shields.io/badge/Streamlit-1.65-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NASA ADS API](https://img.shields.io/badge/NASA%20ADS-Corpus%20v1-264653?style=flat)](https://ui.adsabs.harvard.edu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-E9C46A?style=flat)](LICENSE)

**Space Weather Evidence Mapper** is an end-to-end, reproducible NLP extraction pipeline that mines peer-reviewed heliophysics and space weather literature to extract structured causal evidence of solar perturbations on critical infrastructure. Operating across 3,116 NASA ADS abstracts, the system parses dependency syntax trees, disambiguates domain polysemy with corpus-adapted Lesk WSD, extracts `(event, system, effect)` triples across dependency paths, and synthesizes evidence into an interactive Streamlit evidence matrix with agreement metrics and citation-level verification.

---

## Overview & Domain Scope

Space-weather events routinely perturb civilian and orbital technological systems. Evidence of these operational impacts is distributed across thousands of technical publications indexed in the NASA Astrophysics Data System (ADS). This system ingests, filters, and structures this scientific corpus into actionable causal triples with exact sentence and publication provenance.

The domain ontology strictly governs six entity classes and four causal relation types:

| Entity Family | Canonical Entities | Target Infrastructure / Phenomenon |
|---|---|---|
| **EVENT** | `geomagnetic_storm`, `sep_event` | Coronal mass ejections, solar proton events, geomagnetic substorms, solar flares |
| **SYSTEM** | `gnss`, `hf_radio`, `satellites`, `power_grid` | GPS/GLONASS/Galileo receivers, skywave propagation, orbital charging/drag, GIC transformers |

Causal effect assertions are classified into mutually exclusive relation semantics:
- `degradation`: Operational impairment, phase scintillation, telemetry drift, or signal attenuation.
- `outage`: Total functional loss, loss of lock, single-event latchup, or electrical blackout.
- `no_effect`: Explicitly verified resilience or non-impact under active storm conditions.
- `mitigated`: Active technological hardening, shielding, or deflection countermeasure success.

---

## Pipeline Architecture

The pipeline processes scientific text across ten deterministic, version-controlled stages. Intermediate state is serialized as JSONL and metadata under `data/`, guaranteeing full reproducibility:

```
[ Stage 1: NASA ADS Search API ]
              │ (3,116 abstracts, 8 strata, 2000-2026)
              ▼
[ Stage 2: spaCy Linguistic Tokenization & Lemmatization ]
              │ (Tokens, lemmas, Kp/Dst regex, abbreviation extraction)
              ▼
[ Stage 3: TF-IDF Nearest-Seed Relevance Filter ]
              │ (Top-3 nearest seed cosine, cutoff threshold 0.137 -> 1,002 abstracts)
              ▼
[ Stage 4: Domain Language Modeling & Perplexity Verification ]
              │ (Bigram/Trigram Kneser-Ney, held-out cross-domain comparison)
              ▼
[ Stage 5: Synonym Discovery & Lexical Semantic Grouping ]
              │ (Word2Vec Skip-gram 100d + WordNet synset hypernym closure + Acronym resolution)
              ▼
[ Stage 6: Entity Mention Tagging & Disambiguation ]
              │ (PhraseMatcher + Noun-chunk rules + IDF-weighted Extended Lesk WSD)
              ▼
[ Stage 7: Dependency-Path Causal Relation Extraction ]
              │ (Shortest dependency path <= 7 hops, governing verb/noun cue classification)
              ▼
[ Stage 8: Inverted TF-IDF Sentence Index & Retrieval Engine ]
              │ (P@5 = 0.514, Pooled R@10 = 0.778 over 391 judged sentences)
              ▼
[ Stage 9: Multi-Dimensional Evidence Matrix Synthesis ]
              │ (2x4 event x system matrix, agreement ratio, bibliographic traceability)
              ▼
[ Stage 10: Human-in-the-Loop Gold-Standard Evaluation ]
              │ (Cohen's Kappa IAA, bootstrap 95% CIs, error ablation analysis)
```

---

## Technical Foundations

### 1. Linguistic Normalization & Abbreviation Mining (`src/preprocess.py`)
- Standardizes texts through spaCy's convolutional pipeline (`en_core_web_sm`).
- Extracts space-weather geomagnetic indices (`Kp`, `Dst`, `SYM-H`) via compiled numerical regex patterns.
- Mines local abbreviation patterns (e.g. `IRNSS -> Indian Regional Navigation Satellite System`) on a sentence basis, establishing domain synonym links before vector training.

### 2. Nearest-Seed Relevance Filtering (`src/relevance.py`)
- Standard centroid filtering is biased toward dominant GNSS topics, suppressing sparse power grid papers.
- Instead, computes the mean cosine similarity across the $k=3$ nearest seed vectors:
$$\text{Score}(d) = \frac{1}{k} \sum_{s \in \text{Top-}k(d, S)} \cos(v_d, v_s)$$
- Calibrated threshold of `0.137` retains 1,002 high-precision domain abstracts while rejecting generic solar physics and orbital mechanics papers.

### 3. Domain Language Modeling (`src/language_model.py`)
- Implements bigram and trigram models with Maximum Likelihood (MLE), Laplace (add-one), and interpolated Kneser-Ney smoothing.
- Interpolated Kneser-Ney calculates continuation probabilities based on word fertility:
$$P_{\text{KN}}(w_i \mid w_{i-n+1}^{i-1}) = \frac{\max(c(w_{i-n+1}^i) - d, 0)}{c(w_{i-n+1}^{i-1})} + \lambda(w_{i-n+1}^{i-1}) P_{\text{KN}}(w_i \mid w_{i-n+2}^{i-1})$$
- Evaluated against a reference corpus of 1,875 astrophysics abstracts to quantify lexical divergence.

### 4. Corpus Word2Vec & WordNet Anchoring (`src/synonyms.py`)
- Trains a 100-dimensional Skip-gram embedding model ($window=5, min\_count=3, epochs=20$) over Gensim collocation-phrased sentences.
- Synthesizes prototype centroids for canonical entity labels. Unresolved candidates join a prototype if cosine similarity exceeds `0.55` with a margin of `0.05` above the runner-up.
- Hierarchical fallback inspects WordNet noun hypernym closures against anchor synsets (e.g. `magnetic_storm.n.01`, `power_system.n.01`).

### 5. Corpus-Adapted Lesk Word Sense Disambiguation (`src/wsd.py`)
- Resolves polysemous heads (`storm`, `flare`, `drag`, `grid`, `charging`).
- Augments standard dictionary glosses with Word2Vec nearest-neighbor lexical expansions of the canonical space-weather concept.
- Weights context word overlap using inverse document frequency (IDF) computed over the general astrophysics reference corpus:
$$\text{OverlapScore}(S) = \sum_{w \in (\text{Signature}(S) \cap \text{Context})} \text{IDF}_{\text{astro}}(w)$$

### 6. Dependency-Path Relation Extraction (`src/effects.py`)
- Identifies the lowest common ancestor (LCA) in the spaCy dependency tree for each `(EVENT, SYSTEM)` candidate pair.
- Bounded to paths with length $\le 7$ edges.
- Synthesizes five dependency templates:
  - `dep_svo`: Subject-verb-object canonical structure (`"storm degraded GPS"`)
  - `dep_passive`: Passive agentive construct (`"GPS was disrupted by the storm"`)
  - `dep_nominal`: Head noun governor linkage (`"effects of storms on HF communications"`)
  - `dep_prep`: Prepositional temporal/spatial adjunct (`"errors increased during geomagnetic storms"`)
  - `dep_path`: Arbitrary governed paths carrying verified lexical effect cues.
- Lexical negators (`not`, `no`, `never`, `without`) invert positive degradation into `no_effect`.

---

## Empirical Evaluation

Phase 1 benchmark evaluation measured against a dual-annotated human gold set (300 stratified sentences, Label Studio).

### 1. Language Model Perplexity (Held-Out Test Sets)

| Model | Space Weather $\to$ Space Weather | Space Weather $\to$ Astrophysics | Astrophysics $\to$ Space Weather | Astrophysics $\to$ Astrophysics |
|---|---|---|---|---|
| Bigram MLE | $\infty$ (22% zero) | $\infty$ (34% zero) | $\infty$ (32% zero) | $\infty$ (24% zero) |
| Bigram Laplace | 655.6 | 907.2 | 968.4 | 844.6 |
| Trigram Laplace | 2281.3 | 3091.9 | 3435.5 | 2879.0 |
| Bigram Kneser-Ney | 121.6 | 278.2 | 257.8 | 148.7 |
| **Trigram Kneser-Ney** | **103.9** | **279.9** | **265.7** | **130.7** |

### 2. Entity Extraction Performance (Test Gold: 200 Sentences, 339 Mentions)

| Run Configuration | Exact Precision | Exact Recall | Exact F1 | Partial Precision | Partial Recall | Partial F1 |
|---|---|---|---|---|---|---|
| **Full Pipeline** | **0.534** | **0.563** | **0.548** | **0.749** | **0.791** | **0.769** |
| Gazetteer Baseline | 0.425 | 0.428 | 0.426 | 0.689 | 0.693 | 0.691 |
| spaCy Default NER | 0.048 | 0.003 | 0.006 | 0.905 | 0.056 | 0.106 |
| Full (No WSD) | 0.534 | 0.563 | 0.548 | 0.749 | 0.791 | 0.769 |
| Full (No Synonyms) | 0.537 | 0.560 | 0.548 | 0.754 | 0.788 | 0.771 |

### 3. Triple Causal Extraction Performance (Event, System, Effect)

| Pipeline Configuration | Precision | Recall | F1 Score | 95% Bootstrap CI | Beats Baseline |
|---|---|---|---|---|---|
| **Full Dependency Pipeline** | **0.439** | **0.409** | **0.424** | **[0.276, 0.554]** | **90.3% of resamples** |
| Co-occurrence Baseline | 0.242 | 0.545 | 0.336 | [0.222, 0.440] | — |
| Full (No Negation) | 0.439 | 0.409 | 0.424 | [0.276, 0.554] | 90.3% |
| Full (No WSD) | 0.439 | 0.409 | 0.424 | [0.276, 0.554] | 90.3% |
| Full (No Synonyms) | 0.450 | 0.409 | 0.429 | [0.282, 0.561] | 91.2% |

### 4. Inter-Annotator Agreement (IAA)

| Annotation Unit | Cohen's $\kappa$ Target | Pilot Batch (30 Sents) | Test Overlap (60 Sents) | Status |
|---|---|---|---|---|
| EVENT (Token) | $\kappa \ge 0.60$ | 0.860 | 0.744 | Passed |
| SYSTEM (Token) | $\kappa \ge 0.60$ | 0.830 | 0.640 | Passed |
| EFFECT (Pair) | $\kappa \ge 0.60$ | 0.380 | 0.013 | Disclosed Gate Miss |

---

## Interactive Streamlit Interface

The project includes an interactive demonstration application (`app.py`) built on Streamlit:
- **Evidence Matrix Table**: Real-time rendering of the 2×4 event-by-system matrix with agreement metrics (`agree`, `mixed`, `single`, `gap`).
- **Interactive Drill-Down**: Selecting any matrix cell displays every contributing scientific publication.
- **Evidence Quote Viewer**: Extracted sentences highlight event and system entities with color-coded effect labels and direct hyperlinks to the NASA ADS publication records.
- **Dynamic Model Comparison**: Toggle between the Full Dependency Extraction model and the Co-occurrence Baseline on the fly.

To launch the dashboard:
```bash
streamlit run app.py
```

---

## Installation & Setup

### Requirements
- **Python 3.13** (recommended and tested)
- NASA ADS API Key (optional, only required to re-pull abstracts from ADS)

```bash
# 1. Clone the repository
git clone https://github.com/Sids15/space-nlp.git
cd space-nlp

# 2. Configure virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell / CMD
# source .venv/bin/activate     # macOS / Linux

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Download required NLTK corpora
python -c "import nltk; nltk.download('wordnet')"
```

---

## Full Pipeline Execution

Every stage is executable from the command line:

```bash
# 1. Ingest corpus via NASA ADS Search API (requires ADS_API_KEY in .env)
python src/fetch_corpus.py --version v1

# 2. Tokenize, lemmatize, and extract numerical indices
python src/preprocess.py data/corpus/corpus_v1.jsonl data/corpus/reference_v1.jsonl

# 3. Execute TF-IDF nearest-seed relevance filtering
python src/relevance.py --version v1

# 4. Train and compute language model perplexity
python src/language_model.py --version v1

# 5. Train Word2Vec Skip-gram model and compile domain lexicon
python src/synonyms.py --version v1

# 6. Extract entity mentions with WSD and synonym grouping
python src/entities.py --version v1 --mode full

# 7. Extract dependency-path causal triples
python src/effects.py --version v1

# 8. Build co-occurrence baseline triples
python src/effects.py --version v1 --baseline

# 9. Evaluate TF-IDF vs. BoW sentence retrieval engine
python src/retrieval.py evaluate

# 10. Synthesize 2x4 evidence matrices
python src/matrix.py --version v1 --run full
python src/matrix.py --version v1 --run cooccurrence_full

# 11. Run gold-standard evaluation suite
python src/evaluate.py --gold data/gold/exports/test_gold.json --version v1
```

---

## Repository Structure

```
space-nlp/
├── src/
│   ├── domain.py             # Canonical vocabulary, gazetteer phrases, and head rules
│   ├── fetch_corpus.py       # NASA ADS API harvester
│   ├── preprocess.py         # spaCy tokenization, lemmatization, and indices regex
│   ├── relevance.py          # TF-IDF nearest-seed cosine relevance filter
│   ├── language_model.py     # MLE, Laplace, and Kneser-Ney n-gram language models
│   ├── synonyms.py           # Word2Vec Skip-gram embeddings & WordNet anchor mapping
│   ├── wsd.py                # IDF-weighted corpus-adapted Lesk WSD algorithm
│   ├── entities.py           # Hybrid entity tagger (phrase matching + chunk heads + WSD)
│   ├── effects.py            # Bounded dependency-path causal triple classifier
│   ├── retrieval.py          # TF-IDF inverted index sentence retrieval engine
│   ├── matrix.py             # Event x system evidence matrix compiler
│   ├── gold.py               # Gold set sampling, Label Studio sync, kappa & adjudication
│   ├── evaluate.py           # P/R/F1 scoring engine with bootstrap confidence intervals
│   └── parsed.py             # Shared cached spaCy doc & sentence streaming utility
├── app.py                    # Streamlit interactive evidence explorer application
├── tests/                    # Pytest validation test suite
├── data/                     # Versioned artifacts (corpus, triples, matrices, evaluation)
├── docs/                     # Full technical report, results datasheet, and guidelines
├── requirements.txt          # Frozen dependency specifications
├── LICENSE                   # Open-source MIT License
└── README.md                 # Project technical documentation
```

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).

```text
Copyright (c) 2026 Siddhant Bandi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
