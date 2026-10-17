# Phase 1 results: numbers, tables and required disclosures

Every number below comes from a file in this repo; the source is given under each table, so it can be
re-checked. The explanatory text of the report is for the team to write. This file holds only the facts.

## 1. Corpus (Week 1)

| Item | Value |
|---|---|
| Source | NASA ADS search API, 8 event × system strata, 2000–2026, articles and proceedings |
| Abstracts after dedup and filters (≥ 50 words, English) | 3,116 (`corpus_v1`) |
| Astrophysics reference corpus (most-cited, for LM and IDF) | 1,875 (`reference_v1`) |

Source: `data/corpus/*.meta.json`

## 2. Relevance filter (Week 2)

| Item | Value |
|---|---|
| Method | TF-IDF cosine to the 3 nearest of 30 seed abstracts |
| Threshold / kept | 0.137 / 1,002 of 3,116 |
| Correlation TF-IDF vs BoW / vs Jaccard | 0.679 / 0.749 |
| Cut-off check, 25 just above the threshold | 15 relevant (60%) |
| Cut-off check, 25 just below the threshold | 12 relevant (48%) |

Source: `data/relevance/filter_v1.meta.json`, `python src/relevance.py --check`

## 3. Language models (Week 2)

Perplexity on a held-out 10% (≈ 215k training tokens per domain):

| Model | SW → SW | SW → Astro | Astro → SW | Astro → Astro |
|---|---|---|---|---|
| bigram MLE | ∞ (22% zero-prob) | ∞ (34%) | ∞ (32%) | ∞ (24%) |
| bigram Laplace | 655.6 | 907.2 | 968.4 | 844.6 |
| trigram Laplace | 2281.3 | 3091.9 | 3435.5 | 2879.0 |
| bigram Kneser-Ney | 121.6 | 278.2 | 257.8 | 148.7 |
| trigram Kneser-Ney | **103.9** | 279.9 | 265.7 | **130.7** |

SW = space-weather corpus, Astro = reference corpus. Source: `data/lm/perplexity_v1.json`

## 4. Gold set and inter-annotator agreement (Week 5)

| Set | Sentences | Annotated by |
|---|---|---|
| Pilot (used as dev) | 30 | A and B |
| Test | 200 | A; B on tasks 1–60 (overlap) |

| κ (target ≥ 0.6) | Event (token) | System (token) | Effect (pair) |
|---|---|---|---|
| Pilot, 30 sentences | 0.86 | 0.83 | 0.38 (10 units) |
| Test overlap, 60 sentences | 0.74 | 0.64 | 0.01 (25 units) |

Exact-span F1 between annotators on the test overlap: event 0.56, system 0.44.
The agreement gate was **not** met for effects.

Source: `data/gold/kappa_pilot.json`, `data/gold/kappa_round1.json`

## 5. Extraction results on the test gold set (Week 6)

Gold: 200 sentences, 339 entity mentions, 44 (event, system, effect) triples.

### Entities (events + systems)

| Run | Exact P | Exact R | Exact F1 | Partial P | Partial R | Partial F1 |
|---|---|---|---|---|---|---|
| **full pipeline** | 0.503 | 0.531 | 0.516 | **0.749** | 0.791 | 0.769 |
| gazetteer only (baseline) | 0.425 | 0.428 | 0.426 | 0.689 | 0.693 | 0.691 |
| spaCy NER (baseline) | 0.048 | 0.003 | 0.006 | 0.905 | 0.056 | 0.106 |
| full, no WSD (ablation) | 0.503 | 0.531 | 0.516 | 0.749 | 0.791 | 0.769 |
| full, no synonyms (ablation) | 0.506 | 0.528 | 0.517 | 0.754 | 0.788 | 0.771 |

By slot (full, exact): event P 0.625 / R 0.638 / F1 0.631; system P 0.413 / R 0.447 / F1 0.429.

### Triples (event, system, effect)

| Run | P | R | F1 | F1 95% CI | Beats baseline in |
|---|---|---|---|---|---|
| **full pipeline** | 0.450 | 0.409 | **0.429** | 0.282–0.559 | 91.6% of resamples |
| co-occurrence (baseline) | 0.242 | 0.545 | 0.336 | 0.222–0.440 | n/a |
| full, no negation (ablation) | 0.450 | 0.409 | 0.429 | 0.282–0.559 | 91.6% |
| full, no WSD (ablation) | 0.450 | 0.409 | 0.429 | 0.282–0.559 | 91.6% |
| full, no synonyms (ablation) | 0.462 | 0.409 | 0.434 | 0.286–0.568 | 92.1% |

Pairs only (effect ignored): full F1 0.434, co-occurrence F1 0.517 (recall 0.841).
CI = 1,000 bootstrap resamples over sentences, seed 42.

### Error breakdown (full pipeline, automatic categories)

| Error | Category | Count |
|---|---|---|
| Entity false positive | boundary (overlaps a gold span, same label) | 91 |
| | spurious (no gold overlap) | 79 |
| | wrong label | 8 |
| Entity false negative | boundary | 89 |
| | missed entirely | 61 |
| | wrong label | 9 |
| Triple false positive | pair not in gold | 21 |
| | right pair, wrong effect | 1 |
| Triple false negative | pair missed | 26 |

All 7 gold `no_effect` triples are missed (the pipeline emits `no_effect` once in the whole corpus).
A seeded 50-error sample for the hand-done analysis is in `data/eval/errors_test_gold_v1_sample50.csv`
(kept out of git because it shows gold labels).

Source: `data/eval/results_test_gold_v1.json`

## 6. Evidence matrix (full pipeline)

| Event → system | Papers | Majority effect | Agreement | Status |
|---|---|---|---|---|
| Storm → GNSS | 85 | degradation | 0.955 | agree |
| Storm → HF radio | 20 | degradation | 0.714 | mixed |
| Storm → satellites | 24 | degradation | 0.875 | agree |
| Storm → power grid | 89 | degradation | 0.925 | agree |
| SEP → GNSS | 22 | degradation | 0.955 | agree |
| SEP → HF radio | 21 | degradation | 0.545 | mixed |
| SEP → satellites | 13 | degradation | 1.000 | agree |
| SEP → power grid | 0 | n/a | n/a | gap |

Source: `data/matrix/matrix_full_v1.json`

## 7. Phase 1 gate

| Criterion | Result | Met |
|---|---|---|
| ≥ 500 abstracts | 1,002 after filtering | ✅ |
| κ ≥ 0.6 | event 0.74, system 0.64, effect 0.01 | ❌ (effect) |
| Entity precision ≥ 0.75 | 0.749 partial / 0.503 exact | ❌ |
| Triple F1 beats co-occurrence | 0.429 vs 0.336 | ✅ |
| P@5 on ≥ 20 queries | judgments not yet made | ⏳ |
| Live demo | `streamlit run app.py` | ✅ |

## 8. Disclosures the report must make

1. **AI-written code.** The pipeline code was written with an AI assistant (Claude). The PRD says
   "all pipeline code is written by the team", and the team accepted this deviation.
2. **AI-chosen seeds and AI-labelled relevance check.** The 30 seed abstracts and the y/n labels of the
   50-abstract cut-off sample (section 2) were produced by the AI assistant, not by the team.
3. **Human gold labels.** All gold labels (pilot, test) were made by two human annotators in Label Studio.
4. **Disagreement resolution.** Disagreements on the double-annotated sentences were resolved by
   **seeded random selection of one annotator's whole sentence annotation** (seed 42), not by discussion.
   The same was done for the pilot to form the dev set. Choices: `data/gold/exports/*.adjudication.json`.
5. **Single annotation.** Test sentences 61–200 were labelled by annotator A only.
6. **Tuning.** Boundary and list rules were tuned on the 30-sentence dev set only. The test set was scored
   once before tuning and once after; nothing was changed after the second test run.
7. **Sample bias.** The gold sample favours abstracts that state an event and a system explicitly.
8. **Environment.** Python 3.13.7 instead of the PRD's 3.11.
