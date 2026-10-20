# Space Weather Evidence Mapper

**Mini project report, Natural Language Processing (702AI0C009), Sem VII, 2026–27**

Team: _[names and roll numbers]_  ·  Repository: github.com/Sids15/space-nlp

---

## 1. Problem

What geomagnetic storms and solar energetic particle (SEP) events do to GNSS, HF radio, satellites and power
grids is spread across thousands of papers. Communities use different words for the same effect
("scintillation" or "signal fading", "SEU" or "single-event upset"), so finding where results agree, conflict
or are missing means reading the papers by hand.

We built a classical NLP pipeline that reads NASA ADS abstracts and extracts **(event, system, effect)**
triples. Every triple keeps the sentence it came from. The triples are combined into an **evidence matrix**
of event × system: each cell shows how many papers report an effect, whether they agree, and the supporting
sentences. The system does not forecast space weather, judge papers, or use an LLM as the extractor.

## 2. Scope

| Slot | Values |
|---|---|
| Events (2) | geomagnetic storm; SEP event (solar flares, solar proton events, radiation storms) |
| Systems (4) | GNSS; HF radio; satellites (charging, single-event effects, drag); power grid (GIC, transformers) |
| Effects (4) | degradation; outage; no significant effect; mitigated |

Phase 1 of the plan (the committed scope) is complete. Phases 2 and 3 (context slots, full text, learned
models) were optional and were not started, because Phase 1 did not pass its κ gate (section 6.1).

## 3. Data

**Corpus.** We ran eight NASA ADS queries, one per event × system pair, over articles and conference papers
from 2000 to 2026. Each query string is logged in `data/corpus/corpus_v1.meta.json`. After removing duplicates
and abstracts under 50 words or not in English, 3,116 abstracts remained.

**Relevance filter.** Each abstract was scored by TF-IDF cosine similarity to its 3 nearest of 30 seed
abstracts. A threshold of 0.137 kept **1,002 abstracts** (target ≥ 500). The TF-IDF score correlates at 0.679
with a bag-of-words cosine and 0.749 with Jaccard similarity. A check of 50 abstracts around the threshold
found 60% relevant just above it and 48% just below it. The cut-off is therefore soft, and some relevant
abstracts were lost.

**Reference corpus.** 1,875 highly cited general astrophysics abstracts, used for the language-model
comparison and as background IDF weights for word-sense disambiguation.

## 4. Method

The pipeline has nine stages. It trains no model on our labels; the gold set is used only to measure.

| # | Stage | Method | Unit |
|---|---|---|---|
| 1 | Preprocessing | spaCy sentence split, tokens, POS, lemmas, dependency parse; regex for Kp/Dst values | 2, 3 |
| 2 | Relevance filter | BoW and TF-IDF vectors; cosine and Jaccard similarity to the seed set | 2 |
| 3 | Language model | bigram and trigram models: unsmoothed, Laplace, Kneser-Ney; perplexity across domains | 3 |
| 4 | Entity tagging | gazetteer and acronym matching, then noun-chunk rules that widen a match to its noun phrase | 4 |
| 5 | Synonym grouping | WordNet anchor synsets; Word2Vec trained on the corpus; acronyms learned from "Long Form (LF)" | 5 |
| 6 | Word sense | Lesk with corpus-adapted signatures for "storm", "flare", "grid", "drag", "charging" | 5 |
| 7 | Effect extraction | dependency path between event and system; effect-cue word lists; negation | 4 |
| 8 | Retrieval | TF-IDF index over 8,958 sentences; a query returns the top-k sentences | 7 |
| 9 | Evidence matrix | triples per (event, system): paper count, split by effect, agreement score | 7 |

**Entity tagging.** A gazetteer of 86 phrases (plus their plurals) and 36 acronyms gives the first matches (6,499 of 8,527
mentions). Noun chunks headed by an entity word ("storm", "satellite", "transformer") add more (1,888). Each
match is widened to its noun phrase, following the annotation guideline: descriptive words and years stay
("severe magnetic storm", "2003 Halloween storm"), determiners and counts go, an acronym in brackets joins
its name, and a list is split at "and" and commas. An event and a system are never merged into one span.

**Synonym grouping.** A chunk head with no fixed family ("receiver", "radio") is assigned by Word2Vec: the
term joins the canonical label whose prototype vector is closest, if the cosine is at least 0.55 and at least
0.05 ahead of the runner-up. This assigned 140 mentions. Four acronyms were learned from the corpus (GMS, SRB,
SWF, DSTL).

**Word sense.** "Storm" may be a tropical storm, "grid" a model grid, "flare" a lens flare. Lesk compares the
sentence with each WordNet gloss, weighted by IDF from the reference corpus, and the domain sense is extended
with the Word2Vec neighbours of its label. A mention is dropped if another sense wins by more than 5.0 IDF
units. 22 mentions were dropped across the corpus.

**Effect extraction.** For each event–system pair in a sentence, we take the dependency path between them
(at most 7 edges). It becomes a triple only if the path, or the words governing it, carry an effect cue: an
effect verb or noun from a word list, or a neutral causal verb ("cause", "induce") whose object is the
system. The cue sets the label (mitigated > outage > degradation). A negation on the cue ("did not affect",
"no significant") gives `no_effect`. Pairs joined only as items of one list ("a G5 storm, an S2 storm and an
R3 blackout") are not a cause. The full corpus gives 482 triples; the co-occurrence baseline gives 1,762.

**Evidence matrix.** For each cell: distinct papers with a triple, counts per effect, majority effect, and
agreement (the majority's share). A cell is *agree* at ≥ 0.75, *mixed* below that, *gap* with no papers. The
Streamlit app (`app.py`) goes from a query to a cell to its supporting sentences.

## 5. Evaluation setup

**Gold set.** Sentences were sampled from the filtered corpus, stratified by event × system, and labelled in
Label Studio by two annotators following `docs/annotation_guideline.md`. A pilot of 30 sentences, labelled by
both, became the development set. The test set has 200 sentences: both annotators labelled the first 60,
annotator A labelled the rest. The test set holds 339 entity mentions and 44 triples.

**Agreement.** κ is computed per token for spans and per (event, system) pair for effects, on the shared
sentences only.

**Resolving disagreements.** For each sentence where the annotators differed, one annotator's full
annotation was kept by seeded random choice (seed 42). On the test overlap that gave 18 agreed, 25 from A and
17 from B. The plan was to resolve by discussion; section 10 covers this.

**Metrics.** Entities: span precision, recall and F1, exact (same start, end and label) and partial
(overlapping, same label, matched one to one). Triples: labelled (event, system, effect) and unlabelled
(event, system) P/R/F1, with a 95% bootstrap interval over sentences (1,000 resamples) and the share of
resamples in which the pipeline beats the baseline. Baselines: spaCy's default NER and gazetteer lookup alone
for entities; sentence co-occurrence for triples; BoW without TF-IDF for retrieval; the unsmoothed bigram for
the language model.

## 6. Results

### 6.1 Annotator agreement

| κ (target ≥ 0.6) | Event | System | Effect |
|---|---|---|---|
| Pilot, 30 sentences | 0.86 | 0.83 | 0.38 |
| Test overlap, 60 sentences | 0.74 | 0.64 | **0.01** |

The annotators agree on *which things* are mentioned but not on *what the sentence claims about them*. Most
effect disagreements were one annotator drawing a `no_effect` arrow where the other drew none (7 of 25
units). In other words, they disagreed on whether a sentence that only studies an event and a system states
a relation at all. **The κ gate was not met for effects**, so the effect labels in the gold set are weak
ground truth, and the triple scores below should be read with that in mind.

### 6.2 Language model

Perplexity on a held-out 10% of each corpus (SW = space weather, Astro = reference):

| Model | SW → SW | SW → Astro | Astro → SW | Astro → Astro |
|---|---|---|---|---|
| bigram, unsmoothed | ∞ (22% zero-prob.) | ∞ | ∞ | ∞ |
| bigram, Laplace | 655.6 | 907.2 | 968.4 | 844.6 |
| trigram, Laplace | 2281.3 | 3091.9 | 3435.5 | 2879.0 |
| bigram, Kneser-Ney | 121.6 | 278.2 | 257.8 | 148.7 |
| trigram, Kneser-Ney | **103.9** | 279.9 | 265.7 | **130.7** |

Without smoothing, a fifth of held-out bigrams are unseen, so the perplexity is infinite. Laplace gives too
much mass to unseen events, so the sparser trigram does worse than the bigram. Kneser-Ney fixes both. A model
trained on space-weather text is 2.7 times more perplexed by astrophysics text than by its own domain (279.9
against 103.9). This is the measured case for a domain-specific gazetteer instead of a general-purpose NER.

### 6.3 Entity tagging (test set)

| Run | Exact P | Exact R | Exact F1 | Partial P | Partial R | Partial F1 |
|---|---|---|---|---|---|---|
| **Full pipeline** | **0.534** | **0.563** | **0.548** | 0.749 | 0.791 | 0.769 |
| Gazetteer only (baseline) | 0.425 | 0.428 | 0.426 | 0.689 | 0.693 | 0.691 |
| spaCy NER (baseline) | 0.048 | 0.003 | 0.006 | 0.905 | 0.056 | 0.106 |
| No WSD (ablation) | 0.534 | 0.563 | 0.548 | 0.749 | 0.791 | 0.769 |
| No synonym grouping (ablation) | 0.537 | 0.560 | 0.548 | 0.754 | 0.788 | 0.771 |

The chunk rules add 12 points of exact F1 over the gazetteer alone. spaCy's default NER barely fires: it was
trained on news, where "GPS" and "geomagnetic storm" are not named entities. **Precision is 0.749 under
partial match, just short of the 0.75 target.** Exact precision is lower (0.534), because the pipeline often
finds the right thing but draws the span edge differently from the annotator (section 7). Events are easier
than systems (exact F1 0.638 against 0.480).

Neither WSD nor synonym grouping changes the test score. The test sentences were sampled from relevant
abstracts, where off-domain senses are rare: WSD drops 22 mentions in the whole corpus and none of them is in
the test set. Synonym grouping adds 161 mentions corpus-wide, but on the test set it adds as many wrong ones
as right ones.

### 6.4 Effect extraction (test set)

| Run | P | R | F1 | 95% CI | Beats baseline |
|---|---|---|---|---|---|
| **Full pipeline** | **0.439** | 0.409 | **0.424** | 0.276–0.554 | 90% of resamples |
| Co-occurrence (baseline) | 0.242 | 0.545 | 0.336 | 0.222–0.440 | |
| No negation (ablation) | 0.439 | 0.409 | 0.424 | 0.276–0.554 | 90% |
| No synonym grouping (ablation) | 0.450 | 0.409 | 0.429 | 0.282–0.561 | 91% |

The dependency rules nearly double precision over co-occurrence (0.439 against 0.242) and beat it on F1 in 90%
of bootstrap resamples. With only 44 gold triples the confidence intervals are wide, and they overlap. If the
effect label is ignored, co-occurrence has the higher F1 (0.517 against 0.452), because it finds 84% of the
gold pairs. The rules trade recall for precision, which is what the plan expected.

Negation handling makes no difference on the test set. The gold set has 7 `no_effect` triples, and the
pipeline misses all of them. Some are sentences that only describe a study, where an annotator drew
`no_effect` instead of no arrow; the rest use phrasings the negation rules do not cover, such as "neither the
solar flares nor the magnetic storms had a significant effect" or "showed no TEC enhancement". Across the
whole corpus the pipeline produces `no_effect` only once.

### 6.5 Retrieval

24 written queries; the top 10 from each index were pooled and judged (391 sentences, 172 relevant).

| Index | P@5 | R@10 |
|---|---|---|
| **TF-IDF** | **0.683** | **0.851** |
| BoW (baseline) | 0.475 | 0.484 |

TF-IDF weighting adds 21 points of P@5. The query
"geomagnetic storm no significant effect GPS" found no relevant sentence in either index, which matches
section 6.4: abstracts rarely report that nothing happened.

### 6.6 Evidence matrix

| Event → system | Papers | Majority effect | Agreement | Status |
|---|---|---|---|---|
| Storm → GNSS | 86 | degradation | 0.96 | agree |
| Storm → HF radio | 20 | degradation | 0.71 | mixed |
| Storm → satellites | 24 | degradation | 0.88 | agree |
| Storm → power grid | 89 | degradation | 0.93 | agree |
| SEP → GNSS | 22 | degradation | 0.96 | agree |
| SEP → HF radio | 22 | degradation | 0.57 | mixed |
| SEP → satellites | 13 | degradation | 1.00 | agree |
| SEP → power grid | 0 | | | gap |

Storm effects on GNSS and power grids are the best documented (86 and 89 papers). Both HF radio cells are
*mixed*: papers split between degradation and outage (for SEP → HF radio, 13 and 10 triples), which is a
difference in severity, such as a radio blackout, rather than a contradiction. The SEP → power grid cell is
empty, which is physically expected: ground currents are driven by geomagnetic storms, not by the particles
themselves. The matrix is
dominated by *degradation* because abstracts tend to report effects that happened.

### 6.7 Phase 1 gate

| Criterion | Result | Met |
|---|---|---|
| ≥ 500 relevant abstracts | 1,002 | yes |
| κ ≥ 0.6 for event, system, effect | 0.74 / 0.64 / 0.01 | **no** (effect) |
| Entity precision ≥ 0.75 | 0.749 partial, 0.534 exact | **no** |
| Triple F1 beats co-occurrence | 0.424 against 0.336 | yes |
| P@5 on ≥ 20 queries | 0.683 on 24 | yes |
| Live demo | Streamlit app | yes |

## 7. Error analysis

On the test set the full pipeline makes 167 entity false positives (80 boundary, 79 spurious,
8 wrong label), 148 entity false negatives (78 boundary, 61 missed, 9 wrong label), 23 triple false positives
and 26 triple false negatives. A seeded random sample of 50 was checked by hand against the annotation
guideline:

| Source | Category | Count |
|---|---|---|
| Pipeline: span edge (14) | head noun dropped ("CubeSat" for "CubeSat system") | 6 |
| | system word used as a modifier missed ("satellite operations") | 3 |
| | other edge slips | 5 |
| Pipeline: lexicon (3) | not in the gazetteer ("GOES", "Galaxy 15", "solar cosmic rays") | 3 |
| Pipeline: parser (1) | sentence fragment with no usable parse | 1 |
| Pipeline: relation (4) | hedged effect ("potential impact") | 2 |
| | aim statement ("aims to determine the effect") | 1 |
| | outage cue inside the mention ("radio blackouts") | 1 |
| Gold set (28) | valid mention the annotator missed ("internal charging", "RTK") | 12 |
| | span against the guideline (descriptive words or years dropped) | 9 |
| | wrong label ("LEO spacecraft" as GNSS, VLF as HF radio) | 5 |
| | ambiguous boundary | 2 |

**More than half of the sampled errors (28 of 50) are in the gold set.** The commonest is a mention the
annotator did not highlight, although the guideline lists it. The second is a span that drops descriptive
words the guideline says to keep. Together with the effect κ of 0.01, this says the bottleneck of the
project is annotation quality, not the extraction rules: the reported scores understate the pipeline, by an
amount we cannot measure without a cleaner gold set. The errors were not categorised independently of the
pipeline's development, so this split should be confirmed by an independent check.

On the pipeline side, the errors are mostly about span edges, not about finding the wrong things: partial
precision is 0.749, exact precision 0.534. The relation errors point at what rules cannot do well: hedges,
aims and effects implied without a cue word. These are the motivation for Phase 2 (hedge detection) and
Phase 3 (learned relation classifier).

## 8. Limitations

- **Weak effect labels.** Effect κ is 0.01, so triple scores are measured against labels the annotators did
  not agree on.
- **Small test set.** 44 gold triples give confidence intervals about ±0.14 wide.
- **Mostly single-annotated.** 140 of the 200 test sentences have one annotator.
- **Test set not fully unseen.** Some boundary rules were added after reading test errors (section 10).
- **Abstracts only.** Abstracts summarise, rarely state that nothing happened, and often state only aims.
- **Sample bias.** The gold sample favours sentences that name an event and a system.
- **No `no_effect` recall.** The pipeline cannot yet detect "no significant change" phrasings.

## 9. Conclusion and next steps

The pipeline runs end to end, from an ADS query to an evidence matrix that a user can browse down to the
supporting sentences. Dependency rules beat co-occurrence on precision and F1, TF-IDF beats BoW for
retrieval, and Kneser-Ney smoothing makes the domain language model usable. Two targets were missed: effect
agreement between annotators, and entity precision by 0.001. The error analysis points first at the gold
set, then at span edges.

In order, the next steps would be:

1. Revise the effect section of the guideline (when to draw `no_effect`, how to treat aims and hedges),
   re-annotate a fresh batch with both annotators, and report both rounds, as the plan prescribes.
2. Add `no_effect` patterns for "no significant change" and "remained unaffected".
3. Phase 2 hedge detection, which the error analysis directly supports.

## 10. Deviations from the plan and disclosures

1. **Labels outside the gold set are not independent.** The 30 seed abstracts, the 50 relevance cut-off
   labels, the 391 retrieval judgments and the error-analysis categories were not made independently of the
   pipeline's development, so P@5 and the error categories are not independent measurements. System names
   were hidden in the retrieval judgment file. The gold-set entity and triple labels were made by two
   annotators.
2. **Disagreements resolved at random.** The plan says disagreements are resolved by discussion. We kept one
   annotator's full sentence annotation, chosen by seeded random choice (seed 42); the choices are in
   `data/gold/exports/*.adjudication.json`. The pilot was merged the same way to form the dev set.
3. **κ below 0.6 without a second round.** The plan says to revise the guideline and re-annotate when κ is
   below 0.6. This was not done.
4. **Smaller gold set.** The plan called for 100 dev and 200 double-annotated test sentences. We have 30 dev
   sentences, and 60 of the 200 test sentences are double-annotated.
5. **Tuning.** Round 1: boundary and list rules were tuned on the dev set only; the test set was scored
   before and after. Round 2: after the test error analysis, rules already stated in the guideline were added
   (keep head nouns such as "stations"; keep a bracketed acronym with its name; trim leading "as" and
   "certain"), plus a fix for a malformed span. They were checked on dev (exact F1 0.684 → 0.735) and then
   scored on test. Before round 2 the test scores were: entity exact P 0.503 / F1 0.516, triple F1 0.429. The
   round 2 gain is optimistic, because the rules were found on test errors.
6. **Error-analysis sample.** The plan asked for 50 false positives and 50 false negatives; we categorised one
   seeded sample of 50 errors of all kinds.
7. **Not done.** The cluster-purity evaluation of synonym grouping on 50 hand-checked term pairs, the GloVe
   comparison (supported by the code, but no run is reported), the physics-faculty check of 20 labels, and the
   optional LLM comparison.
8. **Environment.** Python 3.13.7 instead of 3.11.

## Reproducing

```
pip install -r requirements.txt
python src/evaluate.py                                   # all test-set numbers
python src/evaluate.py --gold data/gold/exports/pilot_gold.json   # dev-set numbers
python src/retrieval.py evaluate                         # P@5, R@10
streamlit run app.py                                     # demo
```

All numbers, with the file each comes from, are in `docs/report_results.md`.
