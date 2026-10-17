# How to label the sentences: plain-language guide

You get one sentence at a time from a space-science paper. For each one you do two jobs:

1. **Highlight the things:** space-weather events (the cause) and technology (the victim).
2. **Connect them with an arrow,** but only when the sentence says the event did something to the technology.

Then press **Submit**. Read **only that one sentence**: not the paper, not the sentence before it,
not what you happen to know about the topic.

---

## Job 1: Highlight the things

### The 6 labels

**Events (the cause): 2 labels**

| Key | Label | What it means | Words you will see |
|---|---|---|---|
| 1 | **STORM** | A disturbance of Earth's magnetic field | geomagnetic storm, magnetic storm, superstorm, substorm, solar storm, ionospheric storm, geomagnetic disturbance, GMD, sudden commencement, "the storm", G1–G5 storm |
| 2 | **SEP** | Something bursting off the Sun | solar flare, X-class flare, M9.6 flare, solar energetic particles, solar proton event, SEP event, **solar radiation storm**, ground level enhancement, solar radio burst, S1–S5 storm |

**Technology (the victim): 4 labels**

| Key | Label | What it means | Words you will see |
|---|---|---|---|
| 3 | **GNSS** | Satellite navigation | GPS, GNSS, GLONASS, Galileo, BeiDou, NavIC, WAAS, EGNOS, **navigation satellite system**, GPS receivers, positioning, PPP |
| 4 | **HF_RADIO** | Shortwave (HF) radio | HF radio, high-frequency communication, HF propagation, shortwave, **radio blackout**, R1–R5 blackout, polar cap absorption |
| 5 | **SATELLITES** | Spacecraft, and the ways space weather harms them | satellites, spacecraft, LEO satellites, named spacecraft (Starlink, CRRES, SOHO, Inmarsat…), single event upsets, single event effects, spacecraft charging, satellite drag, atmospheric drag, orbital decay, spacecraft electronics |
| 6 | **POWER_GRID** | The electricity network | power grid, power system, transmission lines, transformers, substations, **GIC / geomagnetically induced currents** |

### How much to highlight

Highlight the **name of the thing plus the words that describe it**. Leave out:
- little words in front: *the, a, an, this, its, several, many, some, 40, two*
- *especially* and similar words
- tails after the name: *"…of 2003"*, *"…in Ethiopia"*, *"…at high latitudes"*

| Sentence piece | Highlight |
|---|---|
| the severe magnetic storm of 1989 | `severe magnetic storm` |
| several high-voltage transformers | `high-voltage transformers` |
| the 2012 Bastille Day storm | `2012 Bastille Day storm` (years and names stay) |
| especially BeiDou | `BeiDou` |
| 12 Starlink satellites | `Starlink satellites` (the count goes) |

**Special cases**

- **An acronym in brackets goes in the same highlight as its name:**
  `global positioning system (GPS)` and `solar proton events (SPEs)` are one highlight each.
- **A list joined by "and", "or" or commas gets separate highlights,** one per item. Never stretch
  one highlight across "and" or a comma.
  *"GPS and BeiDou receivers"* → `GPS` + `BeiDou receivers`
- **When the technology name only describes something else, highlight just the name:**
  *"GPS TEC maps"* → `GPS`; *"GIC measurements"* → `GIC`; *"solar flare detector"* → `solar flare`
- **Highlight every mention, even when it is only a tool or a data source:**
  *"Using Galileo data, we…"* → still highlight `Galileo`. (It just won't get an arrow.)
- **Named spacecraft count:** if something is launched, orbits or "returned data", it's a spacecraft → SATELLITES.

### Watch out: easy mistakes

| Trap | Correct label |
|---|---|
| **solar radiation storm**, S1–S5 storm | **SEP**, not STORM (it is a particle storm from the Sun) |
| **navigation satellite system**, GPS satellites | **GNSS**, not SATELLITES |
| **GIC**, geomagnetically induced currents | **POWER_GRID**; it is the current inside the grid, *not* an event |
| **radio blackout** | **HF_RADIO** |
| NOAA scales | **G** = STORM, **S** = SEP, **R** = HF_RADIO |

### Never highlight these

- **Vague words:** space weather, space weather event, solar activity, space environment, solar cycle,
  "this event", "this variation", "it". (*"the storm"* is fine: it contains the word storm.)
- **Drivers of storms:** CME / coronal mass ejection, solar wind, IMF, high-speed streams.
- **Numbers and indices:** Kp, Dst, SYM-H values, "−305 nT", "Kp = 8".
- **The ionosphere and its changes:** TEC, electron density, ionospheric irregularities,
  ionospheric disturbances, plasma bubbles, scintillation.
- **Measurements and data:** decay profile, harmonic measurements, signatures, data.
- **Generic technology:** "radio signals", "radio waves", "manmade technologies", "infrastructure".
  Radio counts only when it says **HF / high-frequency / shortwave** or **radio blackout**.
- **Charging of anything that isn't a spacecraft:** *"lunar surface charging"* (that's the Moon).
- **Weather:** hurricanes, thunderstorms, tropical storms.
- **Places and times:** regions, latitudes, SAA, "recovery phase", "main phase".
- **Forecasting tools:** neural networks, models.

---

## Job 2: Connect them with an arrow

### Two questions before drawing anything

1. **Is there an event highlight (STORM or SEP) *and* a technology highlight?**
   If not, there is **no arrow**. Submit.
2. **Does this sentence itself say the event did something to the technology,** or that it did nothing?
   If you need another sentence or your own knowledge to connect them, there is **no arrow**.

### The 4 arrow labels

Draw the arrow **from the event to the technology**, then choose:

| Label | Plain meaning | Words that signal it |
|---|---|---|
| **degradation** | made it worse | degraded, disrupted, affected, impacted, damaged, caused errors, hazards, threat, effects of X on Y, impact of X on Y, overheating, "partly" stopped |
| **outage** | made it stop completely | blackout, failure, loss of lock, lost (satellites), shut down, tripped, forced to deorbit |
| **no_effect** | the sentence says nothing happened | did not affect, no significant effect, unaffected, negligible impact |
| **mitigated** | someone reduced or prevented the harm | mitigate, blocking devices, protection, prevented, reduced the GIC |

### Arrow rules

- **"May", "could", "potentially", "might" still count.** *"Storms could damage transformers"* → arrow, degradation.
- **Modelled or calculated effects count.** *"The effects of storms on the grid are calculated"* → arrow, degradation.
- **"Effects / impact of [event] on [technology]"** → degradation.
- **"Partly" or "partial" loss** → degradation, not outage.
- **Just mentioning both is not an effect.** *"We analysed GPS data during the storm"* → no arrow.
- **The event changed the ionosphere only** (*"the flare increased TEC"*) → no arrow. The ionosphere is not technology.
- **Technology → technology is never an arrow.** *"GICs damage transformers"* has no event, so no arrow.
- **One arrow per event-to-family pair is enough.** If you draw several arrows between the same event and the same
  family (e.g. storm → `satellites` and storm → `atmospheric drag`), **give them all the same label,
  the strongest one the sentence states** (outage beats degradation).
- **Listing things side by side is not a cause.** *"A G4 storm, an S1 radiation storm and an R2 blackout
  occurred"* → no arrows.

---

## Worked examples (not from the labelling set)

**1.** *"The 2003 Halloween storm caused large positioning errors in single-frequency GPS receivers."*
- `2003 Halloween storm` → STORM; `single-frequency GPS receivers` → GNSS
- Arrow: storm → receivers, **degradation**

**2.** *"During the X17 flare, HF radio communication over the dayside was completely lost for two hours."*
- `X17 flare` → SEP; `HF radio communication` → HF_RADIO
- Arrow: flare → HF radio communication, **outage** ("completely lost")

**3.** *"The New Zealand power network was not significantly affected by the moderate storm."*
- `New Zealand power network` → POWER_GRID; `moderate storm` → STORM
- Arrow: storm → power network, **no_effect**

**4.** *"Neutral-point blocking devices reduced geomagnetically induced currents (GICs) during the 2017 storms."*
- `geomagnetically induced currents (GICs)` → POWER_GRID; `2017 storms` → STORM
- Arrow: storms → GICs, **mitigated**

**5.** *"We used BeiDou observations to study the ionospheric response to the geomagnetic storm."*
- `BeiDou` → GNSS; `geomagnetic storm` → STORM
- **No arrow.** BeiDou is only the data source.

**6.** *"Solar proton events (SPEs) can trigger single event upsets in satellite memory."*
- `Solar proton events (SPEs)` → SEP; `single event upsets` → SATELLITES; `satellite memory` → SATELLITES
- Arrows: SPEs → single event upsets **degradation** (and, if you draw it, SPEs → satellite memory **degradation**)

**7.** *"The Dst index reached −250 nT and TEC increased by 80% over Brazil."*
- Highlight **nothing**. Submit. (Numbers and TEC are never highlighted, and no event is named.)

**8.** *"GICs can overheat transformers in the grid."*
- `GICs` → POWER_GRID; `transformers` → POWER_GRID; `grid` → POWER_GRID
- **No arrow.** There is no STORM or SEP, so it is technology → technology.

---

## Working tips

- Most sentences take **20–60 seconds**. Many get no highlights at all, and that is a correct answer: **submit them**.
- **Always press Submit.** A skipped sentence is missing data, while an empty submission is a valid answer.
- If unsure, **make your best call, note the ID, move on**. Don't spend 5 minutes on one sentence.
- **Never look at the other annotator's labels** until both of you have exported. Agreement only means
  something if you worked independently.
- Take a break every 50 sentences. Tired annotators drift.

## In Label Studio

- **Highlight:** press the number key (1–6) or click the label, then drag across the words.
- **Change a label:** click the highlight, then press the correct number key.
- **Arrow:** click the event highlight, click *Create relation* in the right panel (or press Alt+R),
  then click the technology highlight. Open the **Relations** tab on the right and pick the label from the dropdown.
- **Export:** Export → **JSON** (not JSON-MIN, not CSV), saved in `data/gold/exports/`.

## After labelling: agreement check (for the team lead)

```
.venv/Scripts/python src/gold.py kappa data/gold/exports/pilot_A.json data/gold/exports/pilot_B.json --out pilot
.venv/Scripts/python src/gold.py diff  data/gold/exports/pilot_A.json data/gold/exports/pilot_B.json
```
Discuss every disagreement that `diff` lists, agree on a rule, and **add that rule with a new example
to this guide**. Then label the test set (both annotators, independently) and run kappa again with
`--out round1`. The target is κ ≥ 0.6 for event, system and effect.
