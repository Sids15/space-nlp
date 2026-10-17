"""Shared domain vocabulary: canonical labels, gazetteer phrases and chunk-head rules.

Event families and system families follow the PRD domain restriction. Every stage (tagging,
synonym grouping, gold-set labels) reads its labels from here so they never drift apart.
"""

EVENTS = ("geomagnetic_storm", "sep_event")
SYSTEMS = ("gnss", "hf_radio", "satellites", "power_grid")
EFFECTS = ("degradation", "outage", "no_effect", "mitigated")
LABEL_OF = {**{c: "EVENT" for c in EVENTS}, **{c: "SYSTEM" for c in SYSTEMS}}

# Lowercase phrases, matched case-insensitively. Plurals are generated automatically.
PHRASES = {
    "geomagnetic_storm": [
        "geomagnetic storm", "magnetic storm", "geomagnetic superstorm", "superstorm", "solar storm",
        "ionospheric storm", "geomagnetic disturbance", "geomagnetic activity", "geomagnetic substorm",
        "substorm", "storm sudden commencement", "sudden storm commencement", "sudden commencement",
    ],
    "sep_event": [
        "solar energetic particle", "solar energetic particle event", "solar proton event",
        "solar particle event", "sep event", "proton event", "radiation storm", "solar radiation storm",
        "ground level enhancement", "solar flare", "solar radio burst",
    ],
    "gnss": [
        "global positioning system", "global navigation satellite system", "satellite navigation",
        "satellite-based navigation", "precise point positioning", "real-time kinematic",
        "gnss receiver", "gps receiver", "gps signal", "gnss signal", "navigation satellite system",
        "satellite based augmentation system", "wide area augmentation system",
    ],
    "hf_radio": [
        "hf radio", "high frequency radio", "high-frequency radio", "hf communication",
        "hf propagation", "hf radio communication", "hf link", "hf signal", "shortwave",
        "shortwave fadeout", "short-wave fadeout", "radio blackout", "radio communication",
        "polar cap absorption", "near vertical incidence skywave",
    ],
    "satellites": [
        "satellite anomaly", "spacecraft anomaly", "spacecraft charging", "surface charging",
        "internal charging", "deep dielectric charging", "electrostatic discharge",
        "single event upset", "single-event upset", "single event effect", "single-event effect",
        "single event latchup", "atmospheric drag", "satellite drag", "orbital decay", "orbit decay",
        "satellite electronics", "solar array", "cubesat",
    ],
    "power_grid": [
        "power grid", "electric grid", "electrical grid", "electric power grid", "power system",
        "electric power system", "power network", "transmission network", "transmission line",
        "power line", "power transformer", "substation", "geomagnetically induced current",
        "geomagnetically-induced current", "geomagnetic induced current",
    ],
}

# Case-sensitive acronyms and names (exact token match), incl. common plurals.
ACRONYMS = {
    "geomagnetic_storm": ["GMD", "GMDs", "SSC", "SSCs"],
    "sep_event": ["SEP", "SEPs", "SPE", "SPEs", "GLE", "GLEs"],
    "gnss": ["GNSS", "GPS", "GLONASS", "BeiDou", "BDS", "Galileo", "NavIC", "IRNSS", "QZSS",
             "SBAS", "WAAS", "EGNOS", "PPP", "RTK"],
    "hf_radio": ["HF", "NVIS", "PCA"],
    "satellites": ["SEU", "SEUs", "SEE", "SEEs", "SEL", "ESD", "Starlink"],
    "power_grid": ["GIC", "GICs"],
}

# Noun-chunk heads (lemma) that signal an entity even when no gazetteer phrase is present.
# value: (label, canonical or None, WSD key or None). None canonical = decided by synonym grouping
# or by a gazetteer modifier inside the chunk.
HEADS = {
    "storm": ("EVENT", "geomagnetic_storm", "storm"),
    "superstorm": ("EVENT", "geomagnetic_storm", None),
    "substorm": ("EVENT", "geomagnetic_storm", None),
    "flare": ("EVENT", "sep_event", "flare"),
    # "event" is not a head: a bare or generic "space weather event" names no family. Family
    # modifiers ("proton event", "SEP event") are gazetteer phrases instead.
    "satellite": ("SYSTEM", "satellites", None),
    "spacecraft": ("SYSTEM", "satellites", None),
    "receiver": ("SYSTEM", None, None),
    "positioning": ("SYSTEM", "gnss", None),
    "navigation": ("SYSTEM", "gnss", None),
    "grid": ("SYSTEM", "power_grid", "grid"),
    "transformer": ("SYSTEM", "power_grid", None),
    "radio": ("SYSTEM", None, None),
    "communication": ("SYSTEM", None, None),
    "drag": ("SYSTEM", "satellites", "drag"),
    "charging": ("SYSTEM", "satellites", "charging"),
}


def phrase_variants(phrase):
    return [phrase] if phrase.endswith("s") else [phrase, phrase + "s"]
