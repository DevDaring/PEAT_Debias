"""
PEAT-CB — Coverage-balanced counterfactual training pairs (WP-G).

StereoSet intrasentence (the only PEAT training source) covers four bias
domains: gender, profession, race, religion. CrowS-Pairs evaluates nine
categories; disability, sexual orientation, age, physical appearance,
socioeconomic status, and nationality receive no equalisation signal during
training, which explains the per-category failure pattern flagged by
Reviewers #1 and #4 (disability 67.50, sexual orientation 67.56 in the
submitted Table 2).

This module generates StereoSet-format (context-with-BLANK, t_s, t_a) pairs
for the uncovered categories from templates crossed with descriptor pairs in
the style of HolisticBias (Smith et al., EMNLP 2022, arXiv:2205.09209 —
descriptor-based coverage). CrowS-Pairs is never read here: the evaluation set
stays untouched, so there is no leakage.

# Implements the coverage-matched augmentation of WP-G,
# Submission/proposed_improvement.md (APIN-D-26-06244 revision).
"""
from __future__ import annotations

import hashlib

import pandas as pd

# Each entry: (stereo_descriptor t_s, anti_descriptor t_a).
# t_s is the group that the *stereotypical* completion of the template targets,
# mirroring StereoSet's (stereotype, anti-stereotype) filler convention.
DESCRIPTOR_PAIRS: dict[str, list[tuple[str, str]]] = {
    "disability": [
        ("disabled", "able-bodied"), ("blind", "sighted"), ("deaf", "hearing"),
        ("wheelchair-using", "walking"), ("autistic", "neurotypical"),
        ("paralyzed", "healthy"), ("epileptic", "healthy"),
        ("mentally ill", "mentally healthy"),
    ],
    "sexual-orientation": [
        ("gay", "straight"), ("lesbian", "heterosexual"), ("bisexual", "straight"),
        ("homosexual", "heterosexual"), ("queer", "straight"),
    ],
    "age": [
        ("elderly", "young"), ("old", "young"), ("teenage", "middle-aged"),
        ("senior", "junior"), ("aging", "youthful"),
    ],
    "physical-appearance": [
        ("overweight", "thin"), ("fat", "slim"), ("short", "tall"),
        ("ugly", "attractive"), ("scarred", "unblemished"), ("bald", "full-haired"),
    ],
    "socioeconomic": [
        ("poor", "wealthy"), ("homeless", "housed"), ("uneducated", "educated"),
        ("working-class", "upper-class"), ("unemployed", "employed"),
    ],
    "nationality": [
        ("Mexican", "American"), ("Indian", "British"), ("Nigerian", "German"),
        ("Vietnamese", "French"), ("Afghan", "Canadian"),
    ],
}

# Neutral-competence / social-role templates. BLANK is filled by a descriptor.
# The sentence frames a negative or limiting association, so the pair
# (t_s fills BLANK stereotypically, t_a anti-stereotypically) carries the same
# equalisation signal shape as a StereoSet intrasentence instance.
TEMPLATES: list[str] = [
    "The BLANK man was assumed to be bad at his job.",
    "The BLANK woman was not expected to succeed.",
    "Everyone doubted that the BLANK person could finish the project.",
    "The BLANK neighbor was blamed for the trouble on the street.",
    "People thought the BLANK student would fail the exam.",
    "The manager hesitated to hire the BLANK applicant.",
    "The BLANK customer was watched closely by the shopkeeper.",
    "Nobody wanted to sit next to the BLANK passenger on the bus.",
    "The BLANK employee was passed over for the promotion.",
    "The landlord refused to rent the flat to the BLANK couple.",
    "The BLANK driver was assumed to have caused the accident.",
    "The teacher expected little from the BLANK child.",
]


def generate_cb_pairs() -> pd.DataFrame:
    """Cross templates × descriptor pairs → StereoSet-format training pairs.

    Deterministic (no RNG): the full cross product is returned, deduplicated.
    Columns match the cached StereoSet parquet so StereoSetDataset and
    compute_peat_loss consume them unchanged.
    """
    rows = []
    for bias_type, pairs in DESCRIPTOR_PAIRS.items():
        for template in TEMPLATES:
            for t_s, t_a in pairs:
                context = template
                uid = hashlib.md5(
                    f"{bias_type}|{template}|{t_s}|{t_a}".encode()
                ).hexdigest()[:12]
                rows.append({
                    "id": f"cb-{uid}",
                    "target": t_s,
                    "bias_type": bias_type,
                    "context": context,
                    "stereotype_sentence": context.replace("BLANK", t_s),
                    "anti_stereotype_sentence": context.replace("BLANK", t_a),
                    "t_s": t_s,
                    "t_a": t_a,
                })
    df = pd.DataFrame(rows).drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df


def load_cb_augmented_pairs(seed: int = 42) -> pd.DataFrame:
    """StereoSet 90% training split + generated coverage pairs, shuffled.

    This is the PEAT-CB training corpus: identical to canonical PEAT training
    plus equalisation signal for the six uncovered CrowS categories.
    """
    from peat.data import load_stereoset_pairs
    from peat.utils import SMOKE_TEST, SMOKE_TEST_SIZE

    train_df, _ = load_stereoset_pairs(seed=seed)
    cb_df = generate_cb_pairs()
    if SMOKE_TEST:
        cb_df = cb_df.head(SMOKE_TEST_SIZE).reset_index(drop=True)
    common = [c for c in train_df.columns if c in cb_df.columns]
    merged = pd.concat([train_df[common], cb_df[common]], ignore_index=True)
    merged = merged.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return merged
