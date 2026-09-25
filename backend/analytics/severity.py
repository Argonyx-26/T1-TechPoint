"""Severity scoring: not every alert is equal. Weapon detections are always
critical with no corroboration needed; everything else gets a base weight
from its rule type, then a co-occurrence bonus when multiple distinct alert
types are active in the same zone within a short time window -- e.g. a
crowd surge *and* a restricted-zone intrusion at the same spot outranks
either alone.
"""
from typing import Iterable, Tuple

from backend import config


def band_for_score(score: float) -> str:
    for threshold, label in config.SEVERITY_BANDS:
        if score >= threshold:
            return label
    return "LOW"


def base_weight(rule: str) -> int:
    return config.SEVERITY_WEIGHTS.get(rule, 20)


def co_occurrence_bonus(rule: str, sibling_rules: Iterable[str]) -> int:
    """`sibling_rules`: other distinct alert types active in the same zone
    within the co-occurrence window (may include duplicates of `rule`)."""
    distinct_others = {r for r in sibling_rules if r != rule}
    bonus = len(distinct_others) * config.CO_OCCURRENCE_BONUS
    return min(bonus, config.CO_OCCURRENCE_CAP)


def score_alert(rule: str, sibling_rules: Iterable[str] = ()) -> Tuple[int, str]:
    """Returns (score clamped to 0-100, severity band label)."""
    score = base_weight(rule) + co_occurrence_bonus(rule, sibling_rules)
    score = max(0, min(100, score))
    return score, band_for_score(score)
