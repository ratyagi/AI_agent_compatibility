"""
labeling_utils.py
High levrl labeling of AI agent compatibility.
This module:
- Defines phrase lists
- Parses TOS text and robots.txt.
- Computes scores and labels like (friendly_label,compat_score,scraping_perm, ai_training_perm, derivative_perm
"""

import re
from typing import Dict, Optional, Any


#Types of phrases to look for in TOS
PROHIBITIVE_PHRASES = [
    r"no automated access",
    r"may not scrape",
    r"may not (crawl|spider|harvest)",
    r"scraping (is )?prohibited",
    r"bots? (are )?prohibited",
    r"no bots",
    r"automated (tools|scripts) (are )?not allowed",
    r"you must not (use|deploy) (any )?bots?",
    r"no (data )?mining",
    r"may not use .* for (machine learning|artificial intelligence)",
    r"may not use .* to (train|improve) (ai|ml|models?)",
]

NEUTRAL_PHRASES = [
    r"with prior written consent",
    r"subject to (our )?approval",
    r"only with our permission",
    r"you must comply with (our )?robots\.txt",
    r"as permitted by applicable law",
    r"for internal use only",
]

FRIENDLY_PHRASES = [
    r"automated access (is )?allowed",
    r"you may crawl",
    r"crawling (is )?permitted",
    r"you may scrape",
    r"for research purposes",
    r"public api available",
    r"subject to this license you may (copy|redistribute|reuse)",
]

AI_FORBID_PHRASES = [
    r"may not use .* for (machine learning|artificial intelligence)",
    r"prohibit(ed)? .* use .* for training (ai|ml|models?)",
    r"you may not (train|improve) (ai|ml|models?)",
]

AI_ALLOW_PHRASES = [
    r"may use .* for (machine learning|artificial intelligence)",
    r"may use .* to (train|improve) (ai|ml|models?)",
    r"permitted .* use .* for (research|academic) purposes",
]

DERIV_FORBID_PHRASES = [
    r"may not create derivative works",
    r"no derivative works",
    r"you may not modify",
]

DERIV_ALLOW_PHRASES = [
    r"you may create derivative works",
    r"you may modify",
    r"you may adapt",
]


def normalize_text(text: str) -> str:
    """Lowercase and remove whitespace."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def count_matches(text: str, patterns) -> int:
    """Counter for phrases in the text."""
    if not text:
        return 0
    count = 0
    for pat in patterns:
        count += len(re.findall(pat, text, flags=re.IGNORECASE))
    return count


def analyze_tos_text(tos_text: str) -> Dict[str, Any]:
    """
    analyze TOS text for friendliness or hostility.
    Returns a dict with pre-defined labels
    """
    text = normalize_text(tos_text or "")

    hostile_count = count_matches(text, PROHIBITIVE_PHRASES)
    neutral_count = count_matches(text, NEUTRAL_PHRASES)
    friendly_count = count_matches(text, FRIENDLY_PHRASES)

    ai_forbid = count_matches(text, AI_FORBID_PHRASES) > 0
    ai_allow = count_matches(text, AI_ALLOW_PHRASES) > 0

    deriv_forbid = count_matches(text, DERIV_FORBID_PHRASES) > 0
    deriv_allow = count_matches(text, DERIV_ALLOW_PHRASES) > 0

    return {
        "hostile_count": hostile_count,
        "neutral_count": neutral_count,
        "friendly_count": friendly_count,
        "ai_forbid": ai_forbid,
        "ai_allow": ai_allow,
        "deriv_forbid": deriv_forbid,
        "deriv_allow": deriv_allow,
    }


def parse_robots_txt(robots_text: str) -> Dict[str, Any]:
    """minimal robots.txt parser focuses on User-agent: * section and disallow/allow."""
    if not robots_text:
        return {
            "has_any_robots": False,
            "ua_star_disallow_root": False,
            "ua_star_has_disallow": False,
            "ua_star_has_allow": False,
        }

    lines = [l.strip() for l in robots_text.splitlines()]
    current_ua_star = False
    ua_star_disallow_root = False
    ua_star_has_disallow = False
    ua_star_has_allow = False

    for line in lines:
        if not line or line.startswith("#"):
            continue
        lower = line.lower()

        if lower.startswith("user-agent:"):
            ua = lower.split(":", 1)[1].strip()
            current_ua_star = (ua == "*" or ua == '"*"')
            continue

        if not current_ua_star:
            continue

        if lower.startswith("disallow:"):
            ua_star_has_disallow = True
            path = lower.split(":", 1)[1].strip()
            if path == "/" or path == "/*":
                ua_star_disallow_root = True

        elif lower.startswith("allow:"):
            ua_star_has_allow = True

    return {
        "has_any_robots": True,
        "ua_star_disallow_root": ua_star_disallow_root,
        "ua_star_has_disallow": ua_star_has_disallow,
        "ua_star_has_allow": ua_star_has_allow,
    }


def derive_friendly_label(tos_signals: Dict[str, Any],robots_info: Dict[str, Any]) -> int:
    """
    Output friendly_label ∈ {0,1,2} based on TOS + robots.
    based on phrase counter.
    """
    hostile = tos_signals["hostile_count"]
    neutral = tos_signals["neutral_count"]
    friendly = tos_signals["friendly_count"]

    score = friendly - hostile

    if robots_info.get("ua_star_disallow_root"):
        score -= 2
    elif robots_info.get("ua_star_has_allow") and not robots_info.get("ua_star_has_disallow"):
        score += 1

    if score <= -1:
        return 0
    elif score >= 2:
        return 2
    else:
        return 1


def derive_scraping_perm(tos_signals: Dict[str, Any],robots_info: Dict[str, Any]) -> int:
    """output scraping_perm ∈ {0,1,2}."""
    hostile = tos_signals["hostile_count"]
    neutral = tos_signals["neutral_count"]
    friendly = tos_signals["friendly_count"]

    ua_disallow_root = robots_info.get("ua_star_disallow_root", False)
    ua_has_disallow = robots_info.get("ua_star_has_disallow", False)
    ua_has_allow = robots_info.get("ua_star_has_allow", False)

    if ua_disallow_root or hostile >= 2:
        return 0

    if friendly >= 1 and not ua_has_disallow:
        return 2

    if ua_has_disallow or neutral > 0 or hostile == 1:
        return 1

    if ua_has_allow:
        return 2

    return 1


def derive_ai_training_perm(tos_signals: Dict[str, Any]) -> int:
    """Output ai_training_perm ∈ {0,1,2} from TOS signals alone."""
    if tos_signals["ai_forbid"]:
        return 0
    if tos_signals["ai_allow"]:
        return 2
    return 1


def derive_derivative_perm(tos_signals: Dict[str, Any]) -> int:
    """
    Decide derivative_perm ∈ {0,1,2} from TOS signals.

    - 0 if explicit prohibition of derivative works / modification.
    - 2 if explicit allowance.
    - 1 otherwise.
    """
    if tos_signals["deriv_forbid"]:
        return 0
    if tos_signals["deriv_allow"]:
        return 2
    return 1


def compute_compat_score(
    friendly_label: int,
    scraping_perm: int,
    ai_training_perm: int,
    derivative_perm: int,
    html_info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Combine labels into a compat_score ∈ [0, 1].

    Weighted instead of a simple average:
    - friendly_label (ACCESS) has the highest weight.
    - scraping_perm is important.
    - derivative_perm has moderate weight.
    - ai_training_perm has the smallest weight.
    """
    # Map class label to a base access score
    base_map = {0: 0.2, 1: 0.5, 2: 0.8}
    access_score = base_map.get(friendly_label, 0.5)

    def perm_to_unit(p: int) -> float:
        if p <= 0:
            return 0.0
        if p == 1:
            return 0.5
        return 1.0

    scrape_score = perm_to_unit(scraping_perm)
    ai_score = perm_to_unit(ai_training_perm)
    deriv_score = perm_to_unit(derivative_perm)

    # Technical factor from HTTP status (kept as-is)
    technical_factor = 1.0
    if html_info:
        status = html_info.get("status_code")
        try:
            if status is not None:
                status = int(status)
                if status >= 400:
                    technical_factor = 0.7
        except (ValueError, TypeError):
            pass

    # Weighted combination:
    # ACCESS dominates, then scraping, then derivative, then AI training.
    compat_raw = (
        0.6 * access_score +
        0.2 * scrape_score +
        0.15 * deriv_score +
        0.05 * ai_score
    )

    compat_raw *= technical_factor

    compat_raw = max(0.0, min(1.0, compat_raw))
    return float(round(compat_raw, 3))


def auto_label_domain(
    domain: str,
    tos_text: str,
    robots_text: Optional[str] = None,
    html_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """High-level auto-labeling function for a single domain."""
    tos_signals = analyze_tos_text(tos_text or "")
    robots_info = parse_robots_txt(robots_text or "")

    friendly_label = derive_friendly_label(tos_signals, robots_info)
    scraping_perm = derive_scraping_perm(tos_signals, robots_info)
    ai_perm = derive_ai_training_perm(tos_signals)
    deriv_perm = derive_derivative_perm(tos_signals)

    compat_score = compute_compat_score(
        friendly_label,
        scraping_perm,
        ai_perm,
        deriv_perm,
        html_info=html_info,
    )

    notes_parts = []

    if tos_signals["hostile_count"] > 0:
        notes_parts.append(f"hostile_phrases={tos_signals['hostile_count']}")
    if tos_signals["friendly_count"] > 0:
        notes_parts.append(f"friendly_phrases={tos_signals['friendly_count']}")
    if robots_info.get("ua_star_disallow_root"):
        notes_parts.append("robots_disallow_root")
    if robots_info.get("ua_star_has_allow"):
        notes_parts.append("robots_has_allow")

    if tos_signals["ai_forbid"]:
        notes_parts.append("ai_training_forbidden_in_tos")
    if tos_signals["ai_allow"]:
        notes_parts.append("ai_training_allowed_in_tos")

    if tos_signals["deriv_forbid"]:
        notes_parts.append("derivative_forbidden_in_tos")
    if tos_signals["deriv_allow"]:
        notes_parts.append("derivative_allowed_in_tos")

    notes = "; ".join(notes_parts) if notes_parts else "auto-labeled"

    return {
        "domain": domain,
        "friendly_label": int(friendly_label),
        "compat_score": float(compat_score),
        "scraping_perm": int(scraping_perm),
        "ai_training_perm": int(ai_perm),
        "derivative_perm": int(deriv_perm),
        "notes": notes,
    }