"""Token extraction, title matching, candidate scoring, and volume ranking."""
from __future__ import annotations

import re
from dataclasses import dataclass

from odinslist.core.normalization import (
    MONTH_ABBR_TO_NUM,
    MONTH_NUM_TO_ABBR,
    STOPWORDS,
    normalize_issue_number,
    normalize_publisher,
)


def series_tokens(s: str) -> set:
    """Extract significant tokens from a series title."""
    if not s:
        return set()
    tokens = re.split(r"[^a-z0-9]+", s.lower())
    return {t for t in tokens if t and t not in STOPWORDS}


def extract_story_tokens(story: str) -> set:
    """Extract significant tokens from a story title."""
    if not story:
        return set()
    tokens = re.split(r"[^a-z0-9]+", story.lower())
    return {t for t in tokens if len(t) >= 4 and t not in STOPWORDS}


def normalize_char_name(name: str) -> str:
    """Normalize a character name for comparison."""
    if not name:
        return ""
    s = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", name.lower())).strip()
    return s[4:] if s.startswith("the ") else s


def month_similarity_score(q_month: str, cv_month: str) -> float:
    """Score similarity between two month abbreviations."""
    if not q_month or not cv_month:
        return 0.0
    if q_month == cv_month:
        return 1.0
    try:
        q_num = MONTH_ABBR_TO_NUM.get(q_month.upper())
        cv_num = MONTH_ABBR_TO_NUM.get(cv_month.upper())
        if not q_num or not cv_num:
            return 0.0
        q_idx = int(q_num) - 1
        cv_idx = int(cv_num) - 1
        diff = abs(q_idx - cv_idx)
        diff = min(diff, 12 - diff)
        return {0: 1.0, 1: 0.7, 2: 0.4}.get(diff, 0.0)
    except (ValueError, TypeError):
        return 0.0


def is_valid_series_descriptor(descriptor: str) -> bool:
    """Check if descriptor is actually a series qualifier, not story text."""
    if not descriptor:
        return False

    descriptor_lower = descriptor.lower()

    valid_keywords = ["giant-size", "giant size", "king-size", "king size",
                      "annual", "special", "giant", "super", "super-size",
                      "team-up", "two-in-one", "two in one", "presents",
                      "comics presents", "triple action", "marvel premiere",
                      "showcase", "brave and the bold"]
    if any(kw in descriptor_lower for kw in valid_keywords):
        return True

    if is_anthology_series(descriptor):
        return True

    if len(descriptor) > 40:
        return False
    if descriptor.count("!") >= 2:
        return False
    if any(word in descriptor_lower for word in ["death", "tale", "story", "epic", "saga", "all-new"]):
        return False

    return False


def is_anthology_series(descriptor: str) -> bool:
    """Check if descriptor is actually an anthology series name."""
    if not descriptor:
        return False

    descriptor_lower = descriptor.lower()

    anthology_keywords = ["team-up","team up","two-in-one","two in one","three-in-one","dc comics presents","dc comics presents annual","marvel team-up","marvel team up","marvel team-up annual","marvel two-in-one","marvel two in one","showcase","1st issue special","first issue special","dc special","dc special series","secret origins","the brave and the bold","brave and the bold","batman the brave and the bold","strange tales","tales to astonish","tales of suspense","journey into mystery","journey into mystery annual","adventure comics","action comics","detective comics","adventure into fear","astonishing tales","marvel premiere","marvel spotlight","marvel feature","marvel presents","marvel comics presents","marvel fanfare","marvel preview","marvel super action","marvel super-heroes","marvel super heroes","marvel super special","special marvel edition","marvel collectors' item classics","marvel tales","fantasy masterpieces","marvel chillers","monsters on the prowl","where monsters dwell","where creatures roam","vault of evil","chamber of darkness","chamber of chills","dead of night","supernatural thrillers","tower of shadows","worlds unknown","unknown worlds of science fiction","unknown worlds","creatures on the loose","fear","frankenstein","house of mystery","house of secrets","ghosts","the unexpected","unexpected","secrets of haunted house","the witching hour","weird war tales","star spangled war stories","our army at war","weird western tales","weird mystery tales","time warp","the superman family","creepy","eerie","vampirella","twisted tales","alien worlds","epic illustrated","amazing adventures","amazing adult fantasy","amazing fantasy","all-star comics","sensation comics","whiz comics","planet comics","mystery in space","strange adventures","tales from the crypt","the haunt of fear","the vault of horror","dark horse presents","negative burn","2000 ad","heavy metal","solo","a1","flight","raw","marvel super-teams","world's finest comics","brave and the bold special","venture","a-next","star spangled comics","bizarre adventures","ghostly tales","chilling adventures in sorcery", "what if", "what if?", "what if?...", "what if..."]

    return any(kw in descriptor_lower for kw in anthology_keywords)


def get_series_special_flags(title: str) -> set:
    """Extract special-series qualifiers used for annual/special disambiguation."""
    if not title:
        return set()
    t = title.lower()
    flags = set()
    if re.search(r"\bannual\b", t):
        flags.add("annual")
    if (
        re.search(r"\bspecial\b", t)
        or re.search(r"\bking[-\s]?size\b", t)
        or re.search(r"\bgiant[-\s]?size\b", t)
        or re.search(r"\bsuper[-\s]?special\b", t)
    ):
        flags.add("special")
    return flags


def generate_title_variants(series_title: str, include_aggressive: bool = True) -> list:
    """Generate title variants for searching."""
    if not series_title:
        return []

    variants = [series_title]

    if series_title.startswith("The "):
        variants.append(series_title[4:])
    else:
        variants.append(f"The {series_title}")

    _DROPPABLE_ADJECTIVES = ["Mighty", "Invincible", "Uncanny", "Amazing", "Spectacular", "Incredible"]
    for adj in _DROPPABLE_ADJECTIVES:
        if adj in series_title:
            variants.append(series_title.replace(f"The {adj} ", "").replace(f"{adj} ", ""))

    if include_aggressive:
        if ", " in series_title:
            after_comma = series_title.split(", ", 1)[1]
            variants.append(after_comma)
            variants.append(series_title.replace(", ", " "))

    for suffix in [" Comics", " Comic", " Magazine", " Book"]:
        if series_title.endswith(suffix):
            variants.append(series_title[:-len(suffix)].strip())

    if include_aggressive:
        and_patterns = [
            r'^(.+?)\s+and\s+(?:his|her|the|their)\s+',
            r'^(.+?)\s+and\s+',
        ]
        for pattern in and_patterns:
            match = re.match(pattern, series_title, re.IGNORECASE)
            if match:
                main_title = match.group(1).strip()
                if len(main_title) >= 3:
                    variants.append(main_title)

    if "-" in series_title:
        variants.append(series_title.replace("-", ""))
        variants.append(series_title.replace("-", " "))

    if include_aggressive:
        _ANNUAL_SUFFIXES = ["Annual", "King-Size Special", "King Size Special", "Special",
                            "Giant-Size", "Giant Size", "Super Special"]
        for suffix in _ANNUAL_SUFFIXES:
            if series_title.lower().endswith(suffix.lower()):
                base_title = series_title[:len(series_title) - len(suffix)].strip()
                if base_title:
                    variants.append(base_title)
                    if suffix.lower() != "annual":
                        variants.append(f"{base_title} Annual")
                break

    seen = set()
    result = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            result.append(v)

    return result


def title_matches(cv_title: str, query_title: str, threshold: float = 0.5) -> bool:
    """Fuzzy title matching using token overlap."""
    cv_tok, q_tok = series_tokens(cv_title), series_tokens(query_title)
    if not cv_tok or not q_tok:
        return False

    overlap = cv_tok & q_tok
    jaccard = len(overlap) / len(cv_tok | q_tok)
    if jaccard >= threshold:
        return True

    cv_main_words = {t for t in cv_tok if len(t) >= 4}
    q_main_words = {t for t in q_tok if len(t) >= 4}

    if cv_main_words and q_main_words:
        main_overlap = cv_main_words & q_main_words
        if main_overlap and len(main_overlap) / max(len(cv_main_words), len(q_main_words)) >= threshold:
            return True

    return False


def publisher_matches(q_pub: str, cv_pub: str) -> bool:
    """Check if publisher names match."""
    if not q_pub or not cv_pub:
        return False
    q, c = q_pub.lower().strip(), cv_pub.lower().strip()
    if q in c or c in q:
        return True
    return normalize_publisher(q) == normalize_publisher(c)


@dataclass(frozen=True)
class QueryContext:
    """All signals for scoring a candidate issue."""
    issue_num: str
    publisher: str
    year_likelihood: dict
    month_abbr: str
    story_tokens: frozenset
    characters_norm: frozenset
    series_title: str


def enhanced_candidate_scoring(detail_data: dict, query: QueryContext) -> float:
    """Score a candidate issue using all available signals."""
    score = 0.0

    # === HARD CONSTRAINT: Issue number ===
    issue_num_api = str(detail_data.get("issue_number", "")).strip()
    if query.issue_num:
        q_norm = normalize_issue_number(query.issue_num)
        cv_norm = normalize_issue_number(issue_num_api)

        if cv_norm != q_norm:
            try:
                if abs(float(q_norm) - float(cv_norm)) <= 1:
                    score -= 8
                else:
                    return -1e9
            except (ValueError, TypeError):
                return -1e9
        else:
            score += 15

    # === SOFT SIGNALS ===

    # Publisher (+5)
    vol = detail_data.get("volume", {})
    pub_name = (vol.get("publisher") or {}).get("name", "")
    if query.publisher and publisher_matches(query.publisher, pub_name):
        score += 5

    # Series title similarity (+12 max)
    vol_name = vol.get("name", "")
    if query.series_title and vol_name:
        q_tok = series_tokens(query.series_title)
        v_tok = series_tokens(vol_name)
        if q_tok and v_tok:
            coverage = len(q_tok & v_tok) / max(len(q_tok), 1)
            if coverage < 0.34:
                score -= 10
            elif coverage < 0.5:
                score -= 6

        q_special = get_series_special_flags(query.series_title)
        v_special = get_series_special_flags(vol_name)
        if q_special and not v_special:
            score -= 12
        elif not q_special and v_special:
            score -= 4
        elif q_special and v_special:
            score += 2

        if title_matches(vol_name, query.series_title, threshold=0.6):
            score += 12
        elif title_matches(vol_name, query.series_title, threshold=0.4):
            score += 6

        if q_tok and v_tok and q_tok < v_tok:
            extra_tokens = len(v_tok - q_tok)
            if extra_tokens >= 1:
                score -= 3 * extra_tokens

    # Year from price (+8 max)
    cover_date = detail_data.get("cover_date", "")
    if cover_date and query.year_likelihood:
        parts = cover_date.split("-")
        if parts and parts[0].isdigit():
            cand_year = int(parts[0])
            likelihood = query.year_likelihood.get(cand_year, 0.0)

            if likelihood >= 1.0:
                score += 8
            elif likelihood >= 0.7:
                score += 5
            elif likelihood >= 0.4:
                score += 2
            elif likelihood >= 0.2:
                score += 1
            else:
                expected_years = [y for y, l in query.year_likelihood.items() if l >= 0.7]
                if expected_years:
                    min_year, max_year = min(expected_years), max(expected_years)
                    if cand_year > max_year:
                        years_off = cand_year - max_year
                    elif cand_year < min_year:
                        years_off = min_year - cand_year
                    else:
                        years_off = 0

                    if years_off > 30:
                        score -= 25
                    elif years_off > 15:
                        score -= 15
                    elif years_off > 5:
                        score -= 8
                    else:
                        score -= 3

    # Month (+8 max)
    if cover_date and query.month_abbr:
        parts = cover_date.split("-")
        if len(parts) >= 2:
            cand_month = MONTH_NUM_TO_ABBR.get(parts[1])
            if cand_month:
                month_sim = month_similarity_score(query.month_abbr, cand_month)
                if month_sim >= 1.0:
                    score += 8
                elif month_sim >= 0.7:
                    score += 4
                elif month_sim >= 0.4:
                    score += 2

    # Story title (+1 max)
    if query.story_tokens:
        story_blob = f"{detail_data.get('name', '')} {detail_data.get('deck', '')}".lower()
        matched = sum(1 for tok in query.story_tokens if tok in story_blob)
        if query.story_tokens:
            score += (matched / len(query.story_tokens)) * 1

    # Characters (+2 max)
    if query.characters_norm:
        char_credits = detail_data.get("character_credits", [])
        cv_chars = {normalize_char_name(c.get("name", "")) for c in char_credits} - {""}
        if cv_chars:
            overlap = query.characters_norm & cv_chars
            if overlap:
                score += (len(overlap) / len(query.characters_norm | cv_chars)) * 2

    return score


def score_volume(volume: dict, q_series_tokens: set, q_publisher_lower: str, year_likelihood: dict) -> float:
    """Score a volume candidate, heavily favoring main series over mini-series."""
    score = 0.0
    vol_name = volume.get("name", "")
    vol_tokens = series_tokens(vol_name)

    if q_series_tokens and vol_tokens:
        overlap = q_series_tokens & vol_tokens
        if len(overlap) >= 2:
            score += (len(overlap) / len(q_series_tokens | vol_tokens)) * 10
        elif overlap:
            score += 2

    if vol_tokens == q_series_tokens:
        score += 15

    vol_lower = vol_name.lower()
    if ":" in vol_name:
        score -= 10
    if " vs " in vol_lower or " vs. " in vol_lower or " versus " in vol_lower:
        score -= 10
    if " starring " in vol_lower or " starring the " in vol_lower:
        score -= 10
    if " and " in vol_lower and "/" not in vol_name:
        score -= 5

    if len(vol_tokens) > len(q_series_tokens) + 2:
        score -= 3

    count_issues = volume.get("count_of_issues")
    if count_issues and isinstance(count_issues, int):
        if count_issues > 100:
            score += 8
        elif count_issues > 50:
            score += 5
        elif count_issues > 20:
            score += 2
        elif count_issues < 10:
            score -= 2

    pub = (volume.get("publisher") or {}).get("name", "")
    if q_publisher_lower and publisher_matches(q_publisher_lower, pub):
        score += 5

    start_year = volume.get("start_year")
    if isinstance(start_year, str) and start_year.isdigit():
        start_year = int(start_year)
    if year_likelihood and isinstance(start_year, int):
        score += year_likelihood.get(start_year, 0.0) * 5

    return score


def rank_volumes(volume_candidates: list, series_tokens_query: set, publisher: str, year_likelihood: dict) -> list:
    """Rank volume candidates by score."""
    if not volume_candidates:
        return []

    scored = [(vol, score_volume(vol, series_tokens_query, publisher.lower(), year_likelihood))
              for vol in volume_candidates]

    scored.sort(key=lambda x: x[1], reverse=True)
    return [vol for vol, sc in scored if sc > 0]
