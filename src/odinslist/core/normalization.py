"""Price parsing, issue normalization, publisher normalization, row finalization."""
from __future__ import annotations

import re
from typing import Optional


# Cover price in cents -> (earliest_year, latest_year) for standard US comics
PRICE_YEAR_RANGES = {
    10: (1939, 1961),   # $0.10
    12: (1962, 1968),   # $0.12
    15: (1969, 1971),   # $0.15
    20: (1972, 1973),   # $0.20
    25: (1974, 1975),   # $0.25
    30: (1976, 1977),   # $0.30
    35: (1978, 1978),   # $0.35
    40: (1979, 1980),   # $0.40
    50: (1981, 1981),   # $0.50
    60: (1982, 1983),   # $0.60
    75: (1984, 1985),   # $0.75
    125: (1986, 1986),  # $1.25
    150: (1987, 1987),  # $1.50
    175: (1988, 1990),  # $1.75
}

MONTH_NUM_TO_ABBR = {
    "01": "JAN", "02": "FEB", "03": "MAR", "04": "APR", "05": "MAY", "06": "JUN",
    "07": "JUL", "08": "AUG", "09": "SEP", "10": "OCT", "11": "NOV", "12": "DEC",
}

MONTH_ABBR_TO_NUM = {
    "JAN": "01", "JANUARY": "01",
    "FEB": "02", "FEBRUARY": "02",
    "MAR": "03", "MARCH": "03",
    "APR": "04", "APRIL": "04",
    "MAY": "05",
    "JUN": "06", "JUNE": "06",
    "JUL": "07", "JULY": "07",
    "AUG": "08", "AUGUST": "08",
    "SEP": "09", "SEPT": "09", "SEPTEMBER": "09",
    "OCT": "10", "OCTOBER": "10",
    "NOV": "11", "NOVEMBER": "11",
    "DEC": "12", "DECEMBER": "12",
}

STOPWORDS = {"the", "and", "of", "a", "an", "is", "to", "in", "on", "for", "with", "at", "by", "from"}

# Single source of truth: canonical name -> all known variants (lowercase)
PUBLISHER_ALIASES = {
    "Marvel":     {"marvel", "marvel comics", "marvel comics group", "curtis", "curtis magazines"},
    "DC":         {"dc", "dc comics", "d.c. comics"},
    "Image":      {"image", "image comics"},
    "Dark Horse": {"dark horse", "dark horse comics"},
    "IDW":        {"idw", "idw publishing"},
    "Valiant":    {"valiant", "valiant comics"},
    "Archie":     {"archie", "archie comics", "archie comic publications"},
    "Atlas":      {"atlas", "atlas comics", "atlas/seaboard", "seaboard"},
    "Eclipse":    {"eclipse", "eclipse comics", "eclipse enterprises"},
    "Tower":      {"tower", "tower comics", "tower publications"},
    "Kitchen Sink": {"kitchen sink", "kitchen sink press", "kitchen sink enterprises"},
    "Charlton":   {"charlton", "charlton comics", "charlton publications"},
    "AC":         {"ac", "ac comics", "americomics"},
    "Pacific":    {"pacific", "pacific comics"},
    "Awesome":    {"awesome", "awesome comics", "awesome entertainment"},
    "Gold Key":   {"gold key", "gold key comics"},
    "Fictioneer": {"fictioneer", "fictioneer books"},
    "Super":      {"super", "super comics"},
}

# Reverse lookup: any variant -> canonical name
_PUB_TO_CANONICAL: dict[str, str] = {}
for _canonical, _variants in PUBLISHER_ALIASES.items():
    for _v in _variants:
        _PUB_TO_CANONICAL[_v] = _canonical
    _PUB_TO_CANONICAL[_canonical.lower()] = _canonical


def parse_price_to_cents(price: str) -> Optional[int]:
    if not price:
        return None
    s = price.strip().replace(" ", "")
    if s.endswith(("p", "P")):
        return None
    if s.endswith(("¢", "c", "C")):
        try:
            return int(round(float(s[:-1])))
        except (ValueError, TypeError):
            return None
    if s.startswith("$"):
        try:
            return int(round(float(s[1:]) * 100))
        except (ValueError, TypeError):
            return None
    try:
        return int(round(float(s) * 100)) if "." in s else (int(s) if int(s) <= 300 else None)
    except (ValueError, TypeError):
        return None


def get_year_likelihood_from_price(cover_price: str, is_annual: bool) -> dict:
    if is_annual:
        return {}
    cents = parse_price_to_cents(cover_price)
    if not cents or cents not in PRICE_YEAR_RANGES:
        return {}

    base_start, base_end = PRICE_YEAR_RANGES[cents]
    base_start, base_end = max(base_start, 1930), min(base_end, 2030)

    year_scores = {}
    for year in range(1930, 2031):
        if base_start <= year <= base_end:
            year_scores[year] = 1.0
        else:
            distance = min(abs(year - base_start), abs(year - base_end))
            if distance <= 2:
                year_scores[year] = 0.7
            elif distance <= 5:
                year_scores[year] = 0.4
            elif distance <= 10:
                year_scores[year] = 0.2
            else:
                year_scores[year] = 0.0
    return year_scores


def normalize_issue_number(issue_num: str) -> str:
    """
    Normalize issue numbers to handle various formats:
    - "NO. 3" -> "3"
    - "No. 2" -> "2"
    - "BOOK FIVE" -> "5"
    - "Volume 1" -> "1"
    - "#142" -> "142"
    - "½" -> "0.5"
    """
    if not issue_num:
        return ""

    original = issue_num.strip()
    issue_num = original.upper()

    # Handle fractions
    issue_num = issue_num.replace('½', '.5')

    # Strip all leading # symbols
    issue_num = issue_num.lstrip('#').strip()

    # Handle "NO. #3" or "No. #12"
    issue_num = re.sub(r'^NO\.?\s*#\s*', '', issue_num, flags=re.IGNORECASE).strip()

    # Word to number mapping
    word_to_num = {
        'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4',
        'FIVE': '5', 'SIX': '6', 'SEVEN': '7', 'EIGHT': '8', 'NINE': '9',
        'TEN': '10', 'ELEVEN': '11', 'TWELVE': '12', 'THIRTEEN': '13',
        'FOURTEEN': '14', 'FIFTEEN': '15', 'SIXTEEN': '16', 'SEVENTEEN': '17',
        'EIGHTEEN': '18', 'NINETEEN': '19', 'TWENTY': '20'
    }

    for word, num in word_to_num.items():
        if word in issue_num:
            issue_num = issue_num.replace(word, num)
            break

    # Remove common prefixes
    prefixes_to_remove = ['NO.', 'NO ', 'NUM.', 'NUM ', 'NUMBER ', '#', 'VOL.', 'VOL ', 'VOLUME ', 'BOOK ', 'ISSUE ', 'ISS.', 'ISS ']
    for prefix in prefixes_to_remove:
        if issue_num.startswith(prefix):
            issue_num = issue_num[len(prefix):].strip()
            break

    issue_num = issue_num.strip()

    # Preserve letter-prefix issue numbers (e.g., "C-35")
    letter_prefix_match = re.match(r'^([A-Z])-(\d+)$', issue_num)
    if letter_prefix_match:
        return f"{letter_prefix_match.group(1)}-{letter_prefix_match.group(2)}"

    # Try to convert to float/int
    try:
        num = float(issue_num)
        if num > 1500:
            return ""
        return str(int(num)) if num == int(num) else str(num)
    except (ValueError, TypeError):
        match = re.search(r'(\d+\.?\d*)', issue_num)
        if match:
            num_str = match.group(1)
            try:
                num = float(num_str)
                if num > 1500:
                    return ""
                return str(int(num)) if num == int(num) else str(num)
            except (ValueError, TypeError):
                return num_str
        return original.strip()


def normalize_publisher(publisher: str) -> str:
    """Normalize common publisher name variants to canonical short forms."""
    if not publisher:
        return ""
    return _PUB_TO_CANONICAL.get(publisher.strip().lower(), publisher.strip())


def finalize_row(title, issue_num, cover_month, publisher, year):
    """Single normalization choke point before TSV output."""
    if cover_month:
        month_upper = cover_month.strip().upper()
        month_num = MONTH_ABBR_TO_NUM.get(month_upper)
        if month_num:
            cover_month = MONTH_NUM_TO_ABBR[month_num]
        elif len(month_upper) != 3:
            cover_month = ""

    issue_num = normalize_issue_number(issue_num) if issue_num else ""
    publisher = normalize_publisher(publisher) if publisher else ""
    title = title.strip() if title else ""

    return title, issue_num, cover_month, publisher, year
