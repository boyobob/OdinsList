from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from openai import OpenAI
from PIL import Image, ImageOps
import argparse
import base64
import csv
import glob as globmod
import io
import os
import requests
from curl_cffi import requests as curl_requests
import json
import re
import time
import random
import sqlite3
from dataclasses import dataclass
from typing import Optional

# ------------------------------
# Configuration
# ------------------------------
@dataclass
class Config:
    vlm_base_url: str = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1")
    model_name: str = os.environ.get("VLM_MODEL", "")
    client: object = None  # OpenAI client, initialized in main()
    batch_mode: bool = False
    box_name: str | None = None
    base_dir: str | None = None
    cover_cache_dir: str = "/tmp/comicvine_covers"
    image_dir: str | None = None
    outfile: str | None = None
    logfile: str | None = None
    log_handle: object = None
    gcd_db_path: str | None = None
    use_gcd: bool = True
    comicvine_api_key: str = os.environ.get("COMICVINE_API_KEY", "")
    comicvine_base_url: str = "https://comicvine.gamespot.com/api"
    use_comicvine: bool = True
    use_resume: bool = False
    use_visual: bool = True

cfg = Config()

def log(msg: str):
    """Write message to both console and log file."""
    print(msg)
    if cfg.log_handle:
        cfg.log_handle.write(msg + "\n")
        cfg.log_handle.flush()

class BrowserSession:
    """Rotating curl_cffi session with browser TLS fingerprint impersonation."""
    IMPERSONATIONS = ["chrome120", "chrome124", "chrome131", "safari17_0", "edge101"]
    MAX_REQUESTS = 20

    def __init__(self):
        self._session = None
        self._index = 0
        self._request_count = 0
        self._primed = False

    def get(self, url: str, **kwargs):
        """Make a GET request, auto-rotating and priming the session as needed."""
        self._maybe_rotate()
        self._maybe_prime()
        self._request_count += 1
        return self._session.get(url, **kwargs)

    def reset(self):
        """Reset the session (useful if Cloudflare starts blocking)."""
        if self._session:
            self._session.close()
        self._session = None
        self._primed = False
        self._request_count = 0
        log("[INFO] Image download session reset")

    def _maybe_rotate(self):
        if self._session is not None and self._request_count < self.MAX_REQUESTS:
            return
        if self._request_count >= self.MAX_REQUESTS:
            log(f"[INFO] Rotating session after {self._request_count} requests")
        impersonate = self.IMPERSONATIONS[self._index % len(self.IMPERSONATIONS)]
        self._index += 1
        self._request_count = 0
        self._session = curl_requests.Session(impersonate=impersonate)
        log(f"[INFO] Created new curl_cffi session impersonating {impersonate}")

    def _maybe_prime(self):
        if self._primed:
            return
        try:
            response = self._session.get("https://comicvine.gamespot.com/", timeout=15)
            if response.status_code == 200:
                log("[INFO] Session primed successfully")
            else:
                log(f"[WARN] Session prime returned {response.status_code}")
        except Exception as e:
            log(f"[WARN] Session prime failed: {e}")
        self._primed = True

browser_session = BrowserSession()

API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

VLM_FIELDS = {
    "canonical_title": "string or null",
    "raw_title_text": "string or null",
    "issue_number": "string or null",
    "cover_month_full": "string or null",
    "cover_month": "string or null",
    "publisher_normalized": "string or null",
    "publisher_raw": "string or null",
    "series_descriptor": "string or null",
    "is_annual_or_special": "bool or null",
    "cover_price": "string or null",
    "country_or_region": "string or null",
    "main_characters": "[string, ...]",
    "story_title_text": "string or null",
}

_schema_block = json.dumps({k: v for k, v in VLM_FIELDS.items()}, indent=2)

USER_PROMPT = f"""You are an expert comic-book archivist. Analyze this comic-book cover image and output a single JSON object on one line.

Infer as much as you can from the cover itself. Use this exact schema:
{_schema_block}

Rules:
- If you are not sure about a value, use null.
- For publisher_normalized, map "MARVEL COMICS GROUP" to "Marvel", "DC COMICS" to "DC".
- Do NOT guess any year.
- Return ONLY raw JSON with double-quoted keys. No markdown, no explanations."""

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

# Single source of truth for month name/abbreviation to number mapping
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
    "Marvel":     {"marvel", "marvel comics", "marvel comics group"},
    "DC":         {"dc", "dc comics", "d.c. comics"},
    "Image":      {"image", "image comics"},
    "Dark Horse": {"dark horse", "dark horse comics"},
    "IDW":        {"idw", "idw publishing"},
    "Valiant":    {"valiant", "valiant comics"},
    "Archie":     {"archie", "archie comics", "archie comic publications"},
}

# Reverse lookup: any variant -> canonical name
_PUB_TO_CANONICAL = {}
for _canonical, _variants in PUBLISHER_ALIASES.items():
    for _v in _variants:
        _PUB_TO_CANONICAL[_v] = _canonical
    _PUB_TO_CANONICAL[_canonical.lower()] = _canonical

def setup_cover_cache():
    """Ensure cover cache directory exists."""
    os.makedirs(cfg.cover_cache_dir, exist_ok=True)

def load_image_b64(path: str) -> str:
    img = Image.open(path)

    # Auto-rotate based on EXIF orientation data
    # This fixes images that were taken rotated (90°, 180°, 270°)
    img = ImageOps.exif_transpose(img)

    # Convert to RGB (remove alpha channel if present)
    img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def safe_json_loads(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON between { and }
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass

        # Try to fix truncated JSON (unclosed arrays/strings)
        try:
            # Find last complete field before truncation
            json_str = text[start:end+1] if (start != -1 and end != -1) else text

            # If it ends with an incomplete array, try to close it
            if json_str.count('[') > json_str.count(']'):
                # Close any unclosed strings first
                if json_str.count('"') % 2 == 1:
                    json_str += '"'
                # Close unclosed arrays
                json_str += ']' * (json_str.count('[') - json_str.count(']'))
                # Close object if needed
                if json_str.count('{') > json_str.count('}'):
                    json_str += '}'
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
    return None

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
    - "NO. 3" → "3"
    - "No. 2" → "2"
    - "BOOK FIVE" → "5"
    - "Volume 1" → "1"
    - "#142" → "142"
    - "½" → "0.5"
    """
    if not issue_num:
        return ""

    original = issue_num.strip()
    issue_num = original.upper()

    # Handle fractions
    issue_num = issue_num.replace('½', '.5')

    # Strip all leading # symbols (handles OCR errors like "##1" → "1")
    issue_num = issue_num.lstrip('#').strip()

    # Word to number mapping for common written-out issue numbers
    word_to_num = {
        'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4',
        'FIVE': '5', 'SIX': '6', 'SEVEN': '7', 'EIGHT': '8', 'NINE': '9',
        'TEN': '10', 'ELEVEN': '11', 'TWELVE': '12', 'THIRTEEN': '13',
        'FOURTEEN': '14', 'FIFTEEN': '15', 'SIXTEEN': '16', 'SEVENTEEN': '17',
        'EIGHTEEN': '18', 'NINETEEN': '19', 'TWENTY': '20'
    }

    # Check for written-out numbers (e.g., "BOOK FIVE", "VOLUME THREE")
    for word, num in word_to_num.items():
        if word in issue_num:
            issue_num = issue_num.replace(word, num)
            break

    # Remove common prefixes that don't affect the number
    # "NO. 3" → "3", "No. 2" → "2", "#142" → "142", "Vol. 1" → "1"
    prefixes_to_remove = ['NO.', 'NO ', 'NUM.', 'NUM ', 'NUMBER ', '#', 'VOL.', 'VOL ', 'VOLUME ', 'BOOK ', 'ISSUE ', 'ISS.', 'ISS ']
    for prefix in prefixes_to_remove:
        if issue_num.startswith(prefix):
            issue_num = issue_num[len(prefix):].strip()
            break

    # Try to extract just the numeric part if there's other text
    # This handles cases like "3 MAR" → "3"
    issue_num = issue_num.strip()

    # Try to convert to float/int
    try:
        num = float(issue_num)
        return str(int(num)) if num == int(num) else str(num)
    except (ValueError, TypeError):
        # If can't convert directly, try to extract first number
        match = re.search(r'(\d+\.?\d*)', issue_num)
        if match:
            num_str = match.group(1)
            try:
                num = float(num_str)
                return str(int(num)) if num == int(num) else str(num)
            except (ValueError, TypeError):
                return num_str
        # If all else fails, return original (stripped)
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


# ------------------------------
# Grand Comics Database (GCD) - Local Search
# ------------------------------
def get_gcd_characters(conn, issue_ids: list) -> dict:
    """
    Fetch character credits for a list of issue IDs from GCD database.
    Returns dict mapping issue_id → list of character names.
    Uses gcd_story → gcd_story_character → gcd_character joins.
    """
    if not issue_ids:
        return {}

    try:
        cursor = conn.cursor()

        # Build query for multiple issue IDs
        placeholders = ','.join('?' * len(issue_ids))
        query = f"""
        SELECT DISTINCT
            st.issue_id,
            c.name as character_name
        FROM gcd_story st
        JOIN gcd_story_character sc ON sc.story_id = st.id
        JOIN gcd_character c ON c.id = sc.character_id
        WHERE st.issue_id IN ({placeholders})
        AND sc.deleted = 0
        AND c.deleted = 0
        ORDER BY st.issue_id, c.name
        """

        cursor.execute(query, issue_ids)

        # Group characters by issue_id
        characters_by_issue = {}
        for row in cursor.fetchall():
            issue_id = row[0]
            char_name = row[1]
            if issue_id not in characters_by_issue:
                characters_by_issue[issue_id] = []
            characters_by_issue[issue_id].append(char_name)

        return characters_by_issue

    except Exception as e:
        log(f"[GCD] Character fetch error: {e}")
        return {}

def _get_publisher_variants(publisher: str) -> list:
    """Get normalized publisher name variants for database matching."""
    if not publisher:
        return []

    canonical = normalize_publisher(publisher)
    if canonical in PUBLISHER_ALIASES:
        # Return capitalized forms of all variants for case-insensitive SQL matching
        return [canonical] + [v.title() if v != v.upper() else v for v in PUBLISHER_ALIASES[canonical]]
    return [publisher]

def _score_and_filter_by_title(results: dict, series_title: str, threshold: float = 0.4, log_label: str = "") -> dict:
    """
    Score results by title similarity and filter out poor matches.
    Returns filtered dict with title_similarity scores added to each result.
    """
    if not results or not series_title:
        return results

    title_variants = generate_title_variants(series_title)

    for issue_id, result in results.items():
        db_title = result['series_name']
        best_similarity = 0.0

        # Calculate Jaccard similarity against all title variants
        for variant in title_variants:
            if variant:
                db_tokens = series_tokens(db_title)
                var_tokens = series_tokens(variant)
                if db_tokens and var_tokens:
                    jaccard = len(db_tokens & var_tokens) / len(db_tokens | var_tokens)
                    best_similarity = max(best_similarity, jaccard)

        result['title_similarity'] = best_similarity

    # Filter out results below threshold
    filtered_results = {k: v for k, v in results.items() if v.get('title_similarity', 0) >= threshold}

    if filtered_results:
        log(f"[GCD] {len(filtered_results)} candidates passed title filter{log_label} (>= {threshold})")

        # Sort by similarity and log top matches
        sorted_results = sorted(filtered_results.values(), key=lambda x: x.get('title_similarity', 0), reverse=True)
        log(f"[GCD] Top matches by title similarity{log_label}:")
        for r in sorted_results[:5]:
            sim = r.get('title_similarity', 0)
            log(f"  [{sim:.2f}] {r['series_name']} #{r['issue_number']} ({r['publisher_name']})")

        # Convert back to dict keyed by issue_id
        return {r['issue_id']: r for r in sorted_results}
    else:
        log(f"[GCD] No candidates passed title filter{log_label} (>= {threshold}), rejecting all")
        return {}

def _build_gcd_filters(issue_num, publisher, year_range, cover_month=None):
    """Build WHERE clause parts and params for GCD search."""
    parts, params = [], []

    if issue_num:
        normalized = normalize_issue_number(issue_num)
        parts.append("(i.number = ? OR i.number = ?)")
        params.extend([issue_num, normalized])

    if publisher:
        pub_variants = _get_publisher_variants(publisher)
        if pub_variants:
            pub_conditions = " OR ".join(["LOWER(p.name) = LOWER(?)" for _ in pub_variants])
            parts.append(f"({pub_conditions})")
            params.extend(pub_variants)

    if year_range:
        min_year, max_year = year_range
        parts.append("(s.year_began <= ? AND (s.year_ended IS NULL OR s.year_ended >= ?))")
        params.extend([max_year, min_year])

    if cover_month:
        month_num = MONTH_ABBR_TO_NUM.get(cover_month.upper()[:3])
        if month_num:
            parts.append("substr(i.key_date, 6, 2) = ?")
            params.append(month_num)

    return parts, params

def search_gcd(series_title: str, issue_num: str, publisher: str, year_range: tuple = None, cover_month: str = None) -> list:
    """
    Search the local Grand Comics Database (SQLite) for matching issues.
    Returns list of matching issues with full metadata, sorted by title similarity.

    SEARCH STRATEGY (based on discrimination power analysis):
    1. PRIMARY FILTERS: Issue# + Publisher + Year Range + Month
       - These filters narrow down to a small candidate set
       - ~83% of the time this gives a unique match
    2. SCORING: Title similarity used to RANK results, not filter
       - Avoids missing matches due to title variations
       - "Sgt. Fury" vs "Sgt. Fury and His Howling Commandos"
    """
    if not cfg.use_gcd or not os.path.exists(cfg.gcd_db_path):
        return []

    try:
        conn = sqlite3.connect(cfg.gcd_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        all_results = {}

        base_query = """
        SELECT
            i.id as issue_id,
            i.number as issue_number,
            i.key_date,
            i.publication_date,
            i.price,
            i.on_sale_date,
            s.id as series_id,
            s.name as series_name,
            s.year_began,
            s.year_ended,
            s.issue_count,
            p.id as publisher_id,
            p.name as publisher_name
        FROM gcd_issue i
        JOIN gcd_series s ON i.series_id = s.id
        JOIN gcd_publisher p ON s.publisher_id = p.id
        WHERE i.deleted = 0 AND s.deleted = 0 AND p.deleted = 0
        """

        if not issue_num:
            conn.close()
            return []

        # First attempt: with month filter
        query_parts, params = _build_gcd_filters(issue_num, publisher, year_range, cover_month)

        if query_parts:
            full_query = base_query + " AND " + " AND ".join(query_parts) + " LIMIT 100"
        else:
            full_query = base_query + " LIMIT 100"

        cursor.execute(full_query, params)
        for row in cursor.fetchall():
            issue_id = row['issue_id']
            if issue_id not in all_results:
                all_results[issue_id] = dict(row)

        # SCORING & FILTERING: Calculate title similarity and filter poor matches
        all_results = _score_and_filter_by_title(all_results, series_title)

        # Fallback: without month constraint
        if len(all_results) == 0 and cover_month:
            log(f"[GCD] No matches with month filter, trying without month...")
            query_parts, params = _build_gcd_filters(issue_num, publisher, year_range)

            if query_parts:
                full_query = base_query + " AND " + " AND ".join(query_parts) + " LIMIT 100"
                cursor.execute(full_query, params)

                for row in cursor.fetchall():
                    issue_id = row['issue_id']
                    if issue_id not in all_results:
                        all_results[issue_id] = dict(row)

                # Score and filter by title similarity
                all_results = _score_and_filter_by_title(all_results, series_title, log_label=" (no month)")

        # Fetch character credits for all found issues
        if all_results:
            issue_ids = [result['issue_id'] for result in all_results.values()]
            characters_by_issue = get_gcd_characters(conn, issue_ids)

            for issue_id, result in all_results.items():
                result['characters'] = characters_by_issue.get(issue_id, [])

        conn.close()

        # Return sorted by title similarity
        return sorted(all_results.values(), key=lambda x: x.get('title_similarity', 0), reverse=True)

    except Exception as e:
        log(f"[GCD ERROR] {e}")
        import traceback
        traceback.print_exc()
        return []

def gcd_to_comicvine_format(gcd_result: dict) -> dict:
    """Convert GCD database result to ComicVine-like format for compatibility."""
    # Convert character names list to ComicVine-like format
    character_credits = []
    if 'characters' in gcd_result and gcd_result['characters']:
        character_credits = [{"name": char_name} for char_name in gcd_result['characters']]

    return {
        "id": f"gcd_{gcd_result['issue_id']}",  # Prefix to distinguish from ComicVine IDs
        "issue_number": gcd_result['issue_number'],
        "cover_date": gcd_result['key_date'],  # YYYY-MM-DD format
        "volume": {
            "name": gcd_result['series_name'],
            "id": f"gcd_{gcd_result['series_id']}",
            "publisher": {
                "name": gcd_result['publisher_name'],
                "id": f"gcd_{gcd_result['publisher_id']}"
            },
            "start_year": gcd_result['year_began'],
        },
        "name": None,  # GCD doesn't always have issue titles
        "deck": None,
        "image": None,  # GCD doesn't have cover images
        "api_detail_url": None,  # No detail URL for local DB
        "character_credits": character_credits,  # Character appearances from GCD
        "source": "GCD",  # Mark as coming from GCD
    }

def series_tokens(s: str) -> set:
    if not s:
        return set()
    tokens = re.split(r"[^a-z0-9]+", s.lower())
    return {t for t in tokens if t and t not in STOPWORDS}

def extract_story_tokens(story: str) -> set:
    if not story:
        return set()
    tokens = re.split(r"[^a-z0-9]+", story.lower())
    return {t for t in tokens if len(t) >= 4 and t not in STOPWORDS}

def normalize_char_name(name: str) -> str:
    if not name:
        return ""
    s = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", name.lower())).strip()
    return s[4:] if s.startswith("the ") else s

def month_similarity_score(q_month: str, cv_month: str) -> float:
    if not q_month or not cv_month:
        return 0.0
    if q_month == cv_month:
        return 1.0
    try:
        # Convert month abbreviations to 0-indexed positions using MONTH_ABBR_TO_NUM
        q_num = MONTH_ABBR_TO_NUM.get(q_month.upper())
        cv_num = MONTH_ABBR_TO_NUM.get(cv_month.upper())
        if not q_num or not cv_num:
            return 0.0
        q_idx = int(q_num) - 1  # Convert "01"-"12" to 0-11
        cv_idx = int(cv_num) - 1
        diff = abs(q_idx - cv_idx)
        diff = min(diff, 12 - diff)  # Handle wrap-around (Dec-Jan)
        return {0: 1.0, 1: 0.7, 2: 0.4}.get(diff, 0.0)
    except (ValueError, TypeError):
        return 0.0

def download_cover(url: str, output_path: str, max_retries: int = 3) -> bool:
    """Download a cover image using curl_cffi (bypasses TLS fingerprinting).

    On any HTTP error, resets session and retries up to max_retries times.
    """
    for attempt in range(max_retries):
        # Random delay between requests to appear more human (0.5-1.5 seconds)
        delay = random.uniform(0.5, 1.5)
        time.sleep(delay)

        try:
            response = browser_session.get(url, timeout=15)

            if response.status_code >= 400:
                log(f"[DOWNLOAD] HTTP {response.status_code} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    log(f"[DOWNLOAD] Resetting session and retrying...")
                    browser_session.reset()
                    time.sleep(2)  # Brief cooldown before retry
                    continue
                else:
                    log(f"[DOWNLOAD] Max retries reached, giving up on {url[:60]}...")
                    return False

            # Success path
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                log(f"[DOWNLOAD] Bad content-type: {content_type}")
                return False

            # Save the image
            with open(output_path, 'wb') as f:
                f.write(response.content)

            # Verify file size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                if attempt > 0:
                    log(f"[DOWNLOAD] Success on retry {attempt + 1}")
                return True

            if os.path.exists(output_path):
                os.remove(output_path)
            return False

        except Exception as e:
            log(f"[DOWNLOAD] Exception: {type(e).__name__}: {e} (attempt {attempt + 1}/{max_retries})")
            if os.path.exists(output_path):
                os.remove(output_path)
            if attempt < max_retries - 1:
                browser_session.reset()
                time.sleep(2)
                continue
            return False

    return False


def get_cached_cover(issue_id: int, image_data: dict) -> Optional[str]:
    """Download and cache a cover image from ComicVine, return local path."""
    if not image_data or not issue_id:
        return None

    cache_path = os.path.join(cfg.cover_cache_dir, f"{issue_id}.jpg")

    # Return cached if exists
    if os.path.exists(cache_path):
        return cache_path

    # Try different image sizes - prefer smaller for speed
    for key in ["small_url", "medium_url", "thumb_url", "original_url"]:
        url = image_data.get(key)
        if url and download_cover(url, cache_path):
            return cache_path

    return None

def compare_covers_with_qwen(original_path: str, candidate_path: str) -> str:
    """Compare two comic covers using Qwen vision model. Returns: SAME, VARIANT, or DIFFERENT."""
    if not original_path or not candidate_path:
        return "DIFFERENT"

    try:
        # Load both images as base64
        original_b64 = load_image_b64(original_path)
        candidate_b64 = load_image_b64(candidate_path)

        prompt = """Compare these two comic book covers and determine if they are the same cover.

Answer with ONLY ONE of these words:
- SAME: These are the same cover (identical artwork, accounting for scan quality, age, or color differences)
- VARIANT: Same issue number but different variant cover artwork
- DIFFERENT: Completely different issues

Consider: character positions, artwork composition, logo style and placement, background elements, overall layout.
Ignore: differences in image quality, slight color variations, wear and tear, or scanning artifacts.

Your answer (one word only):"""

        resp = cfg.client.chat.completions.create(
            model=cfg.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{candidate_b64}"}},
                ],
            }],
            max_tokens=10,
            stream=False,
        )

        result = resp.choices[0].message.content.strip().upper()

        # Extract the classification
        if "SAME" in result:
            return "SAME"
        elif "VARIANT" in result:
            return "VARIANT"
        else:
            return "DIFFERENT"

    except Exception as e:
        log(f"[WARN] Visual comparison failed: {e}")
        return "DIFFERENT"

def _try_visual_match(candidate_detail: dict, original_img_path: str, current_score: float) -> tuple:
    """
    Attempt visual comparison for a single candidate.

    Returns:
        (is_match, adjusted_score, visual_result)
        - is_match: True if SAME (should early exit)
        - adjusted_score: score with visual bonus applied
        - visual_result: "SAME", "VARIANT", "DIFFERENT", or None if no comparison done
    """
    if not cfg.use_visual:
        return False, current_score, None

    # Get candidate info for logging
    vol_name = candidate_detail.get('volume', {}).get('name', 'Unknown')
    issue_no = candidate_detail.get('issue_number', '?')

    if not original_img_path:
        log(f"[VISUAL] Skipping {vol_name} #{issue_no}: no original image path")
        return False, current_score, None

    issue_id = candidate_detail.get("id")
    image_data = candidate_detail.get("image")
    if not image_data:
        log(f"[VISUAL] Skipping {vol_name} #{issue_no}: no cover image data from ComicVine")
        return False, current_score, None

    cover_path = get_cached_cover(issue_id, image_data)
    if not cover_path:
        log(f"[VISUAL] Skipping {vol_name} #{issue_no}: failed to download cover")
        return False, current_score, None

    # Log that visual comparison is starting
    log(f"[VISUAL] Comparing covers: {vol_name} #{issue_no}...")

    visual_result = compare_covers_with_qwen(original_img_path, cover_path)

    if visual_result == "SAME":
        return True, current_score + 25, visual_result
    elif visual_result == "VARIANT":
        return False, current_score + 10, visual_result  # Good match, but keep looking for exact
    else:  # DIFFERENT
        return False, current_score - 10, visual_result

def comicvine_get(path: str, params: dict, max_retries: int = 5):
    """Make a request to ComicVine API with retry logic for rate limiting and network errors."""
    url = f"{cfg.comicvine_base_url}/{path.lstrip('/')}"

    for attempt in range(max_retries):
        try:
            # Base delay between requests (increased to reduce rate limiting)
            time.sleep(1.5)

            resp = requests.get(url, params=params, headers=API_HEADERS, timeout=10)

            # Handle specific status codes
            if resp.status_code == 403:
                log("[ERROR] ComicVine 403 - Access forbidden")
                return None

            if resp.status_code == 420:
                # Rate limiting - wait longer before retry
                wait_time = min(2 ** attempt * 3, 90)  # Aggressive backoff, max 90s
                log(f"[WARN] Rate limited (420). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)

                # If we've hit rate limits multiple times, add extra cooldown
                if attempt >= 2:
                    extra_cooldown = 10
                    log(f"[INFO] Adding {extra_cooldown}s cooldown after repeated rate limiting...")
                    time.sleep(extra_cooldown)
                continue

            # Raise for other HTTP errors (4xx, 5xx)
            resp.raise_for_status()

            # Success - return the JSON response
            return resp.json()

        except requests.exceptions.Timeout:
            wait_time = min(2 ** attempt, 30)
            log(f"[WARN] Request timeout. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                log("[ERROR] Max retries reached after timeout")
                return None

        except requests.exceptions.ConnectionError as e:
            wait_time = min(2 ** attempt, 30)
            log(f"[WARN] Connection error: {e}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                log("[ERROR] Max retries reached after connection error")
                return None

        except requests.exceptions.HTTPError as e:
            # For other HTTP errors that aren't 403 or 420
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)
                log(f"[WARN] HTTP error {e}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                log(f"[ERROR] HTTP error after {max_retries} retries: {e}")
                return None  # Don't crash, just skip this API call

        except Exception as e:
            log(f"[ERROR] Unexpected error in API call: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)
                log(f"[WARN] Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                log(f"[ERROR] Max retries reached, skipping this API call")
                return None  # Don't crash, just skip

    return None

def is_valid_series_descriptor(descriptor: str) -> bool:
    """Check if descriptor is actually a series qualifier, not story text."""
    if not descriptor:
        return False

    descriptor_lower = descriptor.lower()

    # Valid descriptors - including anthology series
    valid_keywords = ["giant-size", "giant size", "king-size", "king size",
                      "annual", "special", "giant", "super", "super-size",
                      # Anthology series (these ARE the actual series name!)
                      "team-up", "two-in-one", "two in one", "presents",
                      "comics presents", "triple action", "marvel premiere",
                      "showcase", "brave and the bold"]
    if any(kw in descriptor_lower for kw in valid_keywords):
        return True

    # Also valid if it's a known anthology series name
    if is_anthology_series(descriptor):
        return True

    # Reject if it looks like story text (too long, has exclamation marks, etc.)
    if len(descriptor) > 40:  # Increased to allow "Marvel Team-Up" etc.
        return False
    if descriptor.count("!") >= 2:
        return False
    if any(word in descriptor_lower for word in ["death", "tale", "story", "epic", "saga", "all-new"]):
        return False

    return False

def is_anthology_series(descriptor: str) -> bool:
    """Check if descriptor is actually an anthology series name (should replace canonical_title)."""
    if not descriptor:
        return False

    descriptor_lower = descriptor.lower()

    # Anthology series where the descriptor IS the actual series name
    # NOTE: "presents" removed - it's a subtitle format, not a series name
    # "Marvel Two-In-One Presents" is NOT the same as "DC Comics Presents"
    anthology_keywords = ["team-up","team up","two-in-one","two in one","three-in-one","dc comics presents","dc comics presents annual","marvel team-up","marvel team up","marvel team-up annual","marvel two-in-one","marvel two in one","showcase","1st issue special","first issue special","dc special","dc special series","secret origins","the brave and the bold","brave and the bold","batman the brave and the bold","strange tales","tales to astonish","tales of suspense","journey into mystery","journey into mystery annual","adventure comics","action comics","detective comics","adventure into fear","astonishing tales","marvel premiere","marvel spotlight","marvel feature","marvel presents","marvel comics presents","marvel fanfare","marvel preview","marvel super action","marvel super-heroes","marvel super heroes","marvel super special","special marvel edition","marvel collectors' item classics","marvel tales","fantasy masterpieces","marvel chillers","monsters on the prowl","where monsters dwell","where creatures roam","vault of evil","chamber of darkness","chamber of chills","dead of night","supernatural thrillers","tower of shadows","worlds unknown","unknown worlds of science fiction","unknown worlds","creatures on the loose","fear","frankenstein","house of mystery","house of secrets","ghosts","the unexpected","unexpected","secrets of haunted house","the witching hour","weird war tales","star spangled war stories","our army at war","weird western tales","weird mystery tales","time warp","the superman family","creepy","eerie","vampirella","twisted tales","alien worlds","epic illustrated","amazing adventures","amazing adult fantasy","amazing fantasy","all-star comics","sensation comics","whiz comics","planet comics","mystery in space","strange adventures","tales from the crypt","the haunt of fear","the vault of horror","dark horse presents","negative burn","2000 ad","heavy metal","solo","a1","flight","raw","marvel super-teams","world’s finest comics","brave and the bold special","venture","a-next","star spangled comics","bizarre adventures","ghostly tales","chilling adventures in sorcery", "what if", "what if?", "what if?...", "what if..."]

    return any(kw in descriptor_lower for kw in anthology_keywords)

# ------------------------------
# Title & Publisher Matching
# ------------------------------
def generate_title_variants(series_title: str) -> list:
    """Generate title variants for searching."""
    if not series_title:
        return []

    variants = [series_title]

    # The/The removal
    if series_title.startswith("The "):
        variants.append(series_title[4:])
    else:
        variants.append(f"The {series_title}")

    # Drop common adjective prefixes: "The Mighty Thor" -> "Thor"
    _DROPPABLE_ADJECTIVES = ["Mighty", "Invincible", "Uncanny", "Amazing", "Spectacular", "Incredible"]
    for adj in _DROPPABLE_ADJECTIVES:
        if adj in series_title:
            variants.append(series_title.replace(f"The {adj} ", "").replace(f"{adj} ", ""))

    # Remove character name prefix before comma
    # "Luke Cage, Hero for Hire" → "Hero for Hire"
    # "Moon Knight, Fist of Khonshu" → "Fist of Khonshu"
    if ", " in series_title:
        after_comma = series_title.split(", ", 1)[1]
        variants.append(after_comma)
        # Also try without comma: "Luke Cage Hero for Hire"
        variants.append(series_title.replace(", ", " "))

    # Remove common suffixes that Qwen sometimes incorrectly includes
    # "Captain America Comics" → "Captain America"
    # "Tales of Suspense Magazine" → "Tales of Suspense"
    for suffix in [" Comics", " Comic", " Magazine", " Book"]:
        if series_title.endswith(suffix):
            variants.append(series_title[:-len(suffix)].strip())

    # Extract main title before "and his/her/the/their" patterns
    # "Sgt. Fury and His Howling Commandos" → "Sgt. Fury"
    # "Batman and the Outsiders" → "Batman"
    # "Power Man and Iron Fist" → "Power Man"
    and_patterns = [
        r'^(.+?)\s+and\s+(?:his|her|the|their)\s+',  # "X and his/her/the/their Y"
        r'^(.+?)\s+and\s+',  # "X and Y"
    ]
    for pattern in and_patterns:
        match = re.match(pattern, series_title, re.IGNORECASE)
        if match:
            main_title = match.group(1).strip()
            if len(main_title) >= 3:  # Avoid too-short variants
                variants.append(main_title)

    # Hyphen variations
    if "-" in series_title:
        variants.append(series_title.replace("-", ""))
        variants.append(series_title.replace("-", " "))

    # Deduplicate
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

    # If there's substantial overlap, it's a match
    jaccard = len(overlap) / len(cv_tok | q_tok)
    if jaccard >= threshold:
        return True

    # Special case: if the main word matches (e.g., "Thor" in both), consider it
    # This handles "The Mighty Thor" vs "Thor"
    cv_main_words = {t for t in cv_tok if len(t) >= 4}
    q_main_words = {t for t in q_tok if len(t) >= 4}

    if cv_main_words and q_main_words:
        main_overlap = cv_main_words & q_main_words
        # Use the same threshold parameter here too
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

# ------------------------------
# Search Functions
# ------------------------------
def _filter_issue_candidates(
    issues: list,
    issue_num: str,
    title_variants: list,
    publisher: str,
    title_threshold: float = 0.5,
    month_num: str | None = None,
) -> dict:
    """Filter ComicVine issue results by number, title, publisher, and optionally month."""
    normalized_issue = normalize_issue_number(issue_num)
    candidates = {}

    for issue in issues:
        vol = issue.get("volume", {})
        vol_name = vol.get("name", "")
        issue_no = str(issue.get("issue_number", "")).strip()

        if issue_num and issue_no not in [issue_num, normalized_issue]:
            continue

        if title_variants:
            if ":" in vol_name and vol_name not in title_variants:
                continue
            if not any(title_matches(vol_name, tv, threshold=title_threshold) for tv in title_variants):
                continue

        if publisher:
            pub_name = (vol.get("publisher") or {}).get("name", "")
            if pub_name and not publisher_matches(publisher, pub_name):
                continue

        if month_num:
            cover_date = issue.get("cover_date", "")
            if cover_date:
                parts = cover_date.split("-")
                if len(parts) >= 2:
                    issue_month = parts[1]
                    if issue_month != month_num:
                        try:
                            if abs(int(issue_month) - int(month_num)) > 1:
                                continue
                        except ValueError:
                            continue

        issue_id = issue.get("id")
        if issue_id and issue_id not in candidates:
            candidates[issue_id] = issue

    return candidates

def search_issues_by_number_and_month(issue_num: str, month_abbr: str, year_likelihood: dict,
                                       series_title: str = None, publisher: str = None) -> list:
    """
    Strategy 2: Use /search endpoint for simplicity.

    Simple Approach:
    1. Search for: "{title} {issue_number}" using /search endpoint
    2. Filter results by month and publisher locally
    3. ONE API call, let ComicVine's search do the work
    """
    if not issue_num:
        return []

    # Build search query with exact phrase matching
    if series_title:
        # Use quotes for exact title matching to avoid fuzzy matches
        # e.g., "Astonishing Tales" #1 instead of Astonishing Tales #1
        search_queries = [
            f'"{series_title}" {issue_num}',
            f'"{series_title}" #{issue_num}',
        ]
    else:
        search_queries = [f"#{issue_num}"]

    log(f"[Strategy 2] Search API: '{series_title}' #{issue_num} {month_abbr}")

    candidates = {}
    month_num = MONTH_ABBR_TO_NUM.get(month_abbr.upper()) if month_abbr else None
    title_variants = generate_title_variants(series_title) if series_title else []

    # Use /search endpoint - ONE simple call
    for query in search_queries:
        params = {
            "api_key": cfg.comicvine_api_key,
            "format": "json",
            "query": query,
            "resources": "issue",  # Only search issues
            "limit": 30,
            "field_list": "id,volume,issue_number,cover_date,api_detail_url,name,deck,image",
        }

        log(f"  Searching: \"{query}\"")
        data = comicvine_get("search/", params)
        if not data or data.get("error") != "OK":
            continue

        results = data.get("results", [])
        if not results:
            log(f"    No results")
            continue

        log(f"    Found {len(results)} results")

        # Use stricter threshold (0.75) for /search since it returns very broad results
        filtered = _filter_issue_candidates(
            results, issue_num, title_variants, publisher,
            title_threshold=0.75, month_num=month_num,
        )
        for issue_id, issue in filtered.items():
            if issue_id not in candidates:
                candidates[issue_id] = issue
                vol_name = issue.get("volume", {}).get("name", "")
                issue_no = str(issue.get("issue_number", "")).strip()
                cover_date = issue.get("cover_date", "")
                log(f"      ✓ {vol_name} #{issue_no} ({cover_date})")

        # If we found good matches, no need to try other queries
        if len(candidates) >= 3:
            break

    log(f"[Strategy 2] Found {len(candidates)} candidates")
    return list(candidates.values())

def search_issues_directly(series_title: str, issue_num: str, publisher: str) -> list:
    """Strategy 3: Search issues endpoint directly - last resort when other strategies fail."""
    if not issue_num:
        return []

    candidates = {}
    title_variants = generate_title_variants(series_title) if series_title else []
    normalized_issue = normalize_issue_number(issue_num)

    # Search by issue number
    for attempt_num in [issue_num, normalized_issue]:
        params = {
            "api_key": cfg.comicvine_api_key,
            "format": "json",
            "filter": f"issue_number:{attempt_num}",
            "limit": 100,
            "field_list": "id,volume,issue_number,cover_date,api_detail_url,name,deck,image",
        }

        data = comicvine_get("issues/", params)
        if not data or data.get("error") != "OK":
            continue

        filtered = _filter_issue_candidates(
            data.get("results", []), issue_num, title_variants, publisher,
        )
        candidates.update(filtered)

    return list(candidates.values())

def fetch_volume_candidates(series_title: str, publisher: str) -> list:
    """Fetch volume candidates from ComicVine."""
    if not series_title:
        return []

    candidates = {}
    title_variants = generate_title_variants(series_title)

    for attempt_title in title_variants[:3]:  # Try top 3 variants
        # Try with and without publisher
        for use_pub in [True, False]:
            filter_parts = [f"name:{attempt_title}"]
            if publisher and use_pub:
                filter_parts.append(f"publisher:{publisher}")

            params = {
                "api_key": cfg.comicvine_api_key,
                "format": "json",
                "filter": ",".join(filter_parts),
                "limit": 20,
                "field_list": "id,name,publisher,start_year,count_of_issues",
            }

            data = comicvine_get("volumes/", params)
            if data and data.get("error") == "OK":
                for vol in data.get("results", []):
                    vid = vol.get("id")
                    if vid and vid not in candidates:
                        candidates[vid] = vol

            if candidates and use_pub:
                break

        if len(candidates) >= 5:
            break

    return list(candidates.values())

def fetch_issue_candidates_for_volume(volume_id: int, issue_num: str) -> list:
    """Fetch issue candidates for a specific volume."""
    if not volume_id:
        return []

    candidates = {}

    if issue_num:
        normalized = normalize_issue_number(issue_num)

        # Try exact match
        for attempt_num in [issue_num, normalized]:
            params = {
                "api_key": cfg.comicvine_api_key,
                "format": "json",
                "filter": f"volume:{volume_id},issue_number:{attempt_num}",
                "limit": 10,
                "field_list": "id,volume,issue_number,cover_date,api_detail_url,name,deck,image",
            }

            data = comicvine_get("issues/", params)
            if data and data.get("error") == "OK":
                results = data.get("results", [])
                if results:
                    log(f"    Found {len(results)} issues for #{attempt_num} in this volume")
                for issue in results:
                    iid = issue.get("id")
                    if iid:
                        candidates[iid] = issue

        # If no exact match found, this volume doesn't have the issue
        if not candidates:
            log(f"    No match for #{issue_num} in this volume")
    else:
        # No issue number - get all issues (fallback)
        params = {
            "api_key": cfg.comicvine_api_key,
            "format": "json",
            "filter": f"volume:{volume_id}",
            "limit": 100,
            "field_list": "id,volume,issue_number,cover_date,api_detail_url,name,deck,image",
        }
        data = comicvine_get("issues/", params)
        if data and data.get("error") == "OK":
            for issue in data.get("results", []):
                iid = issue.get("id")
                if iid:
                    candidates[iid] = issue

    return list(candidates.values())

def fetch_issue_details(api_detail_url: str) -> Optional[dict]:
    """Fetch full issue details."""
    if not api_detail_url:
        return None
    try:
        params = {
            "api_key": cfg.comicvine_api_key,
            "format": "json",
            "field_list": "id,volume,issue_number,cover_date,description,character_credits,name,deck,image",
        }
        rel_path = api_detail_url.replace(cfg.comicvine_base_url + "/", "")
        data = comicvine_get(rel_path, params)
        return data.get("results") if data and data.get("error") == "OK" else None
    except (requests.exceptions.RequestException, KeyError, AttributeError):
        return None

# ------------------------------
# Scoring Functions
# ------------------------------
@dataclass(frozen=True)
class QueryContext:
    issue_num: str
    publisher: str
    year_likelihood: dict
    month_abbr: str
    story_tokens: set
    characters_norm: set
    series_title: str

def enhanced_candidate_scoring(detail_data: dict, query: QueryContext) -> float:
    """
    Score a candidate issue using all available signals.

    Scoring weights (based on discrimination power analysis):
    - Issue number: +15 (hard constraint)
    - Publisher: +5
    - Title similarity: +12 (NEW - very discriminating!)
    - Month: +8 exact, +4 close (doubled from before)
    - Year from price: +8 in range (reduced from +10)
    - Characters: +2 max (reduced - rarely helps)
    - Story tokens: +1 max (reduced - rarely helps)
    """
    score = 0.0

    # === HARD CONSTRAINT: Issue number ===
    issue_num_api = str(detail_data.get("issue_number", "")).strip()
    if query.issue_num:
        q_norm = normalize_issue_number(query.issue_num)
        cv_norm = normalize_issue_number(issue_num_api)

        if cv_norm != q_norm:
            # Allow off-by-one with penalty
            try:
                if abs(float(q_norm) - float(cv_norm)) <= 1:
                    score -= 8  # Heavy penalty for nearby issue
                else:
                    return -1e9  # Hard reject
            except (ValueError, TypeError):
                return -1e9
        else:
            score += 15  # Strong boost for exact match

    # === SOFT SIGNALS ===

    # Publisher (+5)
    vol = detail_data.get("volume", {})
    pub_name = (vol.get("publisher") or {}).get("name", "")
    if query.publisher and publisher_matches(query.publisher, pub_name):
        score += 5

    # Series title similarity (+12 max) - Very discriminating signal
    vol_name = vol.get("name", "")
    if query.series_title and vol_name:
        if title_matches(vol_name, query.series_title, threshold=0.6):
            score += 12
        elif title_matches(vol_name, query.series_title, threshold=0.4):
            score += 6

    # Year from price (+8 max, with heavy penalties for impossible years)
    cover_date = detail_data.get("cover_date", "")
    if cover_date and query.year_likelihood:
        parts = cover_date.split("-")
        if parts and parts[0].isdigit():
            cand_year = int(parts[0])
            likelihood = query.year_likelihood.get(cand_year, 0.0)

            if likelihood >= 1.0:
                score += 8   # In expected range
            elif likelihood >= 0.7:
                score += 5   # Very close
            elif likelihood >= 0.4:
                score += 2   # Somewhat close
            elif likelihood >= 0.2:
                score += 1   # Distant but possible
            else:
                # Calculate how far outside the expected range
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

    # Month (+8 max, doubled from +4) - Very discriminating signal!
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

    # Story title (+1 max, reduced from +3) - Rarely useful
    if query.story_tokens:
        story_blob = f"{detail_data.get('name', '')} {detail_data.get('deck', '')}".lower()
        matched = sum(1 for tok in query.story_tokens if tok in story_blob)
        if query.story_tokens:
            score += (matched / len(query.story_tokens)) * 1

    # Characters (+2 max, reduced from +4) - Rarely useful
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

    # Title similarity
    if q_series_tokens and vol_tokens:
        overlap = q_series_tokens & vol_tokens
        if len(overlap) >= 2:
            score += (len(overlap) / len(q_series_tokens | vol_tokens)) * 10
        elif overlap:
            score += 2

    # BONUS: Exact title match (main series indicator)
    if vol_tokens == q_series_tokens:
        score += 15

    # PENALTY: Mini-series indicators (colons, "vs", "starring", etc.)
    vol_lower = vol_name.lower()
    if ":" in vol_name:
        score -= 10  # "The Kree-Skrull War: The Avengers"
    if " vs " in vol_lower or " vs. " in vol_lower or " versus " in vol_lower:
        score -= 10  # "The X-Men vs The Avengers"
    if " starring " in vol_lower or " starring the " in vol_lower:
        score -= 10  # "The Kree-Skrull War Starring the Avengers"
    if " and " in vol_lower and "/" not in vol_name:
        score -= 5   # "Captain America and Iron Man" (but not "Amazing Spider-Man/Hulk")

    # PENALTY: Volume name much longer than query (mini-series tend to be wordy)
    if len(vol_tokens) > len(q_series_tokens) + 2:
        score -= 3

    # BONUS: Issue count (main series have more issues)
    count_issues = volume.get("count_of_issues")
    if count_issues and isinstance(count_issues, int):
        if count_issues > 100:
            score += 8  # Long-running main series
        elif count_issues > 50:
            score += 5
        elif count_issues > 20:
            score += 2
        elif count_issues < 10:
            score -= 2  # Mini-series

    # Publisher
    pub = (volume.get("publisher") or {}).get("name", "")
    if q_publisher_lower and publisher_matches(q_publisher_lower, pub):
        score += 5

    # Start year vs price
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

# ------------------------------
# GCD-Guided ComicVine Lookup
# ------------------------------
def lookup_comicvine_for_gcd_match(gcd_result: dict) -> Optional[dict]:
    """
    Use GCD metadata to find the corresponding ComicVine issue.
    This allows us to get cover images for visual comparison.

    Returns ComicVine issue details with image data, or None if not found.
    """
    series_name = gcd_result.get("series_name", "")
    issue_num = gcd_result.get("issue_number", "")
    publisher_name = gcd_result.get("publisher_name", "")
    year = gcd_result.get("year_began")

    if not series_name or not issue_num:
        return None

    log(f"[GCD→CV] Looking up '{series_name}' #{issue_num} on ComicVine...")

    # Search for the volume using GCD's corrected series name
    volume_candidates = fetch_volume_candidates(series_name, publisher_name)

    if not volume_candidates:
        log(f"[GCD→CV] No volumes found for '{series_name}'")
        return None

    # Score and rank volumes, preferring ones that match the year
    year_likelihood = {year: 1.0} if year else {}
    ranked_volumes = rank_volumes(volume_candidates, series_tokens(series_name),
                                  publisher_name, year_likelihood)

    # Try top 2 volumes to find the issue
    for vol in ranked_volumes[:2]:
        volume_id = vol.get("id")
        vol_name = vol.get("name", "")

        if not volume_id:
            continue

        # Search for the specific issue in this volume
        issue_candidates = fetch_issue_candidates_for_volume(volume_id, issue_num)

        if issue_candidates:
            # Get full details for the first match (includes image data)
            for ic in issue_candidates:
                detail = fetch_issue_details(ic.get("api_detail_url"))
                if detail and detail.get("image"):
                    log(f"[GCD→CV] ✓ Found: {vol_name} #{issue_num} (with cover image)")
                    return detail

    log(f"[GCD→CV] Could not find ComicVine match with cover image")
    return None

def _score_candidates(
    candidates: list,
    all_candidates: dict,
    query: QueryContext,
    original_img_path: str,
) -> tuple[dict | None, float | None]:
    """Score and visually compare candidates. Returns (detail, score) on SAME match, else (None, None)."""
    for cand in candidates:
        issue_id = cand.get("id")
        if issue_id in all_candidates:
            continue

        detail = fetch_issue_details(cand.get("api_detail_url"))
        if not detail:
            continue

        score = enhanced_candidate_scoring(detail, query)
        if score <= -1e5:
            continue

        is_match, adjusted_score, visual_result = _try_visual_match(
            detail, original_img_path, score
        )

        vol_name = detail.get("volume", {}).get("name", "")
        issue_no = detail.get("issue_number", "")

        if visual_result:
            log(f"  [{visual_result}] {vol_name} #{issue_no}: {score:.1f} -> {adjusted_score:.1f}")
        else:
            log(f"  [{adjusted_score:.1f}] {vol_name} #{issue_no}")

        if is_match:
            log(f"[RESULT] Best match: {vol_name} #{issue_no} (score: {adjusted_score:.1f})")
            return detail, adjusted_score

        all_candidates[issue_id] = (detail, adjusted_score)

    return None, None

# ------------------------------
# Main Search Logic
# ------------------------------
def find_best_match(qwen_data: dict, original_img_path: str = None) -> tuple:
    """Find the best ComicVine match using multiple strategies, including visual comparison."""

    # Extract and clean Qwen data
    series_title = qwen_data.get("canonical_title") or qwen_data.get("raw_title_text") or ""
    issue_num = (qwen_data.get("issue_number") or "").strip()

    # Clean issue number - remove any # prefix
    issue_num = issue_num.lstrip('#').strip()

    publisher = qwen_data.get("publisher_normalized") or qwen_data.get("publisher_raw") or ""

    # Handle null cover_month
    cover_month = qwen_data.get("cover_month")
    if cover_month:
        cover_month = cover_month.strip().upper()
    else:
        cover_month = ""

    if cover_month == "SEPT":
        cover_month = "SEP"

    # Try to get month from full month name (e.g., "JANUARY" -> "JAN")
    cover_month_full = qwen_data.get("cover_month_full")
    if cover_month_full:
        cover_month_full = cover_month_full.strip().upper()
        # Use MONTH_ABBR_TO_NUM (which has both full and abbreviated names) to get month number,
        # then MONTH_NUM_TO_ABBR to get the 3-letter abbreviation
        month_num = MONTH_ABBR_TO_NUM.get(cover_month_full)
        if month_num:
            cover_month = MONTH_NUM_TO_ABBR.get(month_num, cover_month)

    cover_price = qwen_data.get("cover_price") or ""
    is_annual = bool(qwen_data.get("is_annual_or_special"))
    story_title = qwen_data.get("story_title_text") or ""
    characters = qwen_data.get("main_characters") or []
    series_descriptor = qwen_data.get("series_descriptor") or ""

    # Only use descriptor if it's actually valid (not story text)
    if series_descriptor and is_valid_series_descriptor(series_descriptor):
        # ANTHOLOGY SERIES: Descriptor IS the actual series name (e.g., "Marvel Team-Up")
        # Qwen extracts character names as title, but descriptor is the real series
        if is_anthology_series(series_descriptor):
            search_title = series_descriptor
            log(f"[INFO] Anthology series detected - using descriptor as title: '{series_descriptor}'")
        # For annuals, don't duplicate "Annual" in the title
        elif "annual" in series_descriptor.lower() and "annual" not in series_title.lower():
            search_title = f"{series_title} {series_descriptor}".strip()
            log(f"[INFO] Using descriptor: '{series_descriptor}'")
        else:
            search_title = series_title
    else:
        search_title = series_title
        if series_descriptor:
            log(f"[INFO] Ignoring invalid descriptor: '{series_descriptor}'")

    # Prepare scoring data
    story_tokens = extract_story_tokens(story_title)
    q_chars_norm = {normalize_char_name(c) for c in characters} - {""}
    year_likelihood = get_year_likelihood_from_price(cover_price, is_annual)

    if year_likelihood:
        years = [y for y, l in year_likelihood.items() if l >= 0.7]
        if years:
            log(f"[PRICE] '{cover_price}' → years {min(years)}-{max(years)}")

    query = QueryContext(
        issue_num=issue_num,
        publisher=publisher,
        year_likelihood=year_likelihood,
        month_abbr=cover_month,
        story_tokens=story_tokens,
        characters_norm=q_chars_norm,
        series_title=search_title,
    )

    all_candidates = {}

    # ========================================
    # STRATEGY GCD: Grand Comics Database (LOCAL - NO API CALLS!)
    # ========================================
    # Search local SQLite database FIRST - instant and no rate limits!
    if cfg.use_gcd and issue_num:
        year_range = None
        if year_likelihood:
            years_filtered = [y for y, l in year_likelihood.items() if l >= 0.7]
            if years_filtered:
                year_range = (min(years_filtered), max(years_filtered))

        # Widen year range for king-size/giant-size (atypical pricing makes price→year unreliable)
        if series_descriptor and any(kw in series_descriptor.lower() for kw in ["giant-size", "giant size", "king-size", "king size"]):
            if year_range:
                year_range = (year_range[0] - 10, year_range[1] + 10)
                log(f"[INFO] Widened year range for '{series_descriptor}': {year_range[0]}-{year_range[1]}")
            else:
                # No price data — use broad range covering the giant/king-size era
                year_range = (1966, 1985)
                log(f"[INFO] No price year data; using broad range for '{series_descriptor}': 1966-1985")

        log(f"[GCD] Searching local database for '{search_title}' #{issue_num}...")
        gcd_results = search_gcd(search_title, issue_num, publisher, year_range, cover_month)

        if gcd_results:
            log(f"[GCD] Found {len(gcd_results)} local matches")
            for gcd_row in gcd_results:
                # Convert GCD format to ComicVine-like format for compatibility
                issue_data = gcd_to_comicvine_format(gcd_row)

                # Score the GCD result using same scoring logic
                # Use the complete issue_data which now includes character_credits
                detail = issue_data

                score = enhanced_candidate_scoring(detail, query)

                if score > -1e5:
                    vol_name = issue_data["volume"]["name"]
                    issue_no = issue_data["issue_number"]
                    pub_name = issue_data["volume"]["publisher"]["name"]
                    log(f"  [GCD {score:.1f}] {vol_name} #{issue_no} ({pub_name})")

                    issue_id = issue_data["id"]
                    all_candidates[issue_id] = (detail, score)

            # Check if we have high confidence or need to verify with ComicVine
            best_gcd_score = max(s for _, s in all_candidates.values()) if all_candidates else 0

            if best_gcd_score > 40:
                log(f"[GCD] ✓ Strong match found locally (score > 40), skipping API calls!")
                # Skip to visual comparison
            elif best_gcd_score > 0 and cfg.use_comicvine:
                # Medium confidence - use GCD metadata to find ComicVine match for visual verification
                log(f"[GCD] Medium confidence match (score {best_gcd_score:.1f}), searching ComicVine for cover images...")

                # Get top GCD results that need verification (score <= 40)
                gcd_matches_to_verify = [
                    (gcd_row, all_candidates[f"gcd_{gcd_row['issue_id']}"][1])
                    for gcd_row in gcd_results
                    if f"gcd_{gcd_row['issue_id']}" in all_candidates
                    and all_candidates[f"gcd_{gcd_row['issue_id']}"][1] <= 40
                ]
                # Sort by score descending, take top 3
                gcd_matches_to_verify.sort(key=lambda x: x[1], reverse=True)

                for gcd_row, gcd_score in gcd_matches_to_verify[:3]:
                    # Look up this GCD match on ComicVine to get cover image
                    cv_detail = lookup_comicvine_for_gcd_match(gcd_row)

                    if cv_detail:
                        # Score the ComicVine result
                        cv_score = enhanced_candidate_scoring(cv_detail, query)

                        if cv_score > -1e5:
                            cv_id = cv_detail.get("id") or cv_detail.get("volume", {}).get("id")
                            if cv_id:
                                vol_name = cv_detail.get('volume', {}).get('name', '')
                                issue_no = cv_detail.get('issue_number', '')

                                # EARLY EXIT: Try visual comparison immediately
                                is_match, adjusted_score, visual_result = _try_visual_match(
                                    cv_detail, original_img_path, cv_score
                                )

                                if visual_result:
                                    log(f"  [GCD→CV {visual_result}] {vol_name} #{issue_no}: {cv_score:.1f} → {adjusted_score:.1f}")
                                else:
                                    log(f"  [GCD→CV {cv_score:.1f}] {vol_name} #{issue_no}")

                                if is_match:
                                    log(f"[RESULT] Best match: {vol_name} #{issue_no} (score: {adjusted_score:.1f})")
                                    return cv_detail, adjusted_score

                                all_candidates[cv_id] = (cv_detail, adjusted_score)
        else:
            if cfg.use_comicvine:
                log(f"[GCD] No local matches found, trying ComicVine API...")
            else:
                log(f"[GCD] No local matches found (ComicVine API disabled)")

    # ========================================
    # STRATEGY 1: ComicVine Volume-based Search
    # ========================================
    # Only run if GCD didn't find a strong match (score <= 40)
    if cfg.use_comicvine and (not all_candidates or max(s for _, s in all_candidates.values()) <= 40):
        log(f"[Strategy 1] Volume-based search for '{search_title}'")

        volume_candidates = fetch_volume_candidates(search_title, publisher)
        log(f"[Strategy 1] Found {len(volume_candidates)} volumes")

        ranked_volumes = rank_volumes(volume_candidates, series_tokens(search_title),
                                     publisher, year_likelihood)

        for vol in ranked_volumes[:3]:  # Try top 3 volumes
            volume_id = vol.get("id")
            vol_name = vol.get("name")
            if not volume_id:
                continue

            log(f"  Trying volume: {vol_name}")
            issue_candidates = fetch_issue_candidates_for_volume(volume_id, issue_num)
            result, result_score = _score_candidates(
                issue_candidates, all_candidates, query, original_img_path,
            )
            if result:
                return result, result_score

    # ========================================
    # STRATEGY 2: Issue + Month Search
    # ========================================
    if cfg.use_comicvine and issue_num and cover_month and (not all_candidates or max(s for _, s in all_candidates.values()) <= 40):
        log(f"[Strategy 2] Issue+month search for #{issue_num} {cover_month}")
        candidates_2 = search_issues_by_number_and_month(
            issue_num, cover_month, year_likelihood, search_title, publisher
        )

        result, result_score = _score_candidates(
            candidates_2, all_candidates, query, original_img_path,
        )
        if result:
            return result, result_score

    # ========================================
    # STRATEGY 3: Direct Issue Search (last resort)
    # ========================================
    if cfg.use_comicvine and issue_num and (not all_candidates or max(s for _, s in all_candidates.values()) <= 40):
        log(f"[Strategy 3] Direct issue search for #{issue_num}")
        candidates_3 = search_issues_directly(search_title, issue_num, publisher)
        log(f"[Strategy 3] Found {len(candidates_3)} candidates")

        result, result_score = _score_candidates(
            candidates_3, all_candidates, query, original_img_path,
        )
        if result:
            return result, result_score

    if not all_candidates:
        log("[RESULT] No candidates found")
        return None, 0.0

    # Select best (if we reach here, no visual SAME match was found)
    best_id = max(all_candidates.keys(), key=lambda k: all_candidates[k][1])
    best_detail, best_score = all_candidates[best_id]

    vol_name = best_detail.get('volume', {}).get('name', '')
    issue_no = best_detail.get('issue_number', '')
    log(f"[RESULT] Best match: {vol_name} #{issue_no} (score: {best_score:.1f})")

    return best_detail, best_score

# ------------------------------
# Main Processing Loop
# ------------------------------

def deduplicate_tsv(tsv_path: str):
    """
    Remove duplicate entries from TSV file, keeping only the last occurrence of each file.
    This cleans up results from multiple runs, keeping only the most recent result.
    """
    if not os.path.exists(tsv_path):
        return

    try:
        rows = read_tsv_rows(tsv_path)
        total_entries = len(rows)

        # Keep last occurrence of each (box, filename)
        file_data = {}
        for i, row in enumerate(rows):
            key = (row.get("box", ""), row.get("filename", ""))
            file_data[key] = (row, i)

        unique_entries = len(file_data)
        duplicates_removed = total_entries - unique_entries

        if duplicates_removed > 0:
            with open(tsv_path, "w", encoding="utf-8", newline="") as f:
                f.write(TSV_HEADER)
                writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t", extrasaction="ignore")
                for _key, (row, _idx) in sorted(file_data.items(), key=lambda x: x[1][1]):
                    writer.writerow(row)

            log(f"[TSV] Deduplicated: removed {duplicates_removed} duplicate entries, kept {unique_entries} unique files")
        else:
            log(f"[TSV] No duplicates found ({unique_entries} unique files)")

    except Exception as e:
        log(f"[TSV] Deduplication failed: {e}")

def process_box(box_name: str, image_dir: str, outfile: str, logfile: str):
    """Process a single box folder."""
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp")

    # Resume logic - works in both single-box and batch mode
    processed_files = set()
    if cfg.use_resume and os.path.exists(outfile):
        deduplicate_tsv(outfile)
        try:
            rows = read_tsv_rows(outfile)
            file_confidence = {}
            for row in rows:
                key = (row["box"], row["filename"]) if cfg.batch_mode else row["filename"]
                file_confidence[key] = row.get("confidence", "")

            for key, confidence in file_confidence.items():
                if confidence == "high":
                    processed_files.add(key)

            low_medium_count = len([c for c in file_confidence.values() if c in ["low", "medium"]])
            log(f"[INFO] Resuming {box_name}: {len(processed_files)} high-confidence skipped, {low_medium_count} to retry")
        except (OSError, KeyError, ValueError):
            pass

    # Open log file
    cfg.log_handle = open(logfile, "a" if cfg.batch_mode else "w", encoding="utf-8")
    log(f"\nOdinsList Run Log - {box_name}")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)

    # Count total eligible images for progress (Step 8)
    all_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
    total = len(all_files)

    processed_count = 0
    for i, fname in enumerate(all_files, 1):
        resume_key = (box_name, fname) if cfg.batch_mode else fname
        if cfg.use_resume and resume_key in processed_files:
            continue

        img_path = os.path.join(image_dir, fname)
        log(f"\n{'='*60}\n[{i}/{total}] Processing: {fname}\n{'='*60}")

        # Extract with VLM
        img64 = load_image_b64(img_path)
        resp = cfg.client.chat.completions.create(
            model=cfg.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img64}"}},
                ],
            }],
            max_tokens=512, stream=False,
        )

        qwen_text = resp.choices[0].message.content.strip()
        log(f"[VLM] {qwen_text}")

        data = safe_json_loads(qwen_text)
        if not data:
            log(f"[WARN] Failed to parse JSON for {fname}")
            data = {}

        # Initialize from VLM
        title = data.get("canonical_title") or data.get("raw_title_text") or ""
        issue_num = (data.get("issue_number") or "").strip()
        publisher = data.get("publisher_normalized") or data.get("publisher_raw") or ""
        cover_month = data.get("cover_month")
        if cover_month:
            cover_month = cover_month.strip().upper()
        else:
            cover_month = ""
        year = ""

        # Check if we have enough data to search - need at least issue# OR month
        if not issue_num and not cover_month:
            log(f"[SKIP] Insufficient data for search (no issue# and no month) - using VLM-only data")
            best_detail = None
            best_score = 0
        else:
            best_detail, best_score = find_best_match(data, img_path)

        # Apply best match if found
        if best_detail and best_score > 0:
            vol = best_detail.get("volume", {})
            title = vol.get("name", title)
            issue_num = str(best_detail.get("issue_number", issue_num))

            pub_name = (vol.get("publisher") or {}).get("name")
            if pub_name:
                publisher = pub_name

            cover_date = best_detail.get("cover_date", "")
            if cover_date:
                parts = cover_date.split("-")
                if parts and parts[0].isdigit():
                    year = parts[0]
                    if not cover_month and len(parts) >= 2:
                        cover_month = MONTH_NUM_TO_ABBR.get(parts[1], cover_month)
        else:
            log("[INFO] Using VLM-only data")

        # Normalize all fields before output
        title, issue_num, cover_month, publisher, year = finalize_row(
            title, issue_num, cover_month, publisher, year
        )

        confidence = "high" if best_score > 40 else ("medium" if best_score > 20 else "low")

        # Write to TSV (new schema: title|issue_number|month|year|publisher|box|filename|notes|confidence)
        with open(outfile, "a", encoding="utf-8") as f:
            f.write(f"{title}\t{issue_num}\t{cover_month}\t{year}\t{publisher}\t{box_name}\t{fname}\t\t{confidence}\n")

        log(f"[RESULT] {fname} → {title} | #{issue_num} {cover_month} | {publisher} | {year} | {confidence} (score: {best_score:.1f})")
        processed_count += 1

    log(f"\n{'='*80}")
    log(f"DONE {box_name}: {processed_count} images processed")
    log(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not cfg.batch_mode:
        cfg.log_handle.close()

    return processed_count


# ============================================================================
# MAIN EXECUTION
# ============================================================================

TSV_COLUMNS = ["title", "issue_number", "month", "year", "publisher", "box", "filename", "notes", "confidence"]
TSV_HEADER = "\t".join(TSV_COLUMNS) + "\n"

def read_tsv_rows(tsv_path: str) -> list[dict]:
    """Read TSV file into list of dicts with named columns."""
    if not os.path.exists(tsv_path):
        return []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def _auto_detect_gcd_db(images_dir: str) -> Optional[str]:
    """Find a *.db file in the images directory."""
    dbs = globmod.glob(os.path.join(images_dir, "*.db"))
    if len(dbs) == 1:
        return dbs[0]
    if len(dbs) > 1:
        # Prefer most recently modified
        dbs.sort(key=os.path.getmtime, reverse=True)
        return dbs[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="OdinsList — comic book cataloging via VLM + database cross-reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  %(prog)s --images /path/to/Comic_Photos --box Box_01\n"
               "  %(prog)s --images /path/to/Comic_Photos --batch\n"
               "  %(prog)s --images /path/to/Comic_Photos --batch --resume\n",
    )
    parser.add_argument("--images", required=True, help="Base directory with Box_XX folders")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--box", help="Process a single box folder")
    mode_group.add_argument("--batch", action="store_true", help="Process all Box_XX folders")

    parser.add_argument("--out", help="Output TSV path (default: auto-generated in images dir)")
    parser.add_argument("--resume", action="store_true", help="Skip high-confidence matches from previous runs")
    parser.add_argument("--gcd-db", help="Path to GCD SQLite database (default: auto-detect *.db in images dir)")
    parser.add_argument("--vlm-url", help=f"VLM API base URL (default: env VLM_BASE_URL or http://127.0.0.1:8000/v1)")
    parser.add_argument("--vlm-model", help="VLM model name (default: env VLM_MODEL)")
    parser.add_argument("--no-gcd", action="store_true", help="Disable local GCD search")
    parser.add_argument("--no-visual", action="store_true", help="Disable cover image comparison")
    parser.add_argument("--no-comicvine", action="store_true", help="Disable ComicVine API (GCD-only mode)")

    args = parser.parse_args()

    # --- Populate config from CLI > env > defaults ---
    cfg.base_dir = os.path.abspath(args.images)
    if not os.path.isdir(cfg.base_dir):
        parser.error(f"Images directory not found: {cfg.base_dir}")

    cfg.batch_mode = args.batch
    cfg.box_name = args.box
    cfg.use_resume = args.resume

    # VLM configuration
    if args.vlm_url:
        cfg.vlm_base_url = args.vlm_url
    if args.vlm_model:
        cfg.model_name = args.vlm_model
    if not cfg.model_name:
        parser.error("VLM model name is required. Set --vlm-model or VLM_MODEL env var.")

    # ComicVine API
    if args.no_comicvine:
        cfg.use_comicvine = False
    elif not cfg.comicvine_api_key:
        print("[ERROR] COMICVINE_API_KEY not set. Export it or add to .env file.")
        print("  Get a free key at: https://comicvine.gamespot.com/api/")
        print("  Or use --no-comicvine for GCD-only mode.")
        exit(1)

    # GCD database
    if args.no_gcd:
        cfg.use_gcd = False
        cfg.gcd_db_path = None
    elif args.gcd_db:
        cfg.gcd_db_path = os.path.abspath(args.gcd_db)
        if not os.path.exists(cfg.gcd_db_path):
            parser.error(f"GCD database not found: {cfg.gcd_db_path}")
    else:
        cfg.gcd_db_path = _auto_detect_gcd_db(cfg.base_dir)
        if cfg.gcd_db_path:
            print(f"[INFO] Auto-detected GCD database: {cfg.gcd_db_path}")
        else:
            print("[INFO] No GCD database found, using ComicVine API only")
            cfg.use_gcd = False

    # Disable visual comparison if requested
    if args.no_visual:
        cfg.use_visual = False

    # Initialize VLM client
    cfg.client = OpenAI(base_url=cfg.vlm_base_url, api_key="not-needed")
    setup_cover_cache()
    print(f"[INFO] VLM endpoint: {cfg.vlm_base_url}")
    print(f"[INFO] VLM model: {cfg.model_name}")
    print(f"[INFO] Cover cache: {cfg.cover_cache_dir}")

    # --- Run ---
    if cfg.batch_mode:
        box_folders = sorted(globmod.glob(os.path.join(cfg.base_dir, "Box_*")))
        box_folders = [d for d in box_folders if os.path.isdir(d)]

        if not box_folders:
            print(f"[ERROR] No Box_* folders found in {cfg.base_dir}")
            exit(1)

        print(f"[BATCH] Found {len(box_folders)} box folders to process")

        cfg.outfile = args.out or os.path.join(cfg.base_dir, "odinslist_output.tsv")
        cfg.logfile = os.path.splitext(cfg.outfile)[0] + ".log"

        # Only create fresh header if not resuming or file doesn't exist
        if cfg.use_resume and os.path.exists(cfg.outfile):
            print(f"[INFO] Resuming into existing TSV: {cfg.outfile}")
        else:
            if os.path.exists(cfg.logfile):
                os.remove(cfg.logfile)
            with open(cfg.outfile, "w", encoding="utf-8") as f:
                f.write(TSV_HEADER)

        total_processed = 0
        for box_path in box_folders:
            box_name = os.path.basename(box_path)
            print(f"\n[BATCH] Processing {box_name}...")
            count = process_box(box_name, box_path, cfg.outfile, cfg.logfile)
            total_processed += count

        if cfg.log_handle:
            cfg.log_handle.close()

        print(f"\n[BATCH] Complete! Processed {total_processed} images from {len(box_folders)} boxes")
        print(f"[BATCH] Results: {cfg.outfile}")
        print(f"[BATCH] Log: {cfg.logfile}")

    else:
        image_dir = os.path.join(cfg.base_dir, cfg.box_name)
        cfg.outfile = args.out or os.path.join(cfg.base_dir, cfg.box_name, f"{cfg.box_name}.tsv")
        cfg.logfile = os.path.splitext(cfg.outfile)[0] + ".log"

        if not os.path.isdir(image_dir):
            print(f"[ERROR] Directory not found: {image_dir}")
            exit(1)

        # Only create fresh header if not resuming or file doesn't exist
        if cfg.use_resume and os.path.exists(cfg.outfile):
            print(f"[INFO] Resuming into existing TSV: {cfg.outfile}")
        else:
            with open(cfg.outfile, "w", encoding="utf-8") as f:
                f.write(TSV_HEADER)

        process_box(cfg.box_name, image_dir, cfg.outfile, cfg.logfile)

        print(f"\nResults written to: {cfg.outfile}")
        print(f"Log written to: {cfg.logfile}")


if __name__ == "__main__":
    main()
