"""Grand Comics Database (GCD) local SQLite search."""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import Optional

from odinslist.core.normalization import (
    MONTH_ABBR_TO_NUM,
    PUBLISHER_ALIASES,
    normalize_issue_number,
    normalize_publisher,
)
from odinslist.core.scoring import generate_title_variants, series_tokens

logger = logging.getLogger(__name__)


def get_gcd_characters(conn, issue_ids: list) -> dict:
    """
    Fetch character credits for a list of issue IDs from GCD database.
    Returns dict mapping issue_id -> list of character names.
    """
    if not issue_ids:
        return {}

    try:
        cursor = conn.cursor()
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

        characters_by_issue = {}
        for row in cursor.fetchall():
            issue_id = row[0]
            char_name = row[1]
            if issue_id not in characters_by_issue:
                characters_by_issue[issue_id] = []
            characters_by_issue[issue_id].append(char_name)

        return characters_by_issue

    except Exception as e:
        logger.error("Character fetch error: %s", e)
        return {}


def _get_publisher_variants(publisher: str) -> list:
    """Get normalized publisher name variants for database matching."""
    if not publisher:
        return []
    canonical = normalize_publisher(publisher)
    if canonical in PUBLISHER_ALIASES:
        return [canonical] + [v.title() if v != v.upper() else v for v in PUBLISHER_ALIASES[canonical]]
    return [publisher]


def _score_and_filter_by_title(results: dict, series_title: str, threshold: float = 0.4, log_label: str = "") -> dict:
    """Score results by title similarity and filter out poor matches."""
    if not results or not series_title:
        return results

    title_variants = generate_title_variants(series_title, include_aggressive=False)

    for issue_id, result in results.items():
        db_title = result['series_name']
        best_similarity = 0.0

        for variant in title_variants:
            if variant:
                db_tokens = series_tokens(db_title)
                var_tokens = series_tokens(variant)
                if db_tokens and var_tokens:
                    jaccard = len(db_tokens & var_tokens) / len(db_tokens | var_tokens)
                    best_similarity = max(best_similarity, jaccard)
                    shorter, longer = (db_tokens, var_tokens) if len(db_tokens) <= len(var_tokens) else (var_tokens, db_tokens)
                    if shorter and shorter <= longer:
                        smaller_len = len(shorter)
                        if smaller_len == 1:
                            tok = next(iter(shorter))
                            ambiguous_singletons = {
                                "batman", "superman", "thor", "marvel",
                                "adventure", "comics", "amazing", "action",
                                "detective", "spider", "captain",
                            }
                            if len(tok) >= 7 and tok not in ambiguous_singletons:
                                containment = smaller_len / len(longer)
                                best_similarity = max(best_similarity, max(containment, 0.52))
                        elif smaller_len >= 2:
                            containment = smaller_len / len(longer)
                            best_similarity = max(best_similarity, max(containment, 0.55))

        result['title_similarity'] = best_similarity

    filtered_results = {k: v for k, v in results.items() if v.get('title_similarity', 0) >= threshold}

    if filtered_results:
        logger.info("%d candidates passed title filter%s (>= %s)", len(filtered_results), log_label, threshold)
        sorted_results = sorted(filtered_results.values(), key=lambda x: x.get('title_similarity', 0), reverse=True)
        for r in sorted_results[:5]:
            sim = r.get('title_similarity', 0)
            logger.info("  [%.2f] %s #%s (%s)", sim, r['series_name'], r['issue_number'], r['publisher_name'])
        return {r['issue_id']: r for r in sorted_results}
    else:
        logger.info("No candidates passed title filter%s (>= %s), rejecting all", log_label, threshold)
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
            exact_conditions = ["LOWER(p.name) = LOWER(?)" for _ in pub_variants]
            like_conditions = ["LOWER(p.name) LIKE ?"]
            all_conditions = exact_conditions + like_conditions
            parts.append(f"({' OR '.join(all_conditions)})")
            params.extend(pub_variants)
            params.append(f"%{publisher.lower()}%")

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


def _dedup_by_series(results: dict) -> dict:
    """Keep only the best-scoring issue per series to avoid one series dominating."""
    best_per_series = {}
    for issue_id, result in results.items():
        series_id = result.get('series_id')
        existing = best_per_series.get(series_id)
        if not existing or result.get('title_similarity', 0) > existing.get('title_similarity', 0):
            best_per_series[series_id] = result
    return {r['issue_id']: r for r in best_per_series.values()}


def search_gcd(
    series_title: str,
    issue_num: str,
    publisher: str,
    year_range: tuple = None,
    cover_month: str = None,
    *,
    db_path: str,
    use_gcd: bool = True,
) -> list:
    """Search the local Grand Comics Database (SQLite) for matching issues."""
    if not use_gcd or not os.path.exists(db_path):
        return []

    try:
        conn = sqlite3.connect(db_path)
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
            full_query = base_query + " AND " + " AND ".join(query_parts) + " LIMIT 200"
        else:
            full_query = base_query + " LIMIT 200"

        cursor.execute(full_query, params)
        for row in cursor.fetchall():
            issue_id = row['issue_id']
            if issue_id not in all_results:
                all_results[issue_id] = dict(row)

        all_results = _score_and_filter_by_title(all_results, series_title)

        # Fallback: without month constraint
        if len(all_results) == 0 and cover_month:
            logger.info("No matches with month filter, trying without month...")
            query_parts, params = _build_gcd_filters(issue_num, publisher, year_range)

            if query_parts:
                full_query = base_query + " AND " + " AND ".join(query_parts) + " LIMIT 200"
                cursor.execute(full_query, params)

                for row in cursor.fetchall():
                    issue_id = row['issue_id']
                    if issue_id not in all_results:
                        all_results[issue_id] = dict(row)

                all_results = _score_and_filter_by_title(all_results, series_title, log_label=" (no month)")

        # Series-level dedup
        if len(all_results) > 10:
            all_results = _dedup_by_series(all_results)

        # Fetch character credits
        if all_results:
            issue_ids = [result['issue_id'] for result in all_results.values()]
            characters_by_issue = get_gcd_characters(conn, issue_ids)
            for issue_id, result in all_results.items():
                result['characters'] = characters_by_issue.get(issue_id, [])

        conn.close()

        return sorted(all_results.values(), key=lambda x: x.get('title_similarity', 0), reverse=True)

    except Exception as e:
        logger.error("GCD search error: %s", e)
        import traceback
        traceback.print_exc()
        return []


def search_gcd_by_title(
    series_title: str,
    publisher: str = None,
    year_range: tuple = None,
    *,
    db_path: str,
    use_gcd: bool = True,
) -> list:
    """Search GCD by title (no issue number constraint). Strategy 3 fallback."""
    if not use_gcd or not os.path.exists(db_path) or not series_title:
        return []

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

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

        title_variants = generate_title_variants(series_title, include_aggressive=True)
        title_conditions = []
        params = []
        for variant in title_variants:
            if variant:
                title_conditions.append("LOWER(s.name) LIKE ?")
                params.append(f"%{variant.lower()}%")

        if not title_conditions:
            conn.close()
            return []

        parts = [f"({' OR '.join(title_conditions)})"]

        if publisher:
            pub_variants = _get_publisher_variants(publisher)
            if pub_variants:
                exact_conditions = ["LOWER(p.name) = LOWER(?)" for _ in pub_variants]
                like_conditions = ["LOWER(p.name) LIKE ?"]
                parts.append(f"({' OR '.join(exact_conditions + like_conditions)})")
                params.extend(pub_variants)
                params.append(f"%{publisher.lower()}%")

        if year_range:
            min_year, max_year = year_range
            parts.append("(s.year_began <= ? AND (s.year_ended IS NULL OR s.year_ended >= ?))")
            params.extend([max_year, min_year])

        full_query = base_query + " AND " + " AND ".join(parts) + " LIMIT 200"
        cursor.execute(full_query, params)

        all_results = {}
        for row in cursor.fetchall():
            issue_id = row['issue_id']
            if issue_id not in all_results:
                all_results[issue_id] = dict(row)

        all_results = _score_and_filter_by_title(all_results, series_title, log_label=" (title-focused)")

        if all_results:
            issue_ids = [result['issue_id'] for result in all_results.values()]
            characters_by_issue = get_gcd_characters(conn, issue_ids)
            for issue_id, result in all_results.items():
                result['characters'] = characters_by_issue.get(issue_id, [])

        conn.close()
        return sorted(all_results.values(), key=lambda x: x.get('title_similarity', 0), reverse=True)

    except Exception as e:
        logger.error("search_gcd_by_title error: %s", e)
        import traceback
        traceback.print_exc()
        return []


def gcd_to_comicvine_format(gcd_result: dict) -> dict:
    """Convert GCD database result to ComicVine-like format for compatibility."""
    character_credits = []
    if 'characters' in gcd_result and gcd_result['characters']:
        character_credits = [{"name": char_name} for char_name in gcd_result['characters']]

    return {
        "id": f"gcd_{gcd_result['issue_id']}",
        "issue_number": gcd_result['issue_number'],
        "cover_date": gcd_result['key_date'],
        "volume": {
            "name": gcd_result['series_name'],
            "id": f"gcd_{gcd_result['series_id']}",
            "publisher": {
                "name": gcd_result['publisher_name'],
                "id": f"gcd_{gcd_result['publisher_id']}"
            },
            "start_year": gcd_result['year_began'],
        },
        "name": None,
        "deck": None,
        "image": None,
        "api_detail_url": None,
        "character_credits": character_credits,
        "source": "GCD",
    }
