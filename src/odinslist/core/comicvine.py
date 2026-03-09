"""ComicVine API client with rate limiting, caching, and search strategies."""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from curl_cffi import requests as curl_requests

from odinslist.core.normalization import (
    MONTH_ABBR_TO_NUM,
    normalize_issue_number,
)
from odinslist.core.scoring import (
    generate_title_variants,
    publisher_matches,
    series_tokens,
    title_matches,
    rank_volumes,
)

logger = logging.getLogger(__name__)

API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

COMICVINE_RATE_LIMIT_STREAK_LIMIT = max(1, int(os.environ.get("COMICVINE_RATE_LIMIT_STREAK_LIMIT", "3")))
COMICVINE_RATE_LIMIT_PAUSE_SEC = max(60, int(os.environ.get("COMICVINE_RATE_LIMIT_PAUSE_SEC", "900")))


class ComicVineClient:
    """ComicVine API client with rate limiting, caching, and retry logic."""

    def __init__(self, api_key: str, base_url: str = "https://comicvine.gamespot.com/api"):
        self.api_key = api_key
        self.base_url = base_url
        self._session = curl_requests.Session(impersonate="chrome131")
        self._cache: dict = {}
        self._pause_until = 0.0
        self._rate_limit_streak = 0
        self._resource_state: dict[str, dict] = {}
        self._volume_candidates_cache: dict[str, list] = {}
        self._volume_issues_cache: dict[int, list] = {}

    def _get_resource_type(self, path: str) -> str:
        clean = path.strip("/").split("/")[0]
        return clean if clean else "unknown"

    def _get_resource_state(self, resource: str) -> dict:
        if resource not in self._resource_state:
            self._resource_state[resource] = {
                "count": 0,
                "window_start": time.time(),
                "streak": 0,
                "pause_until": 0.0,
            }
        state = self._resource_state[resource]
        if time.time() - state["window_start"] >= 3600:
            state["count"] = 0
            state["window_start"] = time.time()
            state["streak"] = 0
            state["pause_until"] = 0.0
        return state

    def get(self, path: str, params: dict, max_retries: int = 3):
        """Make a request to ComicVine API with retry logic."""
        cache_key = (path, frozenset(params.items()))
        if cache_key in self._cache:
            logger.debug("Cache hit for %s", path)
            return self._cache[cache_key]

        resource = self._get_resource_type(path)
        rstate = self._get_resource_state(resource)

        now = time.time()
        if now < rstate["pause_until"]:
            remaining = int(rstate["pause_until"] - now)
            logger.info("ComicVine /%s/ in cooldown for %ds; skipping", resource, remaining)
            return None

        if now < self._pause_until:
            remaining = int(self._pause_until - now)
            logger.info("ComicVine is in global cooldown for %ds; skipping API call", remaining)
            return None

        url = f"{self.base_url}/{path.lstrip('/')}"

        for attempt in range(max_retries):
            try:
                time.sleep(1.0)
                resp = self._session.get(url, params=params, headers=API_HEADERS, timeout=10)
                rstate["count"] += 1

                if resp.status_code == 403:
                    logger.error("ComicVine 403 - Access forbidden")
                    return None

                if resp.status_code == 420:
                    rstate["streak"] += 1
                    self._rate_limit_streak += 1

                    if rstate["streak"] >= COMICVINE_RATE_LIMIT_STREAK_LIMIT:
                        rstate["pause_until"] = time.time() + COMICVINE_RATE_LIMIT_PAUSE_SEC
                        logger.warning(
                            "ComicVine /%s/ rate limit streak %d reached; cooling down for %ds",
                            resource, rstate["streak"], COMICVINE_RATE_LIMIT_PAUSE_SEC,
                        )
                        return None

                    if self._rate_limit_streak >= COMICVINE_RATE_LIMIT_STREAK_LIMIT:
                        self._pause_until = time.time() + COMICVINE_RATE_LIMIT_PAUSE_SEC
                        logger.warning(
                            "ComicVine global rate limit streak %d reached; cooling down for %ds",
                            self._rate_limit_streak, COMICVINE_RATE_LIMIT_PAUSE_SEC,
                        )
                        return None

                    wait_time = min(2 ** attempt * 3, 90)
                    logger.warning("Rate limited (420) on /%s/. Waiting %ds before retry %d/%d...",
                                   resource, wait_time, attempt + 1, max_retries)
                    time.sleep(wait_time)

                    if attempt >= 2:
                        time.sleep(10)
                    continue

                resp.raise_for_status()

                rstate["streak"] = 0
                self._rate_limit_streak = 0
                result = resp.json()
                self._cache[cache_key] = result
                return result

            except curl_requests.errors.RequestsError as e:
                wait_time = min(2 ** attempt, 30)
                logger.warning("Request error: %s. Waiting %ds before retry %d/%d...",
                               e, wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    logger.error("Max retries reached: %s", e)
                    return None

            except Exception as e:
                logger.error("Unexpected error in API call: %s: %s", type(e).__name__, e)
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached, skipping this API call")
                    return None

        return None

    def _filter_issue_candidates(
        self,
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

    def search_issues_by_number_and_month(
        self,
        issue_num: str,
        month_abbr: str,
        year_likelihood: dict,
        series_title: str = None,
        publisher: str = None,
    ) -> list:
        """Strategy 2: Use /search endpoint."""
        if not issue_num:
            return []

        if series_title:
            search_queries = [
                f'"{series_title}" {issue_num}',
                f'"{series_title}" #{issue_num}',
            ]
        else:
            search_queries = [f"#{issue_num}"]

        logger.info("Strategy 2: Search API: '%s' #%s %s", series_title, issue_num, month_abbr)

        candidates = {}
        month_num = MONTH_ABBR_TO_NUM.get(month_abbr.upper()) if month_abbr else None
        title_variants = generate_title_variants(series_title) if series_title else []

        for query in search_queries:
            params = {
                "api_key": self.api_key,
                "format": "json",
                "query": query,
                "resources": "issue",
                "limit": 30,
                "field_list": "id,volume,issue_number,cover_date,api_detail_url,name,deck,image",
            }

            logger.info("  Searching: \"%s\"", query)
            data = self.get("search/", params)
            if not data or data.get("error") != "OK":
                continue

            results = data.get("results", [])
            if not results:
                continue

            logger.info("    Found %d results", len(results))

            filtered = self._filter_issue_candidates(
                results, issue_num, title_variants, publisher,
                title_threshold=0.75, month_num=month_num,
            )
            for issue_id, issue in filtered.items():
                if issue_id not in candidates:
                    candidates[issue_id] = issue

            if len(candidates) >= 3:
                break

        logger.info("Strategy 2: Found %d candidates", len(candidates))
        return list(candidates.values())

    def search_issues_directly(self, series_title: str, issue_num: str, publisher: str) -> list:
        """Strategy 3: Search issues endpoint directly."""
        if not issue_num:
            return []

        candidates = {}
        title_variants = generate_title_variants(series_title) if series_title else []
        normalized_issue = normalize_issue_number(issue_num)

        for attempt_num in [issue_num, normalized_issue]:
            params = {
                "api_key": self.api_key,
                "format": "json",
                "filter": f"issue_number:{attempt_num}",
                "limit": 100,
                "field_list": "id,volume,issue_number,cover_date,api_detail_url,name,deck,image",
            }

            data = self.get("issues/", params)
            if not data or data.get("error") != "OK":
                continue

            filtered = self._filter_issue_candidates(
                data.get("results", []), issue_num, title_variants, publisher,
            )
            candidates.update(filtered)

        return list(candidates.values())

    def fetch_volume_candidates(self, series_title: str, publisher: str) -> list:
        """Fetch volume candidates from ComicVine."""
        if not series_title:
            return []

        cache_key = series_title.strip().lower()
        if cache_key in self._volume_candidates_cache:
            logger.debug("Volume candidates cache hit for '%s'", series_title)
            return self._volume_candidates_cache[cache_key]

        candidates = {}
        title_variants = generate_title_variants(series_title)

        for attempt_title in title_variants[:3]:
            for use_pub in [True, False]:
                filter_parts = [f"name:{attempt_title}"]
                if publisher and use_pub:
                    filter_parts.append(f"publisher:{publisher}")

                params = {
                    "api_key": self.api_key,
                    "format": "json",
                    "filter": ",".join(filter_parts),
                    "limit": 20,
                    "field_list": "id,name,publisher,start_year,count_of_issues",
                }

                data = self.get("volumes/", params)
                if data and data.get("error") == "OK":
                    for vol in data.get("results", []):
                        vid = vol.get("id")
                        if vid and vid not in candidates:
                            candidates[vid] = vol

                if candidates and use_pub:
                    break

            if len(candidates) >= 5:
                break

        result = list(candidates.values())
        self._volume_candidates_cache[cache_key] = result
        return result

    def _fetch_all_volume_issues(self, volume_id: int) -> list:
        """Fetch all issues for a volume (cached)."""
        if volume_id in self._volume_issues_cache:
            return self._volume_issues_cache[volume_id]
        params = {
            "api_key": self.api_key,
            "format": "json",
            "filter": f"volume:{volume_id}",
            "limit": 100,
            "field_list": "id,volume,issue_number,cover_date,api_detail_url,name,deck,image",
        }
        data = self.get("issues/", params)
        issues = []
        if data and data.get("error") == "OK":
            issues = data.get("results", [])
        self._volume_issues_cache[volume_id] = issues
        return issues

    def fetch_issue_candidates_for_volume(self, volume_id: int, issue_num: str) -> list:
        """Fetch issue candidates for a specific volume."""
        if not volume_id:
            return []

        all_issues = self._fetch_all_volume_issues(volume_id)

        if not issue_num:
            return all_issues

        normalized = normalize_issue_number(issue_num)
        candidates = {}
        for issue in all_issues:
            inum = issue.get("issue_number", "")
            if inum and (inum == issue_num or inum == normalized
                         or normalize_issue_number(inum) == normalized):
                iid = issue.get("id")
                if iid:
                    candidates[iid] = issue

        if candidates:
            logger.info("    Found %d issues for #%s in this volume", len(candidates), issue_num)
        else:
            logger.info("    No match for #%s in this volume", issue_num)

        return list(candidates.values())

    def fetch_issue_details(self, api_detail_url: str) -> Optional[dict]:
        """Fetch full issue details."""
        if not api_detail_url:
            return None
        try:
            params = {
                "api_key": self.api_key,
                "format": "json",
                "field_list": "id,volume,issue_number,cover_date,description,character_credits,name,deck,image",
            }
            rel_path = api_detail_url.replace(self.base_url + "/", "")
            data = self.get(rel_path, params)
            return data.get("results") if data and data.get("error") == "OK" else None
        except (curl_requests.errors.RequestsError, KeyError, AttributeError):
            return None

    def lookup_comicvine_for_gcd_match(self, gcd_result: dict) -> Optional[dict]:
        """Use GCD metadata to find the corresponding ComicVine issue."""
        series_name = gcd_result.get("series_name", "")
        issue_num = gcd_result.get("issue_number", "")
        publisher_name = gcd_result.get("publisher_name", "")
        year = gcd_result.get("year_began")

        if not series_name or not issue_num:
            return None

        logger.info("GCD->CV: Looking up '%s' #%s on ComicVine...", series_name, issue_num)

        volume_candidates = self.fetch_volume_candidates(series_name, publisher_name)

        if not volume_candidates:
            logger.info("GCD->CV: No volumes found for '%s'", series_name)
            return None

        year_likelihood = {year: 1.0} if year else {}
        ranked_volumes = rank_volumes(volume_candidates, series_tokens(series_name),
                                      publisher_name, year_likelihood)

        for vol in ranked_volumes[:2]:
            volume_id = vol.get("id")
            vol_name = vol.get("name", "")

            if not volume_id:
                continue

            issue_candidates = self.fetch_issue_candidates_for_volume(volume_id, issue_num)

            if issue_candidates:
                for ic in issue_candidates:
                    detail = self.fetch_issue_details(ic.get("api_detail_url"))
                    if detail and detail.get("image"):
                        logger.info("GCD->CV: Found: %s #%s (with cover image)", vol_name, issue_num)
                        return detail

        logger.info("GCD->CV: Could not find ComicVine match with cover image")
        return None
