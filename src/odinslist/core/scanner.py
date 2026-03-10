"""Pipeline orchestrator — yields ScanEvent objects for each processing step."""
from __future__ import annotations

import asyncio
import csv
import glob as globmod
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

from openai import OpenAI

from odinslist.core.browser import BrowserSession
from odinslist.core.comicvine import ComicVineClient
from odinslist.core.events import (
    BoxFinished,
    BoxStarted,
    ComicVineMatchFound,
    ComicVineSearching,
    GCDMatchFound,
    GCDNoMatch,
    GCDSearching,
    ImageLoading,
    ScanComplete,
    ScanError,
    ScanEvent,
    VisualComparing,
    VLMExtracting,
    VLMResult,
)
from odinslist.core.gcd import gcd_to_comicvine_format, search_gcd, search_gcd_by_title
from odinslist.core.models import ComicResult, ScanConfig
from odinslist.core.normalization import (
    MONTH_ABBR_TO_NUM,
    MONTH_NUM_TO_ABBR,
    finalize_row,
    get_year_likelihood_from_price,
    normalize_issue_number,
)
from odinslist.core.scoring import (
    QueryContext,
    enhanced_candidate_scoring,
    extract_story_tokens,
    generate_title_variants,
    is_anthology_series,
    is_valid_series_descriptor,
    normalize_char_name,
    rank_volumes,
    series_tokens,
)
from odinslist.core.visual import try_visual_match
from odinslist.core.vlm import USER_PROMPT, load_image_b64, safe_json_loads

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}

TSV_COLUMNS = ["title", "issue_number", "month", "year", "publisher", "box", "filename", "notes", "confidence"]
TSV_HEADER = "\t".join(TSV_COLUMNS) + "\n"


def _ensure_tsv_header(tsv_path: str) -> None:
    """Create TSV parent dirs and header if missing/empty."""
    path = Path(tsv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(TSV_HEADER, encoding="utf-8")


def _tsv_row_line(result: ComicResult) -> str:
    """Serialize a ComicResult to one TSV line in canonical column order."""
    row = result.to_tsv_dict()
    values = [str(row.get(col, "")) for col in TSV_COLUMNS]
    return "\t".join(values) + "\n"


def _replace_or_append_tsv_row(tsv_path: str, result: ComicResult) -> None:
    """Upsert by (box, filename) so reruns update prior rows instead of duplicating."""
    _ensure_tsv_header(tsv_path)
    new_row = _tsv_row_line(result)

    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    replaced = False
    for i, line in enumerate(lines):
        if i == 0:
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 7 and fields[5] == result.box and fields[6] == result.filename:
            lines[i] = new_row
            replaced = True
            break

    if not replaced:
        lines.append(new_row)

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _load_processed_filenames(tsv_path: str, box_name: str) -> set[str]:
    """Load already-written filenames for a given box from TSV checkpoint file."""
    path = Path(tsv_path)
    if not path.exists():
        return set()

    processed: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if (row.get("box") or "") == box_name:
                fname = row.get("filename") or ""
                if fname:
                    processed.add(fname)
    return processed


def discover_boxes(images_path: str) -> list[str]:
    """Find Box_XX folders in the images directory."""
    base = Path(images_path)
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and d.name.startswith("Box_")
    )


def discover_images(images_path: str, box_name: str) -> list[Path]:
    """Find all image files in a box folder."""
    box_dir = Path(images_path) / box_name
    if not box_dir.exists():
        return []
    return sorted(
        f for f in box_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_images(images_path: str, box_name: str) -> int:
    """Count images in a box folder."""
    return len(discover_images(images_path, box_name))


def _auto_detect_gcd_db(images_dir: str) -> Optional[str]:
    """Find a *.db file in the images directory."""
    dbs = globmod.glob(os.path.join(images_dir, "*.db"))
    if len(dbs) == 1:
        return dbs[0]
    if len(dbs) > 1:
        dbs.sort(key=os.path.getmtime, reverse=True)
        return dbs[0]
    return None


async def scan_box(
    box_name: str,
    config: ScanConfig,
    should_pause: Optional[Callable[[], bool]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> AsyncGenerator[ScanEvent, None]:
    """Process all images in a box, yielding events."""
    images = discover_images(config.images_path, box_name)
    processed = (
        _load_processed_filenames(config.outfile, box_name)
        if config.resume and config.outfile
        else set()
    )
    if processed:
        images = [p for p in images if p.name not in processed]
    yield BoxStarted(box_name=box_name, image_count=len(images))

    # Set up shared resources
    client = OpenAI(base_url=config.vlm_base_url, api_key="not-needed")
    browser_session = BrowserSession()
    cv_client = None
    if config.comicvine_enabled and config.comicvine_api_key:
        cv_client = ComicVineClient(api_key=config.comicvine_api_key)

    # Cover cache
    cover_cache_dir = "/tmp/comicvine_covers"
    os.makedirs(cover_cache_dir, exist_ok=True)

    # GCD db path
    gcd_db_path = config.gcd_db_path
    if not gcd_db_path and config.gcd_enabled:
        gcd_db_path = _auto_detect_gcd_db(config.images_path) or ""

    if config.outfile:
        _ensure_tsv_header(config.outfile)

    results_count = 0
    for image_path in images:
        if should_stop and should_stop():
            break

        if should_pause:
            while should_pause():
                if should_stop and should_stop():
                    break
                await asyncio.sleep(0.1)
            if should_stop and should_stop():
                break

        async for event in scan_single_image(
            image_path=image_path,
            box_name=box_name,
            config=config,
            client=client,
            browser_session=browser_session,
            cv_client=cv_client,
            cover_cache_dir=cover_cache_dir,
            gcd_db_path=gcd_db_path,
        ):
            yield event
            if isinstance(event, ScanComplete):
                if config.outfile and event.result:
                    _replace_or_append_tsv_row(config.outfile, event.result)
                results_count += 1

    yield BoxFinished(box_name=box_name, results_count=results_count)


async def scan_single_image(
    image_path: Path,
    box_name: str,
    config: ScanConfig,
    client,
    browser_session: BrowserSession,
    cv_client: Optional[ComicVineClient],
    cover_cache_dir: str,
    gcd_db_path: str,
) -> AsyncGenerator[ScanEvent, None]:
    """Process one image through the full pipeline, yielding events."""
    fname = image_path.name
    yield ImageLoading(filename=fname, image_path=str(image_path))

    try:
        # === VLM EXTRACTION ===
        yield VLMExtracting()

        img64 = load_image_b64(str(image_path))
        resp = client.chat.completions.create(
            model=config.vlm_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img64}"}},
                ],
            }],
            max_tokens=512,
            stream=False,
        )

        qwen_text = resp.choices[0].message.content.strip()
        data = safe_json_loads(qwen_text) or {}

        title = data.get("canonical_title") or data.get("raw_title_text") or ""
        issue_num = (data.get("issue_number") or "").strip().lstrip('#').strip()
        publisher = data.get("publisher_normalized") or data.get("publisher_raw") or ""
        cover_month = (data.get("cover_month") or "").strip().upper()
        year = ""

        # Reject unreasonable issue numbers
        try:
            if issue_num and float(issue_num) > 1500:
                issue_num = ""
        except ValueError:
            pass

        yield VLMResult(title=title, issue=issue_num, publisher=publisher, year=year)

        # === SEARCH ===
        if not issue_num and not cover_month:
            # Not enough data to search
            title, issue_num, cover_month, publisher, year = finalize_row(
                title, issue_num, cover_month, publisher, year
            )
            confidence = "low"
            result = ComicResult(
                title=title, issue_number=issue_num, month=cover_month,
                year=year, publisher=publisher, box=box_name,
                filename=fname, confidence=confidence,
            )
            yield ScanComplete(result=result, confidence=confidence)
            return

        # Prepare search context
        cover_price = data.get("cover_price") or ""
        is_annual = bool(data.get("is_annual_or_special"))
        story_title = data.get("story_title_text") or ""
        characters = data.get("main_characters") or []
        series_descriptor = data.get("series_descriptor") or ""

        # Handle series descriptor
        if series_descriptor and is_valid_series_descriptor(series_descriptor):
            if is_anthology_series(series_descriptor):
                search_title = series_descriptor
            elif series_descriptor.lower() not in title.lower():
                search_title = f"{title} {series_descriptor}".strip()
            else:
                search_title = title
        else:
            search_title = title

        # Cover month normalization
        if cover_month == "SEPT":
            cover_month = "SEP"
        cover_month_full = data.get("cover_month_full")
        if cover_month_full:
            month_num = MONTH_ABBR_TO_NUM.get(cover_month_full.strip().upper())
            if month_num:
                cover_month = MONTH_NUM_TO_ABBR.get(month_num, cover_month)

        # Prepare scoring data
        story_toks = extract_story_tokens(story_title)
        q_chars_norm = frozenset({normalize_char_name(c) for c in characters} - {""})
        year_likelihood = get_year_likelihood_from_price(cover_price, is_annual)

        query = QueryContext(
            issue_num=issue_num,
            publisher=publisher,
            year_likelihood=year_likelihood,
            month_abbr=cover_month,
            story_tokens=frozenset(story_toks),
            characters_norm=q_chars_norm,
            series_title=search_title,
        )

        all_candidates = {}
        best_detail = None
        best_score = 0.0

        # === GCD SEARCH ===
        if config.gcd_enabled and gcd_db_path and issue_num:
            year_range = None
            if year_likelihood:
                years_filtered = [y for y, l in year_likelihood.items() if l >= 0.7]
                if years_filtered:
                    year_range = (min(years_filtered), max(years_filtered))

            if series_descriptor and any(kw in series_descriptor.lower() for kw in ["giant-size", "giant size", "king-size", "king size"]):
                if year_range:
                    year_range = (year_range[0] - 10, year_range[1] + 10)
                else:
                    year_range = (1966, 1985)

            yield GCDSearching(strategy=1, title=search_title, issue=issue_num)
            gcd_results = search_gcd(
                search_title, issue_num, publisher, year_range, cover_month,
                db_path=gcd_db_path, use_gcd=True,
            )

            if gcd_results:
                yield GCDMatchFound(title=gcd_results[0]['series_name'],
                                    confidence=gcd_results[0].get('title_similarity', 0))
                for gcd_row in gcd_results:
                    issue_data = gcd_to_comicvine_format(gcd_row)
                    detail = issue_data
                    score = enhanced_candidate_scoring(detail, query)
                    if score > -1e5:
                        all_candidates[issue_data["id"]] = (detail, score)
            else:
                yield GCDNoMatch(strategy=1)

            # GCD Strategy 2: Broad search without title
            best_gcd_score = max((s for _, s in all_candidates.values()), default=0)
            if best_gcd_score < 40:
                yield GCDSearching(strategy=2, title="", issue=issue_num)
                gcd_results_broad = search_gcd(
                    "", issue_num, publisher, year_range, cover_month,
                    db_path=gcd_db_path, use_gcd=True,
                )
                if gcd_results_broad:
                    for gcd_row in gcd_results_broad:
                        issue_data = gcd_to_comicvine_format(gcd_row)
                        if issue_data["id"] not in all_candidates:
                            score = enhanced_candidate_scoring(issue_data, query)
                            if score > -1e5:
                                all_candidates[issue_data["id"]] = (issue_data, score)
                else:
                    yield GCDNoMatch(strategy=2)

            # GCD Strategy 3: Title-focused (no issue# constraint)
            best_gcd_score = max((s for _, s in all_candidates.values()), default=0)
            if best_gcd_score < 40 and search_title:
                yield GCDSearching(strategy=3, title=search_title, issue="")
                gcd_results_title = search_gcd_by_title(
                    search_title, publisher, year_range,
                    db_path=gcd_db_path, use_gcd=True,
                )
                if gcd_results_title:
                    for gcd_row in gcd_results_title:
                        issue_data = gcd_to_comicvine_format(gcd_row)
                        if issue_data["id"] not in all_candidates:
                            score = enhanced_candidate_scoring(issue_data, query)
                            if score > -1e5:
                                all_candidates[issue_data["id"]] = (issue_data, score)
                else:
                    yield GCDNoMatch(strategy=3)

        # === COMICVINE SEARCH ===
        best_score_so_far = max((s for _, s in all_candidates.values()), default=0)

        if cv_client and best_score_so_far < 40:
            # Strategy 1: Volume-based search
            yield ComicVineSearching(stage="volumes")
            volume_candidates = cv_client.fetch_volume_candidates(search_title, publisher)

            if volume_candidates:
                ranked_volumes = rank_volumes(
                    volume_candidates, series_tokens(search_title),
                    publisher, year_likelihood,
                )

                for vol in ranked_volumes[:3]:
                    volume_id = vol.get("id")
                    if not volume_id:
                        continue

                    yield ComicVineSearching(stage="issues")
                    issue_candidates = cv_client.fetch_issue_candidates_for_volume(volume_id, issue_num)

                    for cand in issue_candidates:
                        issue_id = cand.get("id")
                        if issue_id in all_candidates:
                            continue

                        detail = cv_client.fetch_issue_details(cand.get("api_detail_url"))
                        if not detail:
                            continue

                        score = enhanced_candidate_scoring(detail, query)
                        if score <= -1e5:
                            continue

                        if config.visual_matching:
                            yield VisualComparing(
                                title=detail.get("volume", {}).get("name", ""),
                                issue=str(detail.get("issue_number", "")),
                            )
                        is_match, adjusted_score, visual_result = try_visual_match(
                            detail, str(image_path), score,
                            use_visual=config.visual_matching,
                            cover_cache_dir=cover_cache_dir,
                            browser_session=browser_session,
                            client=client,
                            model=config.vlm_model,
                        )

                        if is_match:
                            vol_name = detail.get("volume", {}).get("name", "")
                            yield ComicVineMatchFound(title=vol_name, confidence=adjusted_score)
                            all_candidates[issue_id] = (detail, adjusted_score)
                            best_detail = detail
                            best_score = adjusted_score
                            break

                        all_candidates[issue_id] = (detail, adjusted_score)

                    if best_detail:
                        break

            # Strategy 2: Issue + Month search
            if not best_detail and issue_num and cover_month:
                best_score_so_far = max((s for _, s in all_candidates.values()), default=0)
                if best_score_so_far < 40:
                    yield ComicVineSearching(stage="search")
                    candidates_2 = cv_client.search_issues_by_number_and_month(
                        issue_num, cover_month, year_likelihood, search_title, publisher
                    )
                    for cand in candidates_2:
                        issue_id = cand.get("id")
                        if issue_id in all_candidates:
                            continue
                        detail = cv_client.fetch_issue_details(cand.get("api_detail_url"))
                        if not detail:
                            continue
                        score = enhanced_candidate_scoring(detail, query)
                        if score <= -1e5:
                            continue
                        if config.visual_matching:
                            yield VisualComparing(
                                title=detail.get("volume", {}).get("name", ""),
                                issue=str(detail.get("issue_number", "")),
                            )
                        is_match, adjusted_score, visual_result = try_visual_match(
                            detail, str(image_path), score,
                            use_visual=config.visual_matching,
                            cover_cache_dir=cover_cache_dir,
                            browser_session=browser_session,
                            client=client,
                            model=config.vlm_model,
                        )
                        if is_match:
                            vol_name = detail.get("volume", {}).get("name", "")
                            yield ComicVineMatchFound(title=vol_name, confidence=adjusted_score)
                            best_detail = detail
                            best_score = adjusted_score
                            break
                        all_candidates[issue_id] = (detail, adjusted_score)

            # Strategy 3: Direct issue search
            if not best_detail and issue_num:
                best_score_so_far = max((s for _, s in all_candidates.values()), default=0)
                if best_score_so_far < 40:
                    yield ComicVineSearching(stage="direct")
                    candidates_3 = cv_client.search_issues_directly(search_title, issue_num, publisher)
                    for cand in candidates_3:
                        issue_id = cand.get("id")
                        if issue_id in all_candidates:
                            continue
                        detail = cv_client.fetch_issue_details(cand.get("api_detail_url"))
                        if not detail:
                            continue
                        score = enhanced_candidate_scoring(detail, query)
                        if score <= -1e5:
                            continue
                        if config.visual_matching:
                            yield VisualComparing(
                                title=detail.get("volume", {}).get("name", ""),
                                issue=str(detail.get("issue_number", "")),
                            )
                        is_match, adjusted_score, visual_result = try_visual_match(
                            detail, str(image_path), score,
                            use_visual=config.visual_matching,
                            cover_cache_dir=cover_cache_dir,
                            browser_session=browser_session,
                            client=client,
                            model=config.vlm_model,
                        )
                        if is_match:
                            vol_name = detail.get("volume", {}).get("name", "")
                            yield ComicVineMatchFound(title=vol_name, confidence=adjusted_score)
                            best_detail = detail
                            best_score = adjusted_score
                            break
                        all_candidates[issue_id] = (detail, adjusted_score)

        # === SELECT BEST ===
        if not best_detail and all_candidates:
            best_id = max(all_candidates.keys(), key=lambda k: all_candidates[k][1])
            best_detail, best_score = all_candidates[best_id]

        # Apply best match
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

        # Normalize
        title, issue_num, cover_month, publisher, year = finalize_row(
            title, issue_num, cover_month, publisher, year
        )

        confidence = "high" if best_score >= 40 else ("medium" if best_score > 20 else "low")

        result = ComicResult(
            title=title,
            issue_number=issue_num,
            month=cover_month,
            year=year,
            publisher=publisher,
            box=box_name,
            filename=fname,
            confidence=confidence,
        )

        yield ScanComplete(result=result, confidence=confidence)

    except Exception as e:
        logger.error("Error processing %s: %s", fname, e, exc_info=True)
        yield ScanError(filename=fname, error=str(e))
