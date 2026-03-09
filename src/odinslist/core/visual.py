"""Cover image downloading, caching, and VLM-based visual comparison."""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Optional

from odinslist.core.browser import BrowserSession
from odinslist.core.vlm import load_image_b64

logger = logging.getLogger(__name__)


def download_cover(
    url: str,
    output_path: str,
    browser_session: BrowserSession,
    max_retries: int = 3,
) -> bool:
    """Download a cover image using curl_cffi (bypasses TLS fingerprinting)."""
    for attempt in range(max_retries):
        delay = random.uniform(0.5, 1.5)
        time.sleep(delay)

        try:
            response = browser_session.get(url, timeout=15)

            if response.status_code >= 400:
                logger.info("HTTP %d (attempt %d/%d)", response.status_code, attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    logger.info("Resetting session and retrying...")
                    browser_session.reset()
                    time.sleep(2)
                    continue
                else:
                    logger.info("Max retries reached, giving up on %s...", url[:60])
                    return False

            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                logger.info("Bad content-type: %s", content_type)
                return False

            with open(output_path, 'wb') as f:
                f.write(response.content)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                if attempt > 0:
                    logger.info("Success on retry %d", attempt + 1)
                return True

            if os.path.exists(output_path):
                os.remove(output_path)
            return False

        except Exception as e:
            logger.info("Exception: %s: %s (attempt %d/%d)", type(e).__name__, e, attempt + 1, max_retries)
            if os.path.exists(output_path):
                os.remove(output_path)
            if attempt < max_retries - 1:
                browser_session.reset()
                time.sleep(2)
                continue
            return False

    return False


def get_cached_cover(
    issue_id: int,
    image_data: dict,
    cover_cache_dir: str,
    browser_session: BrowserSession,
) -> Optional[str]:
    """Download and cache a cover image from ComicVine, return local path."""
    if not image_data or not issue_id:
        return None

    cache_path = os.path.join(cover_cache_dir, f"{issue_id}.jpg")

    if os.path.exists(cache_path):
        return cache_path

    for key in ["small_url", "medium_url", "thumb_url", "original_url"]:
        url = image_data.get(key)
        if url and download_cover(url, cache_path, browser_session):
            return cache_path

    return None


def compare_covers_with_vlm(
    original_path: str,
    candidate_path: str,
    client,
    model: str,
) -> str:
    """Compare two comic covers using VLM. Returns: SAME, VARIANT, or DIFFERENT."""
    if not original_path or not candidate_path:
        return "DIFFERENT"

    try:
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

        resp = client.chat.completions.create(
            model=model,
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

        if "SAME" in result:
            return "SAME"
        elif "VARIANT" in result:
            return "VARIANT"
        else:
            return "DIFFERENT"

    except Exception as e:
        logger.warning("Visual comparison failed: %s", e)
        return "DIFFERENT"


def try_visual_match(
    candidate_detail: dict,
    original_img_path: str,
    current_score: float,
    *,
    use_visual: bool,
    cover_cache_dir: str,
    browser_session: BrowserSession,
    client,
    model: str,
) -> tuple:
    """
    Attempt visual comparison for a single candidate.

    Returns:
        (is_match, adjusted_score, visual_result)
    """
    if not use_visual:
        return False, current_score, None

    vol_name = candidate_detail.get('volume', {}).get('name', 'Unknown')
    issue_no = candidate_detail.get('issue_number', '?')

    if not original_img_path:
        logger.info("Skipping %s #%s: no original image path", vol_name, issue_no)
        return False, current_score, None

    issue_id = candidate_detail.get("id")
    image_data = candidate_detail.get("image")
    if not image_data:
        logger.info("Skipping %s #%s: no cover image data from ComicVine", vol_name, issue_no)
        return False, current_score, None

    cover_path = get_cached_cover(issue_id, image_data, cover_cache_dir, browser_session)
    if not cover_path:
        logger.info("Skipping %s #%s: failed to download cover", vol_name, issue_no)
        return False, current_score, None

    logger.info("Comparing covers: %s #%s...", vol_name, issue_no)

    visual_result = compare_covers_with_vlm(original_img_path, cover_path, client, model)

    if visual_result == "SAME":
        return True, current_score + 25, visual_result
    elif visual_result == "VARIANT":
        return False, current_score + 10, visual_result
    else:
        return False, current_score - 10, visual_result
