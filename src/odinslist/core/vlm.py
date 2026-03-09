"""VLM (Vision Language Model) extraction for comic book covers."""
from __future__ import annotations

import base64
import io
import json
import logging
from typing import AsyncGenerator

from PIL import Image, ImageOps

from odinslist.core.events import VLMExtracting, VLMResult, ScanEvent

logger = logging.getLogger(__name__)

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


def load_image_b64(path: str) -> str:
    """Load an image, auto-rotate via EXIF, convert to RGB, return base64 JPEG."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def safe_json_loads(text: str):
    """Parse JSON with fallback for truncated/malformed VLM output."""
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
            json_str = text[start:end+1] if (start != -1 and end != -1) else text

            if json_str.count('[') > json_str.count(']'):
                if json_str.count('"') % 2 == 1:
                    json_str += '"'
                json_str += ']' * (json_str.count('[') - json_str.count(']'))
                if json_str.count('{') > json_str.count('}'):
                    json_str += '}'
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
    return None


async def extract_cover_metadata(
    image_path: str,
    client,
    model: str,
) -> AsyncGenerator[ScanEvent, None]:
    """Extract metadata from a comic cover image using VLM.

    Yields VLMExtracting, then VLMResult events.
    Returns the parsed dict via the VLMResult event.
    """
    yield VLMExtracting()

    img64 = load_image_b64(image_path)
    resp = client.chat.completions.create(
        model=model,
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

    raw_text = resp.choices[0].message.content.strip()
    logger.debug("VLM raw output: %s", raw_text)

    data = safe_json_loads(raw_text) or {}

    title = data.get("canonical_title") or data.get("raw_title_text") or ""
    issue = (data.get("issue_number") or "").strip()
    publisher = data.get("publisher_normalized") or data.get("publisher_raw") or ""
    year = ""  # VLM doesn't guess years

    yield VLMResult(title=title, issue=issue, publisher=publisher, year=year)
