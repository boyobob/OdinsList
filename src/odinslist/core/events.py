"""Scan events emitted by the scanner pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional

from odinslist.core.models import ComicResult


@dataclass
class ScanEvent:
    """Base event."""

    def to_json(self) -> str:
        """Serialize to JSON line."""
        data = asdict(self)
        data["event"] = type(self).__name__
        return json.dumps(data)


@dataclass
class BoxStarted(ScanEvent):
    box_name: str = ""
    image_count: int = 0


@dataclass
class BoxFinished(ScanEvent):
    box_name: str = ""
    results_count: int = 0


@dataclass
class ImageLoading(ScanEvent):
    filename: str = ""
    image_path: str = ""


@dataclass
class VLMExtracting(ScanEvent):
    pass


@dataclass
class VLMResult(ScanEvent):
    title: str = ""
    issue: str = ""
    publisher: str = ""
    year: str = ""


@dataclass
class GCDSearching(ScanEvent):
    strategy: int = 0
    title: str = ""
    issue: str = ""


@dataclass
class GCDMatchFound(ScanEvent):
    title: str = ""
    confidence: float = 0.0


@dataclass
class GCDNoMatch(ScanEvent):
    strategy: int = 0


@dataclass
class ComicVineSearching(ScanEvent):
    stage: str = ""  # "volumes", "issues", "covers"


@dataclass
class VisualComparing(ScanEvent):
    title: str = ""
    issue: str = ""


@dataclass
class ComicVineMatchFound(ScanEvent):
    title: str = ""
    confidence: float = 0.0


@dataclass
class ScanComplete(ScanEvent):
    result: Optional[ComicResult] = None
    confidence: str = ""


@dataclass
class ScanError(ScanEvent):
    filename: str = ""
    error: str = ""
