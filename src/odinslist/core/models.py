"""Data classes for OdinsList."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from odinslist.config import OdinsListConfig


@dataclass
class ComicResult:
    """A single identified comic."""
    title: str = ""
    issue_number: str = ""
    month: str = ""
    year: str = ""
    publisher: str = ""
    box: str = ""
    filename: str = ""
    notes: str = ""
    confidence: str = ""

    def to_tsv_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "issue_number": self.issue_number,
            "month": self.month,
            "year": self.year,
            "publisher": self.publisher,
            "box": self.box,
            "filename": self.filename,
            "notes": self.notes,
            "confidence": self.confidence,
        }


@dataclass
class ScanConfig:
    """Configuration for a scan run."""
    images_path: str = ""
    boxes: list[str] = field(default_factory=list)
    gcd_db_path: str = ""
    vlm_base_url: str = ""
    vlm_model: str = ""
    comicvine_api_key: str = ""
    gcd_enabled: bool = True
    comicvine_enabled: bool = True
    visual_matching: bool = True
    resume: bool = False
    outfile: str = ""

    @classmethod
    def from_config(cls, cfg: OdinsListConfig, resume: bool = False) -> ScanConfig:
        return cls(
            images_path=cfg.input_root_dir,
            gcd_db_path=cfg.gcd_db_path,
            vlm_base_url=cfg.vlm_base_url,
            vlm_model=cfg.vlm_model,
            comicvine_api_key=cfg.comicvine_api_key,
            gcd_enabled=cfg.gcd_enabled,
            comicvine_enabled=cfg.comicvine_enabled,
            visual_matching=cfg.comicvine_enabled,  # auto-follows comicvine
            resume=resume,
            outfile=cfg.output_tsv_path,
        )
