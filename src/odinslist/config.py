"""Configuration management for OdinsList."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

import tomli_w


@dataclass
class OdinsListConfig:
    """All OdinsList settings."""
    # Paths
    input_root_dir: str = ""
    output_tsv_path: str = ""
    gcd_db_path: str = ""

    # VLM
    vlm_base_url: str = "http://127.0.0.1:8000/v1"
    vlm_model: str = ""

    # ComicVine
    comicvine_api_key: str = ""

    # Feature toggles
    gcd_enabled: bool = True
    comicvine_enabled: bool = True

    # Run parameters
    run_mode: str = "batch"       # "batch" | "single_box"
    single_box_dir: str = ""      # only used when run_mode == "single_box"


def default_config_path() -> Path:
    """Return ~/.config/odinslist/config.toml."""
    return Path.home() / ".config" / "odinslist" / "config.toml"


def is_first_run(config_path: Optional[Path] = None) -> bool:
    """Check if this is the first run (no config file)."""
    path = config_path or default_config_path()
    return not path.exists()


def load_config(config_path: Optional[Path] = None) -> OdinsListConfig:
    """Load config from TOML file. Returns defaults if file missing."""
    path = config_path or default_config_path()
    if not path.exists():
        return OdinsListConfig()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    cfg = OdinsListConfig()
    paths = data.get("paths", {})
    cfg.input_root_dir = paths.get("input_root_dir", cfg.input_root_dir)
    cfg.output_tsv_path = paths.get("output_tsv_path", cfg.output_tsv_path)
    cfg.gcd_db_path = paths.get("gcd_db", cfg.gcd_db_path)

    vlm = data.get("vlm", {})
    cfg.vlm_base_url = vlm.get("base_url", cfg.vlm_base_url)
    cfg.vlm_model = vlm.get("model", cfg.vlm_model)

    cv = data.get("comicvine", {})
    cfg.comicvine_api_key = cv.get("api_key", cfg.comicvine_api_key)

    feat = data.get("features", {})
    cfg.gcd_enabled = feat.get("gcd_enabled", cfg.gcd_enabled)
    cfg.comicvine_enabled = feat.get("comicvine_enabled", cfg.comicvine_enabled)

    run = data.get("run", {})
    cfg.run_mode = run.get("run_mode", cfg.run_mode)
    cfg.single_box_dir = run.get("single_box_dir", cfg.single_box_dir)

    return cfg


def save_config(cfg: OdinsListConfig, config_path: Optional[Path] = None) -> None:
    """Save config to TOML file."""
    path = config_path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "paths": {
            "input_root_dir": cfg.input_root_dir,
            "output_tsv_path": cfg.output_tsv_path,
            "gcd_db": cfg.gcd_db_path,
        },
        "vlm": {
            "base_url": cfg.vlm_base_url,
            "model": cfg.vlm_model,
        },
        "comicvine": {
            "api_key": cfg.comicvine_api_key,
        },
        "features": {
            "gcd_enabled": cfg.gcd_enabled,
            "comicvine_enabled": cfg.comicvine_enabled,
        },
        "run": {
            "run_mode": cfg.run_mode,
            "single_box_dir": cfg.single_box_dir,
        },
    }

    with open(path, "wb") as f:
        tomli_w.dump(data, f)
