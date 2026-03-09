"""CLI entry point: no args -> TUI, with args -> headless."""
from __future__ import annotations

import argparse
import sys

from odinslist import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odinslist",
        description="Automated comic book cataloging",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Paths
    parser.add_argument("--images", type=str, default=None, help="Base directory with Box_XX folders")

    # Mode (mutually exclusive for headless)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--box", type=str, default=None, help="Process a single box folder")
    mode.add_argument("--batch", action="store_true", default=False, help="Process all Box_XX folders")

    # Options
    parser.add_argument("--out", type=str, default=None, help="Output TSV path")
    parser.add_argument("--resume", action="store_true", default=False, help="Skip high-confidence matches")
    parser.add_argument("--gcd-db", type=str, default=None, help="Path to GCD database")
    parser.add_argument("--vlm-url", type=str, default=None, help="VLM API base URL")
    parser.add_argument("--vlm-model", type=str, default=None, help="VLM model name")
    parser.add_argument("--no-gcd", action="store_true", default=False, help="Disable GCD search")
    parser.add_argument("--no-comicvine", action="store_true", default=False, help="Disable ComicVine")
    parser.add_argument("--ipc", action="store_true", default=False, help="Run in IPC mode (JSON Lines on stdin/stdout)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.ipc:
        _run_ipc()
    elif args.box or args.batch:
        _run_headless(args)
    else:
        _run_tui()


def _run_ipc():
    """Launch IPC mode for TUI communication."""
    import asyncio
    from odinslist.ipc import run_ipc
    asyncio.run(run_ipc())


def _run_tui():
    """Launch the OpenTUI frontend."""
    import os
    import shutil
    import subprocess

    # src/odinslist/cli.py -> src/odinslist -> src -> src/tui
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tui_dir = os.path.join(src_dir, "tui")

    if not os.path.exists(os.path.join(tui_dir, "package.json")):
        print(f"Error: TUI not found at {tui_dir}", file=sys.stderr)
        print("Expected src/tui/package.json relative to the package source.", file=sys.stderr)
        sys.exit(1)

    if not shutil.which("bun"):
        print("Error: 'bun' is not installed. Install it: https://bun.sh", file=sys.stderr)
        sys.exit(1)

    # Check if node_modules exist, install if missing
    if not os.path.isdir(os.path.join(tui_dir, "node_modules")):
        print("Installing TUI dependencies...")
        subprocess.run(["bun", "install"], cwd=tui_dir, check=True)

    env = os.environ.copy()
    env["ODINSLIST_ROOT"] = src_dir
    env["ODINSLIST_PYTHON"] = sys.executable  # Pass venv python to TUI
    subprocess.run(["bun", "run", "src/index.tsx"], cwd=tui_dir, env=env)


def _run_headless(args):
    """Run scan without TUI (backward-compatible CLI mode)."""
    import asyncio
    from odinslist.config import load_config

    cfg = load_config()

    # Override config with CLI args
    if args.images:
        cfg.input_root_dir = args.images
    if args.vlm_url:
        cfg.vlm_base_url = args.vlm_url
    if args.vlm_model:
        cfg.vlm_model = args.vlm_model
    if args.gcd_db:
        cfg.gcd_db_path = args.gcd_db
    if args.no_gcd:
        cfg.gcd_enabled = False
    if args.no_comicvine:
        cfg.comicvine_enabled = False
    if args.out:
        cfg.output_tsv_path = args.out

    # Determine boxes and set run mode
    from odinslist.core.scanner import discover_boxes
    if args.batch:
        cfg.run_mode = "batch"
        boxes = discover_boxes(cfg.input_root_dir)
    else:
        cfg.run_mode = "single_box"
        cfg.single_box_dir = f"{cfg.input_root_dir}/{args.box}"
        boxes = [args.box]

    from odinslist.core.scanner import scan_box
    from odinslist.core.models import ScanConfig

    scan_cfg = ScanConfig.from_config(cfg, resume=args.resume)
    scan_cfg.boxes = boxes

    async def run():
        for box in boxes:
            async for event in scan_box(box, scan_cfg):
                _print_event(event)

    asyncio.run(run())


def _print_event(event):
    """Print a scan event as a tagged log line for headless mode."""
    from odinslist.core import events as ev

    match event:
        case ev.ImageLoading(filename=f):
            print(f"[VLM]       Loading {f}")
        case ev.VLMExtracting():
            print(f"[VLM]       Extracting cover details...")
        case ev.VLMResult(title=t, issue=i, publisher=p, year=y):
            print(f'[MATCH]     "{t}" #{i} \u00b7 {p} \u00b7 {y}')
        case ev.GCDSearching(strategy=s, title=t, issue=i):
            print(f"[GCD]       Strategy {s}: title=\"{t}\" issue={i}")
        case ev.GCDMatchFound(title=t, confidence=c):
            print(f"[GCD]       Match found: {t} (confidence: {c:.1f})")
        case ev.GCDNoMatch(strategy=s):
            print(f"[GCD]       Strategy {s}: no match")
        case ev.ComicVineSearching(stage=s):
            print(f"[COMICVINE] {s}...")
        case ev.ScanComplete(result=r, confidence=c):
            print(f"[RESULT]    {r.title} #{r.issue_number} ({r.year}) \u2014 {c}")
        case ev.ScanError(filename=f, error=e):
            print(f"[ERROR]     {f}: {e}")
        case ev.BoxStarted(box_name=b, image_count=n):
            print(f"[BOX]       Starting {b} ({n} images)")
        case ev.BoxFinished(box_name=b):
            print(f"[BOX]       Finished {b}")
