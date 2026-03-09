"""IPC mode: JSON Lines protocol over stdin/stdout."""
from __future__ import annotations

import asyncio
import json
import os
import queue as queue_mod
import sys
import threading
from pathlib import Path
from typing import Optional

from odinslist.config import (
    OdinsListConfig,
    is_first_run,
    load_config,
    save_config,
)
from odinslist.core.models import ScanConfig
from odinslist.core.scanner import discover_boxes, count_images, scan_box


def _write_event(data: dict) -> None:
    """Write a JSON line to stdout and flush."""
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


def _write_error(msg: str) -> None:
    """Write to stderr for debug/error logging."""
    sys.stderr.write(f"[ipc] {msg}\n")
    sys.stderr.flush()




async def _handle_scan_preview(cfg: OdinsListConfig) -> None:
    """Count images using config as source of truth."""
    if cfg.run_mode == "single_box" and cfg.single_box_dir:
        box_name = Path(cfg.single_box_dir).name
        count = count_images(cfg.input_root_dir, box_name)
        _write_event({
            "event": "scan_preview",
            "total_images": count,
            "boxes": [{"name": box_name, "count": count}],
        })
    else:
        boxes = discover_boxes(cfg.input_root_dir)
        box_info = []
        total = 0
        for b in boxes:
            c = count_images(cfg.input_root_dir, b)
            box_info.append({"name": b, "count": c})
            total += c
        _write_event({
            "event": "scan_preview",
            "total_images": total,
            "boxes": box_info,
        })


async def _handle_scan(
    cfg: OdinsListConfig,
    cancel_event: asyncio.Event,
    resume: bool = False,
) -> None:
    """Run scan using config as the single source of truth."""
    scan_cfg = ScanConfig.from_config(cfg, resume=resume)

    # Derive boxes from config
    if cfg.run_mode == "single_box" and cfg.single_box_dir:
        box_name = Path(cfg.single_box_dir).name
        boxes = [box_name]
    else:
        boxes = discover_boxes(cfg.input_root_dir)

    scan_cfg.boxes = boxes

    event_queue: queue_mod.Queue = queue_mod.Queue()
    stop_flag = threading.Event()

    def scan_worker() -> None:
        loop = asyncio.new_event_loop()
        async def run() -> None:
            for box_name in boxes:
                if stop_flag.is_set():
                    break
                async for ev in scan_box(box_name, scan_cfg, should_stop=stop_flag.is_set):
                    if stop_flag.is_set():
                        break
                    event_queue.put(json.loads(ev.to_json()))
        try:
            loop.run_until_complete(run())
        except Exception as exc:
            event_queue.put({"event": "ScanError", "filename": "", "error": str(exc)})
        finally:
            loop.close()
            event_queue.put(None)

    thread = threading.Thread(target=scan_worker, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        try:
            event = await loop.run_in_executor(None, lambda: event_queue.get(timeout=0.1))
        except queue_mod.Empty:
            if cancel_event.is_set():
                stop_flag.set()
                break
            if not thread.is_alive():
                break
            continue
        if event is None:
            break
        if cancel_event.is_set():
            stop_flag.set()
            break
        _write_event(event)

    if not cancel_event.is_set():
        _write_event({"event": "run_complete"})


async def _handle_get_config(cfg: OdinsListConfig) -> None:
    from dataclasses import asdict
    data = asdict(cfg)
    data["is_first_run"] = is_first_run()
    _write_event({"event": "config", "config": data})


async def _handle_set_config(updates: dict) -> OdinsListConfig:
    cfg = load_config()
    for key, value in updates.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    save_config(cfg)
    return cfg


async def _handle_validate_config(cfg: OdinsListConfig) -> None:
    """Check config for errors/warnings and send results."""
    errors: list[str] = []
    warnings: list[str] = []

    if not cfg.vlm_base_url.strip():
        errors.append("Missing VLM API endpoint — configure in Settings")

    if not cfg.vlm_model.strip():
        warnings.append("Missing VLM model name — configure in Settings")

    if cfg.gcd_enabled:
        if not cfg.gcd_db_path.strip():
            warnings.append("GCD enabled but database path not set — configure in Settings")
        elif not Path(cfg.gcd_db_path).exists():
            warnings.append("GCD database file not found at configured path")

    if cfg.comicvine_enabled:
        if not cfg.comicvine_api_key.strip():
            warnings.append("ComicVine enabled but API key not set — configure in Settings")

    _write_event({
        "event": "config_validation",
        "errors": errors,
        "warnings": warnings,
    })


async def _handle_list_dirs(path: str) -> None:
    """List immediate subdirectories of the given path."""
    target = Path(path).expanduser()
    if not target.is_dir():
        _write_event({"event": "dirs", "path": path, "dirs": [], "error": "Not a directory"})
        return

    dirs: list[str] = []
    try:
        for entry in sorted(target.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                dirs.append(entry.name)
    except PermissionError:
        pass

    _write_event({"event": "dirs", "path": path, "dirs": dirs})


async def run_ipc() -> None:
    """Main IPC loop: read JSON commands from stdin, dispatch handlers."""
    cfg = load_config()
    cancel_event = asyncio.Event()
    scan_task: Optional[asyncio.Task] = None

    # Send ready signal
    _write_event({
        "event": "ready",
        "backend_cwd": os.getcwd(),
        "backend_module": __file__,
    })

    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break

        try:
            cmd = json.loads(line.decode().strip())
        except json.JSONDecodeError:
            _write_error(f"Invalid JSON: {line}")
            continue

        command = cmd.get("cmd", "")

        try:
            if command == "scan-preview":
                cfg = load_config()
                await _handle_scan_preview(cfg)

            elif command == "scan":
                cancel_event.clear()
                cfg = load_config()  # reload fresh snapshot
                _write_event({
                    "event": "scan_started",
                    "images_dir": cfg.input_root_dir,
                    "output_path": cfg.output_tsv_path,
                    "resume": bool(cmd.get("resume", False)),
                })
                scan_task = asyncio.create_task(
                    _handle_scan(cfg, cancel_event, resume=bool(cmd.get("resume", False)))
                )

            elif command == "pause":
                cancel_event.set()
                _write_event({"event": "paused"})

            elif command == "resume":
                _write_event({"event": "resumed"})

            elif command == "cancel":
                cancel_event.set()
                _write_event({"event": "cancelled"})

            elif command == "get-config":
                await _handle_get_config(cfg)

            elif command == "set-config":
                cfg = await _handle_set_config(cmd.get("config", {}))
                await _handle_get_config(cfg)

            elif command == "validate-config":
                await _handle_validate_config(cfg)

            elif command == "list-dirs":
                await _handle_list_dirs(cmd.get("path", ""))

            elif command == "quit":
                break

            else:
                _write_error(f"Unknown command: {command}")

        except Exception as exc:
            _write_error(f"Error handling '{command}': {exc}")
            _write_event({"event": "error", "command": command, "message": str(exc)})
