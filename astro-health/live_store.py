"""Live readings from trackers: the one place every device writes to.

Any source (Bluetooth heart-rate strap, phone, script, the camera) calls

    live_store.write("Asha Verma", {"heart_rate": 72}, source="polar-h10")

and the dashboard picks the reading up within a couple of seconds, shows it in
place of the simulated value with a LIVE tag, and classifies it green / yellow /
red like any other value.

How it is stored (plain files in live/, so separate programs can share them):
    latest_<source>.json    the newest reading from each source
    history_<source>.csv    every reading, for the live charts
A reading stops counting as live once it is older than its time-to-live (ttl),
after which the dashboard falls back to the simulated value.
"""
from __future__ import annotations

import csv
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

LIVE_DIR = Path(__file__).parent / "live"
DEFAULT_TTL_S = 15.0        # continuous trackers: stale after 15 s of silence
MANUAL_TTL_S = 120.0        # typed-in or one-off readings stay for two minutes
HISTORY_WINDOW_S = 300.0    # live charts show the last five minutes
_write_lock = threading.Lock()   # the HTTP receiver handles several requests at once


@dataclass(frozen=True)
class Reading:
    astronaut: str
    key: str
    value: float
    timestamp: float        # seconds since 1970 (time.time())
    source: str

    @property
    def age(self) -> float:
        return time.time() - self.timestamp


def _safe(name: str) -> str:
    """Source names become part of a file name, so keep them simple."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")[:40] or "source"


def _replace(tmp: Path, target: Path) -> None:
    """Atomic rename. Windows can refuse while another program reads the file, so retry briefly."""
    for attempt in range(5):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            time.sleep(0.02 * (attempt + 1))
    tmp.unlink(missing_ok=True)   # give up on this one update; the next will go through


def write(astronaut: str, readings: dict[str, float], source: str,
          ttl: float = DEFAULT_TTL_S, live_dir: Path = LIVE_DIR) -> None:
    """Record one or more parameter values for an astronaut. Safe to call many times a second."""
    live_dir.mkdir(exist_ok=True)
    now = time.time()
    clean = {k: float(v) for k, v in readings.items()}
    name = _safe(source)

    latest_path = live_dir / f"latest_{name}.json"
    with _write_lock:
        # Keep this source's other parameters (a device may send heart rate and SpO2 separately).
        newest = _load_json(latest_path) or {"source": source, "readings": {}}
        for key, value in clean.items():
            newest["readings"][f"{astronaut}|{key}"] = {"value": value, "timestamp": now, "ttl": ttl}
        tmp = latest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(newest), encoding="utf-8")
        _replace(tmp, latest_path)

        with (live_dir / f"history_{name}.csv").open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for key, value in clean.items():
                writer.writerow([f"{now:.2f}", astronaut, key, value])


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def latest(live_dir: Path = LIVE_DIR) -> dict[tuple[str, str], Reading]:
    """Every reading that is still live, keyed by (astronaut, parameter key).

    If two sources report the same parameter for the same astronaut, the newest wins.
    """
    out: dict[tuple[str, str], Reading] = {}
    if not live_dir.exists():
        return out
    now = time.time()
    for path in live_dir.glob("latest_*.json"):
        data = _load_json(path)
        if not data:
            continue
        for combined, item in data.get("readings", {}).items():
            astronaut, _, key = combined.partition("|")
            if now - item["timestamp"] > item.get("ttl", DEFAULT_TTL_S):
                continue
            reading = Reading(astronaut, key, item["value"], item["timestamp"], data.get("source", path.stem))
            current = out.get((astronaut, key))
            if current is None or reading.timestamp > current.timestamp:
                out[(astronaut, key)] = reading
    return out


def history(astronaut: str, key: str, window_s: float = HISTORY_WINDOW_S,
            live_dir: Path = LIVE_DIR) -> list[tuple[float, float]]:
    """(timestamp, value) pairs for one astronaut and parameter over the last few minutes."""
    points: list[tuple[float, float]] = []
    if not live_dir.exists():
        return points
    cutoff = time.time() - window_s
    for path in live_dir.glob("history_*.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as f:
                # Only the tail matters; these files are small enough to scan in a hackathon.
                for row in csv.reader(f):
                    if len(row) == 4 and row[1] == astronaut and row[2] == key and float(row[0]) >= cutoff:
                        points.append((float(row[0]), float(row[3])))
        except (OSError, ValueError):
            continue
    return sorted(points)


def sources(live_dir: Path = LIVE_DIR) -> list[dict]:
    """One row per (source, astronaut) that is currently live, for the Trackers page."""
    grouped: dict[tuple[str, str], dict] = {}
    for (astronaut, key), r in latest(live_dir).items():
        row = grouped.setdefault((r.source, astronaut), {"Source": r.source, "Crew member": astronaut,
                                                       "Parameters": [], "_newest": 0.0})
        row["Parameters"].append(key)
        row["_newest"] = max(row["_newest"], r.timestamp)
    rows = []
    for row in grouped.values():
        rows.append({"Source": row["Source"], "Crew member": row["Crew member"],
                     "Parameters": ", ".join(sorted(row["Parameters"])),
                     "Last reading": f"{max(0, time.time() - row['_newest']):.0f} s ago"})
    return sorted(rows, key=lambda r: (r["Source"], r["Crew member"]))


def clear(live_dir: Path = LIVE_DIR) -> None:
    """Forget all live readings and history (the Trackers page has a button for this)."""
    if live_dir.exists():
        for path in list(live_dir.glob("latest_*.json")) + list(live_dir.glob("history_*.csv")):
            path.unlink(missing_ok=True)
