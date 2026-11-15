"""Threshold loading and green / yellow / red classification.

All limits live in thresholds.csv, so changing a band never means touching code.

Each parameter has up to four numbers:
    red_low  yellow_low  yellow_high  red_high
        value <= red_low   or  value >= red_high    -> RED
        value <  yellow_low or value >  yellow_high -> YELLOW
        otherwise                                   -> GREEN
A blank cell means "no limit on that side". A row with all four blank is
info-only (shown, never alarmed).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

GREEN, YELLOW, RED, INFO = "green", "yellow", "red", "info"
SEVERITY = {INFO: 0, GREEN: 0, YELLOW: 1, RED: 2}

# Colour is never the only signal: every status also has an icon and a word.
STATUS_STYLE = {
    GREEN: {"label": "OK", "icon": "✔", "color": "#0ca30c", "text": "#ffffff"},
    YELLOW: {"label": "WATCH", "icon": "▲", "color": "#fab219", "text": "#0b0b0b"},
    RED: {"label": "ALERT", "icon": "✖", "color": "#d03b3b", "text": "#ffffff"},
    INFO: {"label": "INFO", "icon": "•", "color": "#8a8a85", "text": "#ffffff"},
}

THRESHOLDS_FILE = Path(__file__).parent / "thresholds.csv"
_BOUNDS = ["red_low", "yellow_low", "yellow_high", "red_high"]


@dataclass(frozen=True)
class Param:
    key: str
    name: str
    unit: str
    hazard: str
    scope: str  # "crew", "cabin" or "camera"
    red_low: float | None
    yellow_low: float | None
    yellow_high: float | None
    red_high: float | None
    decimals: int
    source: str

    @property
    def info_only(self) -> bool:
        return all(getattr(self, b) is None for b in _BOUNDS)

    def classify(self, value: float | None) -> str:
        if value is None or pd.isna(value) or self.info_only:
            return INFO
        if self.red_low is not None and value <= self.red_low:
            return RED
        if self.red_high is not None and value >= self.red_high:
            return RED
        if self.yellow_low is not None and value < self.yellow_low:
            return YELLOW
        if self.yellow_high is not None and value > self.yellow_high:
            return YELLOW
        return GREEN

    def fmt(self, value: float | None) -> str:
        if value is None or pd.isna(value):
            return "–"
        return f"{value:.{self.decimals}f}"

    def green_range_text(self) -> str:
        """Human-readable description of the green band, e.g. '51 to 90'."""
        lo, hi = self.yellow_low, self.yellow_high
        if self.info_only:
            return "no limit"
        if lo is not None and hi is not None:
            return f"{lo:g} to {hi:g}"
        if lo is not None:
            return f"at least {lo:g}"
        if hi is not None:
            return f"up to {hi:g}"
        return "see red limits"


def load_params(path: Path = THRESHOLDS_FILE) -> dict[str, Param]:
    df = pd.read_csv(path)
    params: dict[str, Param] = {}
    for row in df.to_dict("records"):
        bounds = {b: (None if pd.isna(row[b]) else float(row[b])) for b in _BOUNDS}
        params[row["key"]] = Param(
            key=row["key"], name=row["name"], unit=row["unit"], hazard=row["hazard"],
            scope=row["scope"], decimals=int(row["decimals"]), source=row["source"], **bounds,
        )
    return params


def worst(statuses) -> str:
    """Overall status = the most severe individual status."""
    result = GREEN
    for s in statuses:
        if SEVERITY[s] > SEVERITY[result]:
            result = s
    return result
