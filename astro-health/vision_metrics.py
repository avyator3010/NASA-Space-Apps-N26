"""Maths for the camera module. Pure numpy - no camera or MediaPipe needed,
so it can be unit-tested anywhere (see tests/test_core.py).

Two measurements:

1. FATIGUE from the eyes.
   Eye aspect ratio (EAR) = eye height / eye width from six landmarks per eye.
   Open eye ~0.25-0.35, closed eye < ~0.2.
   PERCLOS = % of the last minute during which the eyes were closed.

2. HEART RATE without contact (remote photoplethysmography, rPPG).
   Each heartbeat pushes blood into the skin and changes how much green light
   it reflects by a tiny amount. Average the green channel over the face, then
   find the strongest frequency between 45 and 180 beats per minute.
   EXPERIMENTAL: needs steady light and a still face.
"""
from __future__ import annotations

from collections import deque

import numpy as np

# MediaPipe face-mesh landmark indices: [outer corner, top 1, top 2, inner corner, bottom 2, bottom 1]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """pts: 6x2 array of eye landmarks in the order given above."""
    pts = np.asarray(pts, dtype=float)
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return float((v1 + v2) / (2.0 * h)) if h > 0 else 0.0


class FatigueTracker:
    """Turns a stream of EAR values into PERCLOS, blink rate and closure time."""

    def __init__(self, window_s: float = 60.0, calibration_s: float = 5.0,
                 closed_fraction: float = 0.72, default_threshold: float = 0.20):
        self.window_s = window_s
        self.calibration_s = calibration_s
        self.closed_fraction = closed_fraction
        self.threshold = default_threshold
        self.calibrated = False
        self._calib: list[float] = []
        self._t0: float | None = None
        self._samples: deque[tuple[float, bool]] = deque()   # (time, closed?)
        self._blinks: deque[float] = deque()                 # times of completed blinks
        self._closed_since: float | None = None

    def update(self, t: float, ear: float) -> None:
        if self._t0 is None:
            self._t0 = t
        # First few seconds: learn this person's open-eye EAR and set the
        # "closed" threshold relative to it (eye shapes differ a lot).
        if not self.calibrated:
            self._calib.append(ear)
            if t - self._t0 >= self.calibration_s and len(self._calib) >= 10:
                self.threshold = self.closed_fraction * float(np.median(self._calib))
                self.calibrated = True

        closed = ear < self.threshold
        if closed and self._closed_since is None:
            self._closed_since = t
        elif not closed and self._closed_since is not None:
            if t - self._closed_since < 1.0:      # short closure = a blink
                self._blinks.append(t)
            self._closed_since = None

        self._samples.append((t, closed))
        while self._samples and t - self._samples[0][0] > self.window_s:
            self._samples.popleft()
        while self._blinks and t - self._blinks[0] > self.window_s:
            self._blinks.popleft()

    @property
    def perclos(self) -> float:
        """Percentage of the window with eyes closed (0-100)."""
        if not self._samples:
            return 0.0
        return 100.0 * sum(c for _, c in self._samples) / len(self._samples)

    @property
    def blink_rate(self) -> float:
        """Blinks per minute, scaled up if less than a full window has passed."""
        if not self._samples:
            return 0.0
        span = max(self._samples[-1][0] - self._samples[0][0], 1.0)
        return len(self._blinks) * 60.0 / min(span, self.window_s)

    def closed_for(self, t: float) -> float:
        """Seconds the eyes have been continuously closed (0 if open)."""
        return 0.0 if self._closed_since is None else t - self._closed_since


def estimate_heart_rate(times, values, fs: float = 30.0, lo_bpm: float = 45.0,
                        hi_bpm: float = 180.0, min_seconds: float = 8.0):
    """Estimate pulse from a green-channel trace.

    times  - sample times in seconds (need not be evenly spaced)
    values - mean green intensity of the skin region at each time
    Returns (bpm, quality). quality is 0-1: the share of in-band signal power
    sitting at the pulse peak. Treat quality < 0.3 as unreliable.
    Returns (None, 0.0) if there is not enough data yet.
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(t) < 30 or t[-1] - t[0] < min_seconds:
        return None, 0.0

    # Webcams drop frames, so resample onto an even time grid first.
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    x = np.interp(grid, t, v)
    # Remove slow drift (lighting, movement) with a quadratic fit.
    idx = np.arange(len(x))
    x = x - np.polyval(np.polyfit(idx, x, 2), idx)
    if np.std(x) < 1e-9:
        return None, 0.0
    x = (x / np.std(x)) * np.hanning(len(x))

    n_fft = 1 << int(np.ceil(np.log2(len(x) * 8)))   # zero-pad for a finer frequency grid
    power = np.abs(np.fft.rfft(x, n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    band = (freqs >= lo_bpm / 60.0) & (freqs <= hi_bpm / 60.0)
    if not band.any() or power[band].sum() <= 0:
        return None, 0.0

    peak = int(np.argmax(np.where(band, power, 0.0)))
    near_peak = band & (np.abs(freqs - freqs[peak]) <= 0.12)
    quality = float(power[near_peak].sum() / power[band].sum())
    return float(freqs[peak] * 60.0), quality
