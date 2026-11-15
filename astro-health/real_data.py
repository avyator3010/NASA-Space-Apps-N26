"""Replace simulated radiation and cabin data with real ISS records.

The records come from NASA's Environmental Data Application (run
fetch_real_data.py to download them into data/). This module:

  1. reads a mission's files and converts them to the dashboard's units,
  2. lines the record up with the mission clock (mission day 0 = first day of the record),
  3. overwrites the simulated values wherever a real reading exists,
  4. keeps a "coverage" mask so the dashboard can label every value REAL or SIMULATED.

    mission = build_mission("RR-7")     # or build_mission(None) for all-simulated
    mission.crew, mission.cabin, mission.coverage, mission.events

WHAT IS REAL AFTER THIS
    cabin temperature, humidity, CO2          measured on the ISS
    dose rate, neutron estimate, event dose   derived from the measured daily dose
Everything else (vitals, sleep, exercise, O2, pressure, noise ...) stays simulated.

UNIT CONVERSIONS (all assumptions are constants below, so they are easy to change)
    CO2      NASA gives ppm. Partial pressure in mmHg = ppm x cabin pressure / 1e6,
             with the ISS's sea-level cabin pressure of 760 mmHg.
    Dose     NASA gives ABSORBED dose (mGy/day), split into cosmic rays (GCR) and
             the South Atlantic Anomaly (SAA). Health limits are in sieverts, so
             each part is multiplied by a quality factor measured by a NASA
             tissue-equivalent counter in low Earth orbit (Badhwar et al., STS-40):
             2.74 for GCR and 1.83 for trapped (SAA) radiation.
             The result is a dose EQUIVALENT, used here as an estimate of the
             effective dose the limits are written in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import simulator

DATA_DIR = Path(__file__).parent / "data"

CABIN_PRESSURE_MMHG = 760.0
Q_GCR = 2.74          # quality factor, galactic cosmic rays
Q_SAA = 1.83          # quality factor, trapped protons (South Atlantic Anomaly)
Q_OVERALL = 2.38      # used only if a file has a total but no GCR/SAA split
MAX_GAP_HOURS = 3     # gaps up to this long are bridged; longer ones stay simulated

CABIN_KEYS = ["cabin_temp", "humidity", "co2"]
RADIATION_KEYS = ["dose_rate", "neutron_rate", "spe_dose"]
# Real accumulation on top of each (fictional) astronaut's simulated earlier missions.
PARTLY_REAL_KEYS = ["career_dose"]


@dataclass
class RealRecord:
    mission: str
    info: dict
    origin: pd.Timestamp                 # real date and time of mission hour 0
    cabin: pd.DataFrame                  # index = mission hour; NaN where there is no reading
    habitat_dose: pd.Series              # mSv/day by mission hour; NaN where there is no reading
    notes: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.info.get("title") or self.mission

    def real_time(self, hour: int) -> pd.Timestamp:
        return self.origin + pd.Timedelta(hours=int(hour))

    def covered_days(self, series: pd.Series) -> tuple[int, int] | None:
        hours = series.dropna().index
        return None if len(hours) == 0 else (int(hours.min() // 24), int(hours.max() // 24))


@dataclass
class Mission:
    crew: dict[str, pd.DataFrame]
    cabin: pd.DataFrame
    events: list
    coverage: dict[str, np.ndarray]      # parameter key -> True where the value is a real record
    record: RealRecord | None = None

    def source(self, key: str, hour: int) -> str:
        """'real', 'part' or 'sim' for one parameter at one mission hour."""
        mask = self.coverage.get(key)
        if mask is None or not mask[hour]:
            return "sim"
        return "part" if key in PARTLY_REAL_KEYS else "real"


# ------------------------------------------------------------------------- reading files
def available(data_dir: Path = DATA_DIR) -> list[str]:
    """Missions that have at least one data file in data/."""
    if not data_dir.exists():
        return []
    names = {p.name.rsplit("_", 1)[0] for p in data_dir.glob("*_telemetry.csv")}
    names |= {p.name.rsplit("_", 1)[0] for p in data_dir.glob("*_radiation.csv")}
    return sorted(names)


def _find(columns, *needles) -> str | None:
    """First column whose name contains all the given fragments (case-insensitive)."""
    for col in columns:
        low = col.lower()
        if all(n in low for n in needles):
            return col
    return None


def _read_telemetry(path: Path) -> pd.DataFrame:
    """Hourly means of cabin temperature (deg C), humidity (%) and CO2 (mmHg), indexed by real time."""
    raw = pd.read_csv(path)
    time_col = _find(raw.columns, "time") or raw.columns[0]
    # Prefer NASA's "_ISS" columns; the same table also has ground-control columns.
    cols = {
        "cabin_temp": _find(raw.columns, "temp", "iss") or _find(raw.columns, "temp"),
        "humidity": _find(raw.columns, "rh", "iss") or _find(raw.columns, "rh"),
        "co2_ppm": _find(raw.columns, "co2", "iss") or _find(raw.columns, "co2"),
    }
    out = pd.DataFrame({"time": pd.to_datetime(raw[time_col], errors="coerce")})
    for key, col in cols.items():
        # "none" and any other text become NaN (missing).
        out[key] = pd.to_numeric(raw[col], errors="coerce") if col else np.nan
    out = out.dropna(subset=["time"]).set_index("time").sort_index()
    hourly = out.resample("1h").mean()
    hourly = hourly.interpolate(limit=MAX_GAP_HOURS, limit_area="inside")
    hourly["co2"] = hourly.pop("co2_ppm") * CABIN_PRESSURE_MMHG / 1e6
    return hourly.dropna(how="all")


def _read_radiation(path: Path) -> pd.Series:
    """Daily dose-equivalent rate (mSv/day), indexed by real date."""
    raw = pd.read_csv(path)
    date_col = _find(raw.columns, "date") or _find(raw.columns, "time") or raw.columns[0]
    dates = pd.to_datetime(raw[date_col], errors="coerce").dt.normalize()
    gcr, saa, total = (_find(raw.columns, n) for n in ("gcr", "saa", "total"))
    if gcr and saa:
        dose = pd.to_numeric(raw[gcr], errors="coerce") * Q_GCR + pd.to_numeric(raw[saa], errors="coerce") * Q_SAA
    elif total:
        dose = pd.to_numeric(raw[total], errors="coerce") * Q_OVERALL
    else:
        raise ValueError(f"{path.name}: expected GCR and SAA columns, or a Total column")
    series = pd.Series(dose.to_numpy(), index=dates).dropna()
    return series[series.index.notna()].sort_index()


def load_record(mission: str, data_dir: Path = DATA_DIR, hours: int = simulator.HOURS) -> RealRecord:
    tel_path, rad_path = data_dir / f"{mission}_telemetry.csv", data_dir / f"{mission}_radiation.csv"
    info_path = data_dir / f"{mission}_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    notes: list[str] = []

    telemetry = _read_telemetry(tel_path) if tel_path.exists() else pd.DataFrame(columns=CABIN_KEYS)
    radiation = _read_radiation(rad_path) if rad_path.exists() else pd.Series(dtype=float)
    if telemetry.empty and radiation.empty:
        raise FileNotFoundError(f"No usable data for {mission} in {data_dir}")

    # Mission hour 0 = midnight on the first day of the record.
    starts = [s.index.min().normalize() for s in (telemetry, radiation) if len(s)]
    origin = min(starts)
    window_end = origin + pd.Timedelta(hours=hours)
    if len(telemetry) and len(radiation):
        tel_start, tel_end = telemetry.index.min(), telemetry.index.max()
        if radiation.index.max() < tel_start.normalize() or radiation.index.min() > tel_end:
            # e.g. RR-9, where NASA's radiation table is dated a year before the telemetry.
            notes.append("Radiation record ignored: its dates do not overlap the cabin record.")
            radiation = radiation.iloc[0:0]
            origin = tel_start.normalize()
            window_end = origin + pd.Timedelta(hours=hours)

    index = pd.RangeIndex(hours, name="hour")
    cabin = pd.DataFrame(np.nan, index=index, columns=CABIN_KEYS)
    if len(telemetry):
        tel = telemetry[(telemetry.index >= origin) & (telemetry.index < window_end)]
        hour_idx = ((tel.index - origin) / pd.Timedelta(hours=1)).astype(int)
        for key in CABIN_KEYS:
            cabin.loc[hour_idx, key] = tel[key].to_numpy()
        if telemetry.index.max() >= window_end:
            notes.append(f"Record is longer than the mission; only the first {hours // 24} days are used.")

    habitat = pd.Series(np.nan, index=index, name="habitat_dose")
    for date, value in radiation.items():      # one value per day, held for that day's 24 hours
        start_hour = int((date - origin) / pd.Timedelta(hours=1))
        if 0 <= start_hour < hours:
            habitat.iloc[start_hour:start_hour + 24] = value
    return RealRecord(mission=mission, info=info, origin=origin, cabin=cabin, habitat_dose=habitat, notes=notes)


# ------------------------------------------------------------------------- applying
def apply_real_record(record: RealRecord, crew: dict[str, pd.DataFrame], cabin: pd.DataFrame
                      ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, np.ndarray]]:
    """Overwrite simulated values with real ones wherever the record has a reading.

    Returns new (crew, cabin, coverage); the inputs are not modified.
    """
    coverage: dict[str, np.ndarray] = {}
    cabin = cabin.copy()
    for key in CABIN_KEYS:
        mask = record.cabin[key].notna().to_numpy()
        cabin[key] = np.where(mask, record.cabin[key].to_numpy(), cabin[key].to_numpy())
        coverage[key] = mask

    rad_mask = record.habitat_dose.notna().to_numpy()
    new_crew = {}
    for astronaut in simulator.CREW:
        df = crew[astronaut.name].copy()
        if rad_mask.any():
            simulated_habitat = df["dose_rate"].to_numpy() / astronaut.shielding
            habitat = np.where(rad_mask, record.habitat_dose.to_numpy(), simulated_habitat)
            for key, values in simulator.radiation_columns(habitat, astronaut).items():
                df[key] = values
        new_crew[astronaut.name] = df
    for key in RADIATION_KEYS + PARTLY_REAL_KEYS:
        coverage[key] = rad_mask
    return new_crew, cabin, coverage


def _clear_day(mask: np.ndarray, default_day: float, length_days: float) -> float | None:
    """Keep a scripted event where it is, or move it past the real record if they would collide."""
    lo, hi = int(default_day * 24), int((default_day + length_days) * 24)
    if not mask[lo:hi].any():
        return default_day
    moved = np.flatnonzero(mask).max() / 24 + 15
    return float(np.floor(moved)) + 0.25 if moved + length_days < simulator.MISSION_DAYS - 5 else None


def _real_events(record: RealRecord) -> list:
    """Jump-button entries for the most notable moments in the real record."""
    out = []
    co2 = record.cabin["co2"].dropna()
    if len(co2):
        h = int(co2.idxmax())
        out.append(("Real record: highest CO2", (h // 24, h % 24),
                    f"Measured on the ISS, {record.real_time(h):%d %b %Y}: {co2.max():.1f} mmHg"))
    dose = record.habitat_dose.dropna()
    if len(dose):
        h = int(dose.idxmax()) + 12
        out.append(("Real record: highest radiation", (h // 24, h % 24),
                    f"Measured on the ISS, {record.real_time(h):%d %b %Y}: about {dose.max():.2f} mSv/day"))
    return out


def build_mission(mission: str | None = None, data_dir: Path = DATA_DIR) -> Mission:
    """The whole data set for the dashboard: simulated, or simulated with a real record on top."""
    if not mission:
        crew, cabin = simulator.generate()
        return Mission(crew=crew, cabin=cabin, events=simulator.events(), coverage={})

    record = load_record(mission, data_dir)
    # The scripted solar storm and CO2 fault are simulated. If the real record covers
    # the days they would fall on, move them to after the record so the two never mix.
    spe_day = _clear_day(record.habitat_dose.notna().to_numpy(), simulator.SPE_DAY, 3)
    co2_day = _clear_day(record.cabin["co2"].notna().to_numpy(), simulator.CO2_FAULT_DAY, 2)
    crew, cabin = simulator.generate(spe_day=spe_day, co2_fault_day=co2_day)
    crew, cabin, coverage = apply_real_record(record, crew, cabin)
    events = sorted(simulator.events(spe_day, co2_day) + _real_events(record), key=lambda e: e[1])
    return Mission(crew=crew, cabin=cabin, events=events, coverage=coverage, record=record)


def describe(record: RealRecord) -> str:
    """One-line summary for the dashboard header."""
    parts = []
    cab = record.covered_days(record.cabin["co2"])
    rad = record.covered_days(record.habitat_dose)
    if cab:
        parts.append(f"cabin temperature, humidity and CO2 on mission days {cab[0]}–{cab[1]}")
    if rad:
        parts.append(f"radiation on days {rad[0]}–{rad[1]}")
    return (f"Real ISS record: {record.title}, starting {record.origin:%d %b %Y}. "
            f"Real data: {' and '.join(parts)}. Everything else is simulated.")


if __name__ == "__main__":
    found = available()
    if not found:
        print("No real records in data/. Run:  python fetch_real_data.py")
    for name in found:
        rec = load_record(name)
        print(describe(rec))
        for note in rec.notes:
            print("  note:", note)
        print(rec.cabin.describe().T[["count", "min", "mean", "max"]].round(2))
        print(f"  dose rate (mSv/day): {rec.habitat_dose.min():.2f} to {rec.habitat_dose.max():.2f}")
