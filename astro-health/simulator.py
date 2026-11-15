"""Simulated mission data.

There is no public live feed of astronaut vitals, so this module generates a
realistic 180-day mission at one sample per hour. It is deterministic (fixed
seed), so everyone on the team sees the same data.

    crew, cabin = generate()

    crew  - dict {astronaut name: DataFrame}, one column per crew parameter
    cabin - DataFrame of shared habitat parameters
Both are indexed by mission hour (0 .. MISSION_DAYS*24 - 1).

Each reading = baseline + daily rhythm + random noise + scripted event effects.
Real records (real_data.py) and live trackers (live_store.py) are layered on
top of this by the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MISSION_NAME = "Lunar orbit habitat – 180-day mission"
MISSION_DAYS = 180
HOURS = MISSION_DAYS * 24
SEED = 2026

# Default start days of the two habitat-wide scripted events.
SPE_DAY = 62.25          # solar particle event
CO2_FAULT_DAY = 95.3     # CO2 scrubber fault (lasts 1.5 days)
# A dose rate above this (mSv/day) counts as a solar particle event in progress.
SPE_THRESHOLD = 5.0
NEUTRON_SHARE = 0.33     # neutrons are roughly a third of the total dose on the ISS


@dataclass(frozen=True)
class Astronaut:
    name: str
    role: str
    resting_hr: float      # bpm
    prior_dose: float      # mSv received on earlier missions
    shielding: float       # multiplier on radiation dose (sleep-station location etc.)
    bone_loss_rate: float  # % per 30 days
    muscle_loss_rate: float


# Fictional crew.
CREW = [
    Astronaut("Asha Verma", "Commander", 61, 310.0, 1.00, 0.45, 1.4),
    Astronaut("Leo Brandt", "Pilot", 63, 95.0, 0.95, 0.35, 1.2),
    Astronaut("Mei Tanaka", "Flight Engineer", 60, 20.0, 1.05, 0.62, 2.2),
    Astronaut("Sam Okafor", "Medical Officer", 66, 140.0, 0.90, 0.40, 1.3),
]


def events(spe_day: float | None = SPE_DAY, co2_fault_day: float | None = CO2_FAULT_DAY):
    """Scripted events as (label, (day, hour), description), for the jump buttons."""
    out = [("Crew illness", (42, 12), "Leo Brandt runs a fever (days 41–44)")]
    if spe_day is not None:
        h = int(spe_day * 24) + 10          # 10 h in = near the peak
        out.append(("Solar particle event", (h // 24, h % 24), f"Simulated solar storm (day {int(spe_day)})"))
    if co2_fault_day is not None:
        h = int(co2_fault_day * 24) + 13
        out.append(("CO2 scrubber fault", (h // 24, h % 24),
                    f"Simulated fault: cabin CO2 passes the limit (day {int(co2_fault_day)})"))
    out += [
        ("Exercise shortfall", (118, 12), "Mei Tanaka injured, exercise drops (days 110–125)"),
        ("Sleep loss", (136, 9), "Sam Okafor sleeps badly (days 130–140)"),
    ]
    return sorted(out, key=lambda e: e[1])


EVENTS = events()


def _daily(values_per_day: np.ndarray) -> np.ndarray:
    """Repeat one value per day across that day's 24 hourly samples."""
    return np.repeat(values_per_day, 24)


def _window(start_day: float, end_day: float) -> np.ndarray:
    """Smooth 0..1 envelope that is 1 between start_day and end_day."""
    d = np.arange(HOURS) / 24.0
    ramp = 0.5  # days
    up = np.clip((d - start_day) / ramp, 0, 1)
    down = np.clip((end_day - d) / ramp, 0, 1)
    return np.minimum(up, down)


def _spe_rate(start_day: float | None) -> np.ndarray:
    """Extra dose rate (mSv/day) from a solar particle event starting on start_day."""
    rate = np.zeros(HOURS)
    if start_day is None:
        return rate
    h = np.arange(HOURS, dtype=float)
    start = start_day * 24
    rise_hours, decay_hours, peak = 5.0, 9.0, 330.0
    rising = (h >= start) & (h < start + rise_hours)
    rate[rising] = peak * (h[rising] - start) / rise_hours
    falling = h >= start + rise_hours
    rate[falling] = peak * np.exp(-(h[falling] - start - rise_hours) / decay_hours)
    rate[rate < 0.01] = 0.0
    return rate


def radiation_columns(habitat_rate: np.ndarray, astronaut: Astronaut) -> dict[str, np.ndarray]:
    """All radiation parameters for one astronaut from the habitat dose rate (mSv/day).

    Used for both simulated and real dose rates, so the two are treated alike.
    """
    dose_rate = np.asarray(habitat_rate, dtype=float) * astronaut.shielding
    hourly = dose_rate / 24.0
    # Event dose: dose accumulated during each continuous spell above the threshold.
    active = np.asarray(habitat_rate) > SPE_THRESHOLD
    spell = np.cumsum(active & ~np.roll(active, 1) | (np.arange(len(active)) == 0) & active)
    in_spell = pd.Series(np.where(active, hourly, 0.0)).groupby(spell).cumsum().to_numpy()
    return {
        "dose_rate": dose_rate,
        "career_dose": astronaut.prior_dose + np.cumsum(hourly),
        "spe_dose": np.where(active, in_spell, 0.0),
        "neutron_rate": dose_rate * NEUTRON_SHARE,
    }


def generate(spe_day: float | None = SPE_DAY, co2_fault_day: float | None = CO2_FAULT_DAY
             ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build the simulated mission. Pass None for an event day to leave that event out."""
    rng = np.random.default_rng(SEED)
    rng_extra = np.random.default_rng(SEED + 1)   # for parameters added later, so older data is unchanged
    h = np.arange(HOURS)
    day = h / 24.0
    circadian = np.sin(2 * np.pi * (h % 24 - 15) / 24)  # peaks mid-afternoon
    spe = _spe_rate(spe_day)

    # ---- shared cabin environment -------------------------------------------------
    co2_fault = _window(co2_fault_day, co2_fault_day + 1.5) if co2_fault_day is not None else np.zeros(HOURS)
    cabin = pd.DataFrame(
        {
            "co2": 2.15 + 0.18 * rng.standard_normal(HOURS) + 2.1 * co2_fault,
            "o2": 150 + 2.5 * rng.standard_normal(HOURS) - 6 * co2_fault,
            "cabin_temp": 22.4 + 0.5 * circadian + 0.4 * rng.standard_normal(HOURS),
            "cabin_pressure": 101.3 + 0.35 * rng.standard_normal(HOURS),
            "humidity": 45 + 4 * rng.standard_normal(HOURS) + 9 * co2_fault,
            "noise": 61.5 + 1.2 * rng.standard_normal(HOURS),
            # Resupply arrives every 60 days; margin counts down in between.
            "med_supplies": 75 - (day % 60) * 1.05,
            "comms_delay": 1.3 + 0.02 * rng.standard_normal(HOURS),
        },
        index=pd.RangeIndex(HOURS, name="hour"),
    )

    crew: dict[str, pd.DataFrame] = {}
    for a in CREW:
        n_days = MISSION_DAYS
        ill = _window(41, 44.5) if a.name == "Leo Brandt" else np.zeros(HOURS)
        injured = _window(110, 125) if a.name == "Mei Tanaka" else np.zeros(HOURS)
        sleepless = _window(130, 140) if a.name == "Sam Okafor" else np.zeros(HOURS)

        # Daily quantities (one value per day).
        sleep = _daily(np.clip(7.45 + 0.45 * rng.standard_normal(n_days), 4.0, 9.0)) - 1.9 * sleepless
        sleep_eff = _daily(90 + 2.8 * rng.standard_normal(n_days)) - 12 * sleepless
        exercise = _daily(np.clip(146 + 13 * rng.standard_normal(n_days), 60, 190)) * (1 - 0.55 * injured)
        hydration = _daily(2.35 + 0.22 * rng.standard_normal(n_days)) - 0.5 * ill
        calories = _daily(100 + 2.8 * rng.standard_normal(n_days)) - 14 * ill
        # Mood dips in the third quarter of the mission, a known isolation effect.
        third_quarter = np.exp(-(((day - 125) / 28) ** 2))
        mood = _daily(8.0 + 0.7 * rng.standard_normal(n_days)) - 1.3 * third_quarter - 2.3 * sleepless - 1.5 * ill

        # Radiation: habitat background plus any solar particle event.
        habitat_rate = 0.5 + 0.04 * rng.standard_normal(HOURS) + spe

        # Slow deconditioning trends (% change from pre-flight).
        months = day / 30.0
        bmd = -a.bone_loss_rate * months + 0.15 * rng.standard_normal(HOURS).cumsum() / np.sqrt(HOURS)
        muscle = -a.muscle_loss_rate * months - 3.0 * np.cumsum(injured) / (24 * 15)
        aerobic = -9.0 * (1 - np.exp(-day / 25)) + 0.5 * months * (a.muscle_loss_rate < 2) - 4.0 * np.cumsum(injured) / (24 * 15)
        body_mass = -0.5 * months - 0.8 * np.cumsum(ill) / (24 * 3.5)

        df = pd.DataFrame(
            {
                "heart_rate": a.resting_hr + 4 * circadian + 2.5 * rng.standard_normal(HOURS) + 34 * ill + 5 * sleepless,
                "spo2": np.clip(97.6 + 0.5 * rng.standard_normal(HOURS) - 3.2 * ill - 1.5 * co2_fault, 80, 100),
                "bp_systolic": 123 + 3 * circadian + 3.5 * rng.standard_normal(HOURS) - 9 * ill,
                "core_temp": 36.8 + 0.25 * circadian + 0.08 * rng.standard_normal(HOURS) + 1.9 * ill,
                "resp_rate": 15 + 1.1 * rng.standard_normal(HOURS) + 10.5 * ill + 2.5 * co2_fault,
                "hrv_pct": 100 + 6 * rng.standard_normal(HOURS) - 30 * ill - 24 * sleepless,
                **radiation_columns(habitat_rate, a),
                "exercise_min": exercise,
                "bmd_change": bmd,
                "muscle_change": muscle,
                "aerobic_change": aerobic,
                "body_mass_change": body_mass,
                "iop": 15.5 + 0.02 * day * (a.name == "Asha Verma") * 1.9 + 0.8 * rng.standard_normal(HOURS),
                "sleep_hours": sleep,
                "sleep_efficiency": np.clip(sleep_eff, 40, 99),
                "reaction_time": 275 + 38 * (7.4 - sleep) + 14 * rng.standard_normal(HOURS) + 35 * co2_fault,
                "mood": np.clip(np.round(mood), 1, 10),
                # Eye closure rises after short nights (the camera measures this for real).
                "perclos": np.clip(3.5 + 2.6 * np.clip(7.0 - sleep, 0, None) + 0.9 * rng_extra.standard_normal(HOURS), 0.5, 40),
                "hydration": np.clip(hydration, 0.5, 4),
                "calorie_pct": np.clip(calories, 40, 120),
            },
            index=pd.RangeIndex(HOURS, name="hour"),
        )
        crew[a.name] = df
    return crew, cabin


def fmt_time(hour: int) -> str:
    """Mission elapsed time as text, e.g. 'Day 62, 16:00'."""
    return f"Day {hour // 24}, {hour % 24:02d}:00"


if __name__ == "__main__":
    crew_data, cabin_data = generate()
    print(cabin_data.describe().T[["min", "mean", "max"]].round(2))
    for name, frame in crew_data.items():
        print(f"\n{name}")
        print(frame.describe().T[["min", "mean", "max"]].round(2))
