"""Download real ISS environment records from NASA.

    python fetch_real_data.py              # downloads the default mission (RR-7)
    python fetch_real_data.py RR-1 RR-6    # other missions
    python fetch_real_data.py --list       # show the missions NASA offers

Source: NASA Open Science Data Repository, Environmental Data Application (EDA)
        https://visualization.osdr.nasa.gov/eda/
It holds cabin telemetry (temperature, relative humidity, CO2) and daily
radiation dose recorded on the International Space Station during Rodent
Research missions.

For each mission this writes into the data/ folder:
    <MISSION>_telemetry.csv   Controller_Time_GMT, Temp_degC_ISS, RH_percent_ISS, CO2_ppm_ISS
    <MISSION>_radiation.csv   Date, GCR_mGy_per_day, SAA_mGy_per_day, Total_mGy_per_day
    <MISSION>_info.json       mission title, NASA link, download date
The values are saved exactly as NASA publishes them; real_data.py does the
unit conversion when the dashboard loads them.

Needs only the Python standard library.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

EDA_URL = "https://visualization.osdr.nasa.gov/eda/"
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MISSION = "RR-7"   # longest record with both cabin and radiation data (about 80 days)

# Missions listed in the EDA when this was written (November 2026), and what each holds.
MISSIONS = {
    "RR-1": "cabin + radiation, Sep-Oct 2014", "RR-3": "cabin + radiation, Apr-May 2016",
    "RR-4": "radiation only, Feb-Mar 2017", "RR-5": "cabin only, Jun-Aug 2017",
    "RR-6": "cabin + radiation, Dec 2017-Feb 2018", "RR-7": "cabin + radiation, Jun-Sep 2018",
    "RR-8": "radiation only, Dec 2018-Jan 2019", "RR-9": "cabin Aug-Sep 2018 (its radiation table is dated 2017, so only cabin data is used)",
    "RR-10": "cabin only, Dec 2020-Jan 2021", "RR-12": "cabin + radiation, Apr-May 2019",
    "RR-17": "radiation only, Jul-Sep 2019", "RR-18": "cabin only, Dec 2021-Aug 2022",
    "RR-19": "cabin + radiation, Dec 2019-Jan 2020", "RR-20": "cabin only, Nov 2023-Apr 2024",
    "RR-23": "cabin + radiation, Dec 2020-Jan 2021", "RR-25": "cabin only, Nov 2022-Feb 2023",
    "RR-28": "cabin only, Mar-May 2024",
}

_DATA_TAG = re.compile(r'<script[^>]*\bid="data"[^>]*>(.*?)</script>', re.DOTALL)


def _number(value):
    """NASA marks a missing reading with the text 'none'. Keep numbers, blank the rest."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else ""


def parse_eda_page(html: str) -> dict:
    """Pull the data out of an EDA mission page.

    The page embeds everything in a <script id="data" type="application/json"> tag.
    Inside, 'telemetry_data' and 'radiation_data' are themselves JSON text.
    Returns {"telemetry": rows, "radiation": rows, "info": {...}}; either list may be empty.
    """
    match = _DATA_TAG.search(html)
    if not match:
        raise ValueError("No embedded data found - NASA may have changed the page layout.")
    payload = json.loads(match.group(1))

    def inner(key):
        raw = payload.get(key)
        if not raw:
            return {}
        return json.loads(raw) if isinstance(raw, str) else raw

    tel, rad = inner("telemetry_data"), inner("radiation_data")
    telemetry = [
        {"Controller_Time_GMT": t, "Temp_degC_ISS": _number(temp), "RH_percent_ISS": _number(rh),
         "CO2_ppm_ISS": _number(co2)}
        for t, temp, rh, co2 in zip(tel.get("time", []), tel.get("temperature_iss", []),
                                    tel.get("rh_iss", []), tel.get("co2_iss", []))
    ]
    radiation = [
        {"Date": d, "GCR_mGy_per_day": _number(g), "SAA_mGy_per_day": _number(s), "Total_mGy_per_day": _number(tot)}
        for d, g, s, tot in zip(rad.get("time", []), rad.get("gcr", []), rad.get("saa", []), rad.get("total", []))
    ]
    info = {
        "mission": payload.get("mission"),
        "title": payload.get("mission_title"),
        "nasa_link": payload.get("mission_link"),
        "milestones": tel.get("milestones") or rad.get("milestones") or {},
    }
    return {"telemetry": telemetry, "radiation": radiation, "info": info}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(mission: str, parsed: dict, data_dir: Path = DATA_DIR) -> list[Path]:
    data_dir.mkdir(exist_ok=True)
    written = []
    for kind in ("telemetry", "radiation"):
        if parsed[kind]:
            path = data_dir / f"{mission}_{kind}.csv"
            _write_csv(path, parsed[kind])
            written.append(path)
    info = dict(parsed["info"], mission=mission, source=EDA_URL + f"dashboard_{mission}",
                downloaded=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                credit="Data courtesy of the NASA OSDR Environmental Data Application (EDA), "
                       "https://visualization.osdr.nasa.gov/eda/")
    info_path = data_dir / f"{mission}_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return written + [info_path]


def download(mission: str) -> str:
    url = EDA_URL + f"dashboard_{mission}"
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (astro-health-monitor; NASA Space Apps student project)"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8")


def main(argv: list[str]) -> int:
    if "--list" in argv:
        for name, what in MISSIONS.items():
            print(f"  {name:6s} {what}")
        return 0
    failed = False
    for mission in (argv or [DEFAULT_MISSION]):
        print(f"Downloading {mission} from NASA EDA ...")
        try:
            parsed = parse_eda_page(download(mission))
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            failed = True
            print(f"  Could not get {mission}: {exc}\n"
                  f"  Manual route: open {EDA_URL} in a browser, choose the mission, open the\n"
                  f"  Data Tables tab, download the Telemetry Table as CSV and save it as\n"
                  f"  data/{mission}_telemetry.csv (the dashboard reads NASA's own column names).")
            continue
        for path in save(mission, parsed):
            print(f"  wrote {path.relative_to(Path(__file__).parent)}")
        print(f"  {len(parsed['telemetry'])} cabin readings, {len(parsed['radiation'])} daily radiation readings")
    if not failed:
        print('\nDone. Restart the dashboard and pick the record under "Radiation and cabin data".')
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
