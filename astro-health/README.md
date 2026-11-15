# Astronaut Health Monitor

NASA Space Apps 2026 project: a dashboard that tracks crew health against
NASA's five hazards of human spaceflight, with green / yellow / red status for
every parameter.

Each value on the page comes from one of three places, and every tile says which:

1. **Live** - a tracker, a phone or the webcam sent it in the last few seconds
2. **Real** - a NASA record measured on the International Space Station
3. **Simulated** - everything else (see `simulator.py`)

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens in your browser. Use **Jump to an event** in the sidebar
for the demo moments, or switch on **Play** to watch the mission run.

Camera (optional, in a second terminal):

```
python camera_monitor.py
```

Look at the screen for 5 seconds while it calibrates, then open the **Camera**
section of the dashboard. Press `q` in the preview window to stop. The first
run downloads a ~4 MB face model into `models/`.

Tests:

```
python tests/test_core.py
```

## Real ISS data

```
python fetch_real_data.py
```

This downloads a real record from NASA's Environmental Data Application into
`data/` (default: Rodent Research 7, about 80 days in mid-2018). Restart the
dashboard and it is selected automatically under **Radiation and cabin data**
in the sidebar; switch back to "Simulated" there at any time.

What becomes real, for the mission days the record covers:

| Parameter | NASA publishes | Dashboard shows | Conversion |
|---|---|---|---|
| Cabin temperature | deg C, every 5 min | deg C, hourly mean | none |
| Humidity | %, every 5 min | %, hourly mean | none |
| Cabin CO2 | ppm, every 5 min | mmHg, hourly mean | ppm x 760 / 1,000,000 |
| Radiation dose rate | absorbed dose, mGy/day, split into cosmic rays (GCR) and South Atlantic Anomaly (SAA) | dose equivalent, mSv/day | GCR x 2.74 + SAA x 1.83 |

Things to know, and to say in the pitch:

- The record starts at mission day 0. After it ends, those parameters go back
  to simulated; charts mark the switch and tiles change their tag.
- The simulated solar storm and CO2 fault are moved to after the real record,
  so real and scripted data never overlap. Two extra jump buttons go to the
  highest CO2 and highest radiation in the real record.
- The cabin sensors belong to the Rodent Research habitat on the ISS, so they
  are a stand-in for crew-cabin air, not an official cabin log.
- Values are shown as published. Gaps of up to 3 hours are bridged; longer
  gaps stay simulated. Nothing is filtered out.
- The mGy to mSv step uses quality factors measured by a NASA instrument in low
  Earth orbit (Badhwar et al., STS-40). It gives a dose equivalent, used here
  as an estimate of the effective dose the NASA limits are written in.
- Other missions: `python fetch_real_data.py --list`, then
  `python fetch_real_data.py RR-1 RR-6`.
- Credit line for slides: "Data courtesy of the NASA OSDR Environmental Data
  Application (EDA), https://visualization.osdr.nasa.gov/eda/".

## Connecting trackers

A live reading replaces the simulated value for one crew member, is tagged
LIVE, gets a five-minute live chart, and is checked against the same limits.
The dashboard notices new readings within about two seconds. The **Trackers**
section lists what is connected and has a form for sending a test reading.

| Route | Command | Use it for |
|---|---|---|
| Bluetooth | `python tracker_ble.py --astronaut "Asha Verma"` | Heart-rate chest straps and arm bands, and sports watches in "broadcast heart rate" mode. Needs `pip install bleak` |
| Wi-Fi / HTTP | `python tracker_server.py --lan` | Anything else: a phone (open the address it prints for a form), a script, or another app posting JSON |
| Webcam | `python camera_monitor.py --astronaut "Asha Verma"` | Contactless heart rate and eye closure |

Sending a reading from a script:

```
curl -X POST http://localhost:8600/reading -H "Content-Type: application/json" \
     -d '{"astronaut": "Asha Verma", "source": "my-watch", "heart_rate": 72, "spo2": 98}'
```

Any per-person parameter key from `thresholds.csv` is accepted. From your own
Python code it is one line: `live_store.write("Asha Verma", {"heart_rate": 72}, source="my-device")`.

Not included: the Fitbit, Garmin, Apple and Oura cloud services. Each needs a
registered developer account and a sign-in flow, which is too much for a
weekend; their data can still come in through the HTTP route.

## Files

| File | What it does |
|---|---|
| `app.py` | The dashboard: layout, tiles, charts, alerts, playback |
| `thresholds.csv` | Every parameter and its green/yellow/red limits, with sources. **Edit limits here, not in code** |
| `status.py` | Reads the thresholds and classifies a value as green, yellow or red |
| `simulator.py` | Generates the 180-day mission, the crew and the scripted events |
| `real_data.py` | Lays a real ISS record over the simulation and tracks which values are real |
| `fetch_real_data.py` | Downloads real records from NASA into `data/` |
| `live_store.py` | Shared store that every tracker writes to and the dashboard reads |
| `tracker_ble.py` | Bluetooth heart-rate trackers |
| `tracker_server.py` | Receives readings from phones, scripts and other devices over HTTP |
| `camera_monitor.py` | Webcam loop: finds the face, writes readings to `camera_state.json` |
| `vision_metrics.py` | The maths behind the camera: eye openness, PERCLOS, heart rate |
| `tests/test_core.py` | Checks for the classification, simulator and camera maths |

## How the colours work

Each row of `thresholds.csv` has up to four numbers:

```
red_low   yellow_low   yellow_high   red_high
```

- at or below `red_low`, or at or above `red_high`: **red**
- below `yellow_low` or above `yellow_high`: **yellow**
- otherwise: **green**

Leave a cell blank for "no limit on that side". Red limits come from
NASA-STD-3001 or the NHS NEWS2 score wherever one exists; the `source` column
says which bands are our own.

## Common changes

- **Change a limit**: edit the row in `thresholds.csv`, restart the dashboard.
- **Add a parameter**: add a row to `thresholds.csv` (pick a `hazard` and a
  `scope` of `crew` or `cabin`), then add a column with the same `key` in
  `simulator.py`. The dashboard picks it up automatically.
- **Add or change an event**: see `EVENTS` and the `_window(...)` lines in
  `simulator.py`.
- **Change the crew**: edit `CREW` in `simulator.py`.

## Still to do

- [ ] Verify the four sources marked "to verify" in `thresholds.csv`
- [ ] Check the cabin O2 bands against Table 6.2-1 of NASA-STD-3001 Volume 2
- [ ] Run `python fetch_real_data.py` and check the real record loads (the
      loader is tested against NASA's page layout, but the download itself has
      only been run from a browser)
- [ ] Try `tracker_ble.py` with a real heart-rate strap or watch (packet
      decoding is tested, the Bluetooth connection has not been run)
- [ ] Not yet simulated: blood-forming organ dose, femoral-neck bone density, VO2max
- [ ] Design pass and pitch

## Sources

- NASA-STD-3001 Volume 1 Rev C (crew health) and Volume 2 (habitat)
- NASA Open Science Data Repository, Environmental Data Application (real ISS records)
- Badhwar et al., TEPC measurements on STS-40 (radiation quality factors)
- NASA OCHMO technical briefs: CO2 (TB-004), ECLSS (TB-002), Water (TB-027), Acoustics (TB-035)
- NEWS2, Royal College of Physicians (vital-sign bands)
- NASA Human Research Program: hazards and risks pages
- Canadian Space Agency: neutron radiation monitoring
