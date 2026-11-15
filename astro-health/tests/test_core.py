"""Quick checks for the logic that does not need a camera or a browser.

Run from the project folder:   python tests/test_core.py
(also works with pytest)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import simulator  # noqa: E402
import status  # noqa: E402
from vision_metrics import FatigueTracker, estimate_heart_rate, eye_aspect_ratio  # noqa: E402

P = status.load_params()


def test_classification_matches_agreed_table():
    hr = P["heart_rate"]
    assert [hr.classify(v) for v in (40, 41, 50, 51, 90, 91, 130, 131)] == \
        ["red", "yellow", "yellow", "green", "green", "yellow", "yellow", "red"]
    assert [P["spo2"].classify(v) for v in (91, 92, 95, 96)] == ["red", "yellow", "yellow", "green"]
    assert [P["career_dose"].classify(v) for v in (100, 400, 600)] == ["green", "yellow", "red"]
    assert [P["bmd_change"].classify(v) for v in (-1, -4, -6)] == ["green", "yellow", "red"]
    assert [P["co2"].classify(v) for v in (2.0, 2.8, 3.2)] == ["green", "yellow", "red"]
    assert [P["cabin_temp"].classify(v) for v in (17, 19, 22, 26, 28)] == \
        ["red", "yellow", "green", "yellow", "red"]
    assert P["comms_delay"].classify(1.3) == "info"
    assert P["heart_rate"].classify(None) == "info"
    assert status.worst(["green", "yellow", "info"]) == "yellow"
    assert status.worst(["green", "red", "yellow"]) == "red"


def test_simulator_covers_every_parameter():
    crew, cabin = simulator.generate()
    assert len(crew) == len(simulator.CREW)
    for key, p in P.items():
        if p.scope == "crew":
            assert all(key in df.columns for df in crew.values()), key
        elif p.scope == "cabin":
            assert key in cabin.columns, key
    for df in list(crew.values()) + [cabin]:
        assert len(df) == simulator.HOURS and not df.isna().any().any()
    # Same seed -> same data for everyone on the team.
    crew2, _ = simulator.generate()
    assert crew["Asha Verma"].equals(crew2["Asha Verma"])


def test_scripted_events_trigger_alerts():
    crew, cabin = simulator.generate()
    spe_hour = 62 * 24 + 16
    assert P["dose_rate"].classify(crew["Asha Verma"].dose_rate[spe_hour]) == "red"
    assert P["co2"].classify(cabin.co2[95 * 24 + 20]) == "red"
    assert P["core_temp"].classify(crew["Leo Brandt"].core_temp[42 * 24 + 12]) == "yellow"
    assert P["dose_rate"].classify(crew["Asha Verma"].dose_rate[10 * 24]) == "green"


def test_eye_aspect_ratio():
    open_eye = np.array([[0, 0], [1, 0.5], [2, 0.5], [3, 0], [2, -0.5], [1, -0.5]])
    shut_eye = open_eye * [1, 0.1]
    assert abs(eye_aspect_ratio(open_eye) - 1 / 3) < 1e-9
    assert eye_aspect_ratio(shut_eye) < 0.05


def test_fatigue_tracker_perclos_and_blinks():
    tr = FatigueTracker()
    fps, t = 30, 0.0
    for i in range(60 * fps):
        t = i / fps
        second = i // fps
        # Open at 0.30; a 0.2 s blink every 5 s; one long 6 s closure at 40-46 s.
        blink = (second % 5 == 2) and (i % fps) < 6 and second >= 6
        long_close = 40 <= t < 46
        tr.update(t, 0.08 if (blink or long_close) else 0.30)
    assert tr.calibrated and 0.20 < tr.threshold < 0.23
    assert 10 < tr.perclos < 16, tr.perclos           # ~6 s long closure + blinks out of 60 s
    assert 8 <= tr.blink_rate <= 12, tr.blink_rate    # blinks only; the long closure is not a blink
    assert tr.closed_for(t) == 0.0


def test_heart_rate_from_synthetic_pulse():
    rng = np.random.default_rng(0)
    for bpm in (55, 72, 110):
        # Uneven frame times, a 1% pulse ripple, slow lighting drift and noise.
        t = np.cumsum(rng.uniform(0.025, 0.042, 400))
        v = 120 + 1.2 * np.sin(2 * np.pi * bpm / 60 * t) + 0.8 * t + 0.6 * rng.standard_normal(len(t))
        est, quality = estimate_heart_rate(t, v)
        assert est is not None and abs(est - bpm) < 3, (bpm, est)
        assert quality > 0.3
    assert estimate_heart_rate([0, 1, 2], [1, 2, 3]) == (None, 0.0)   # too little data
    noise = rng.standard_normal(400)
    _, q = estimate_heart_rate(np.arange(400) / 30, noise)
    assert q < 0.3                                                    # pure noise is flagged unreliable


# ----------------------------------------------------------------------------- real records
def _eda_page(n_days: int = 80, start: str = "2018-06-29") -> str:
    """A page in the exact layout NASA's EDA serves, filled with made-up TEST values.

    Layout facts copied from the live site: the data sits in a <script id="data"> tag, the
    telemetry and radiation blocks are JSON text inside JSON, missing readings are the text
    "none", telemetry is every 5 minutes with gaps, and radiation is one row per day.
    """
    import json

    import pandas as pd
    times = pd.date_range(start + " 00:02", periods=n_days * 288, freq="5min")
    keep = ~((times >= "2018-07-10 06:00") & (times < "2018-07-10 20:00"))    # a 14-hour gap
    times = times[keep]
    hours = (times - times[0]) / pd.Timedelta(hours=1)
    co2 = [int(v) for v in 3000 + 600 * np.sin(hours / 30)]
    temp = [round(float(v), 1) for v in 23 + np.sin(hours / 12)]
    rh = [int(v) for v in 42 + 4 * np.cos(hours / 20)]
    co2[5] = temp[5] = rh[5] = "none"
    days = pd.date_range(start, periods=max(n_days - 4, 0), freq="D")
    gcr = [0.17] * len(days)
    saa = [0.10] * len(days)
    telemetry = {"time": [start + " 00:00:00"] + [t.strftime("%Y-%m-%d %H:%M:%S") for t in times],
                 "temperature_ground": [0] * (len(times) + 1), "temperature_iss": ["none"] + temp,
                 "rh_ground": [0] * (len(times) + 1), "rh_iss": ["none"] + rh,
                 "co2_ground": [0] * (len(times) + 1), "co2_iss": ["none"] + co2,
                 "milestones": {"Launch": start + " 00:00"}}
    radiation = {"time": [d.strftime("%Y-%m-%d") for d in days], "gcr": gcr, "saa": saa,
                 "total": [g + s for g, s in zip(gcr, saa)], "accumulated": list(np.cumsum([0.27] * len(days))),
                 "milestones": {}}
    payload = {"telemetry_data": json.dumps(telemetry), "radiation_data": json.dumps(radiation),
               "mission": "TEST-1", "mission_title": "Test record (not real data)",
               "mission_link": "https://example.invalid/TEST-1"}
    return ('<html><body><h1>x</h1><script id="data" type="application/json">'
            + json.dumps(payload) + "</script><script>var a = 1;</script></body></html>")


def _saved_test_record(tmp: Path, **kwargs) -> Path:
    import fetch_real_data
    fetch_real_data.save("TEST-1", fetch_real_data.parse_eda_page(_eda_page(**kwargs)), tmp)
    return tmp


def test_fetch_parses_nasa_page_layout():
    import fetch_real_data
    parsed = fetch_real_data.parse_eda_page(_eda_page(n_days=3))
    first, second = parsed["telemetry"][0], parsed["telemetry"][1]
    assert first["CO2_ppm_ISS"] == "" and first["Temp_degC_ISS"] == ""          # "none" -> blank
    assert second["Controller_Time_GMT"] == "2018-06-29 00:02:00" and second["CO2_ppm_ISS"] == 3000
    assert parsed["radiation"] == []                                           # n_days - 4 < 0
    assert parsed["info"]["title"].startswith("Test record")
    try:
        fetch_real_data.parse_eda_page("<html>no data here</html>")
        raise AssertionError("should have failed")
    except ValueError:
        pass


def test_real_record_units_and_alignment():
    import tempfile

    import real_data
    with tempfile.TemporaryDirectory() as d:
        data_dir = _saved_test_record(Path(d))
        assert real_data.available(data_dir) == ["TEST-1"]
        rec = real_data.load_record("TEST-1", data_dir)
    assert str(rec.origin) == "2018-06-29 00:00:00"
    # 3000 ppm at 760 mmHg = 2.28 mmHg
    assert abs(rec.cabin.co2[0] - 3000 * 760 / 1e6) < 0.02
    # 0.17 mGy GCR x 2.74 + 0.10 mGy SAA x 1.83 = 0.649 mSv/day
    assert abs(rec.habitat_dose[0] - (0.17 * 2.74 + 0.10 * 1.83)) < 1e-9
    assert rec.habitat_dose.notna().sum() == 76 * 24 and rec.habitat_dose[76 * 24:].isna().all()
    gap = rec.cabin.co2[11 * 24 + 6:11 * 24 + 20]         # the 14-hour gap on 10 July
    assert gap.isna().sum() >= 14 - 2 * real_data.MAX_GAP_HOURS and gap.isna().any()   # long gaps are not invented
    assert rec.cabin.co2[80 * 24 + 5:].isna().all()       # nothing after the record ends


def test_real_record_replaces_simulation_and_moves_events():
    import tempfile

    import real_data
    sim = real_data.build_mission(None)
    assert sim.coverage == {} and sim.record is None and sim.source("co2", 100) == "sim"
    with tempfile.TemporaryDirectory() as d:
        m = real_data.build_mission("TEST-1", _saved_test_record(Path(d)))
    h = 30 * 24
    assert m.source("co2", h) == "real" and m.source("dose_rate", h) == "real"
    assert m.source("career_dose", h) == "part" and m.source("o2", h) == "sim" and m.source("heart_rate", h) == "sim"
    assert m.source("co2", 100 * 24) == "sim"                       # after the record ends
    assert abs(m.cabin.co2[h] - m.record.cabin.co2[h]) < 1e-12      # real value used as is
    assert m.cabin.o2.equals(sim.cabin.o2)                          # untouched parameters stay simulated
    for a in simulator.CREW:
        df = m.crew[a.name]
        assert abs(df.dose_rate[h] - m.record.habitat_dose[h] * a.shielding) < 1e-12
        steps = np.diff(df.career_dose.to_numpy())
        assert np.allclose(steps, df.dose_rate.to_numpy()[1:] / 24)             # career dose still adds up
        assert df.heart_rate.equals(sim.crew[a.name].heart_rate)                # vitals unchanged
    # The scripted solar storm (day 62) would fall inside the real record, so it moves after it.
    labels = {e[0]: e[1] for e in m.events}
    assert labels["Solar particle event"][0] > 76 and labels["CO2 scrubber fault"][0] == 95
    storm_hour = labels["Solar particle event"][0] * 24 + labels["Solar particle event"][1]
    assert m.source("dose_rate", storm_hour) == "sim" and P["dose_rate"].classify(m.crew["Asha Verma"].dose_rate[storm_hour]) == "red"
    assert P["dose_rate"].classify(m.crew["Asha Verma"].dose_rate[62 * 24 + 16]) == "green"   # no storm in the real days
    assert "Real record: highest CO2" in labels and "Real record: highest radiation" in labels


# ----------------------------------------------------------------------------- trackers
def test_live_store_roundtrip_and_expiry():
    import tempfile
    import time

    import live_store
    with tempfile.TemporaryDirectory() as d:
        live = Path(d)
        assert live_store.latest(live) == {} and live_store.sources(live) == []
        live_store.write("Asha Verma", {"heart_rate": 72}, source="strap A", live_dir=live)
        live_store.write("Asha Verma", {"spo2": 98}, source="strap A", live_dir=live)       # same source, 2nd parameter
        live_store.write("Leo Brandt", {"heart_rate": 140}, source="watch/B", ttl=0.2, live_dir=live)
        now = live_store.latest(live)
        assert now[("Asha Verma", "heart_rate")].value == 72 and now[("Asha Verma", "spo2")].value == 98
        assert now[("Leo Brandt", "heart_rate")].source == "watch/B"
        time.sleep(0.05)
        live_store.write("Asha Verma", {"heart_rate": 75}, source="phone", live_dir=live)   # newer source wins
        assert live_store.latest(live)[("Asha Verma", "heart_rate")].value == 75
        assert [v for _, v in live_store.history("Asha Verma", "heart_rate", live_dir=live)] == [72, 75]
        assert len(live_store.sources(live)) == 3
        time.sleep(0.25)
        assert ("Leo Brandt", "heart_rate") not in live_store.latest(live)                  # expired
        live_store.clear(live)
        assert live_store.latest(live) == {}


def test_tracker_server_validates_and_stores():
    import json
    import tempfile
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    import live_store
    import tracker_server
    with tempfile.TemporaryDirectory() as d:
        original = live_store.LIVE_DIR
        live_store.write.__defaults__ = (live_store.DEFAULT_TTL_S, Path(d))    # point the store at a temp folder
        try:
            code, reply = tracker_server.handle_reading({"astronaut": "Asha Verma", "heart_rate": 135, "spo2": "97"})
            assert code == 200 and reply["status"] == {"heart_rate": "red", "spo2": "green"}
            assert tracker_server.handle_reading({"astronaut": "Nobody", "heart_rate": 70})[0] == 400
            assert tracker_server.handle_reading({"astronaut": "Asha Verma", "hart_rate": 70})[0] == 400
            assert tracker_server.handle_reading({"astronaut": "Asha Verma", "heart_rate": "fast"})[0] == 400
            assert tracker_server.handle_reading({"astronaut": "Asha Verma"})[0] == 400
            assert tracker_server.handle_reading({"astronaut": "Asha Verma", "co2": 2})[0] == 400   # cabin, not a person

            server = ThreadingHTTPServer(("127.0.0.1", 0), tracker_server.Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            body = json.dumps({"astronaut": "Mei Tanaka", "source": "test watch", "heart_rate": 64}).encode()
            req = urllib.request.Request(base + "/reading", data=body, headers={"Content-Type": "application/json"})
            assert json.loads(urllib.request.urlopen(req).read())["status"] == {"heart_rate": "green"}
            assert b"Send a reading" in urllib.request.urlopen(base + "/").read()
            assert "heart_rate" in json.loads(urllib.request.urlopen(base + "/params").read())["parameters"]
            try:
                urllib.request.urlopen(urllib.request.Request(base + "/reading", data=b"not json"))
                raise AssertionError("bad JSON should be rejected")
            except urllib.error.HTTPError as err:
                assert err.code == 400
            server.shutdown()
            server.server_close()
            assert live_store.latest(Path(d))[("Mei Tanaka", "heart_rate")].source == "test watch"
        finally:
            live_store.write.__defaults__ = (live_store.DEFAULT_TTL_S, original)


def test_bluetooth_heart_rate_packets():
    from tracker_ble import parse_heart_rate
    assert parse_heart_rate(bytes([0x00, 72])) == (72, [])                                  # 8-bit value
    assert parse_heart_rate(bytes([0x01, 0x2C, 0x01])) == (300, [])                         # 16-bit value
    bpm, rr = parse_heart_rate(bytes([0x10, 60, 0x00, 0x04, 0x00, 0x02]))                    # two RR intervals
    assert bpm == 60 and rr == [1000.0, 500.0]
    bpm, rr = parse_heart_rate(bytes([0x18, 80, 0x10, 0x00, 0x00, 0x03]))                    # energy field skipped
    assert bpm == 80 and rr == [750.0]



if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
