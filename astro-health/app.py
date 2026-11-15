"""Astronaut health dashboard.

Run with:   streamlit run app.py

Where each value on the page comes from, in order of priority:
    1. LIVE       a tracker, phone or the camera sent it in the last few seconds  (live_store.py)
    2. REAL       a NASA record measured on the ISS covers this mission hour      (real_data.py)
    3. SIMULATED  otherwise                                                        (simulator.py)
Every tile says which of the three it is showing.

Page layout, top to bottom:
    sidebar   - mission clock, data source, crew member picker, tracker status
    header    - mission name, time and what is real
    overview  - one status card per astronaut plus one for the cabin
    alerts    - everything currently yellow or red
    section   - tiles and charts for one hazard group, or the camera / trackers / thresholds views
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import live_store
import real_data
import simulator
import status
from status import GREEN, INFO, RED, STATUS_STYLE, YELLOW

CAMERA_STATE = Path(__file__).parent / "camera_state.json"
HAZARD_SECTIONS = ["Vitals", "Radiation", "Gravity", "Isolation", "Cabin", "Distance"]
SECTIONS = HAZARD_SECTIONS + ["Camera", "Trackers", "Thresholds"]
SECTION_HELP = {
    "Vitals": "Core vital signs, banded with the NHS NEWS2 early-warning score.",
    "Radiation": "NASA hazard 1: space radiation. Limits from NASA-STD-3001 Volume 1.",
    "Gravity": "NASA hazard 4: gravity fields. Bone, muscle and fitness loss in microgravity.",
    "Isolation": "NASA hazard 2: isolation and confinement. Sleep, fatigue and mood.",
    "Cabin": "NASA hazard 5: hostile/closed environments. Limits from NASA-STD-3001 Volume 2.",
    "Distance": "NASA hazard 3: distance from Earth. Self-sufficiency: food, water, supplies.",
}
WINDOWS = {"24 hours": 24, "7 days": 7 * 24, "30 days": 30 * 24, "Whole mission": simulator.HOURS}
LINE_COLOR = "#3b6fd4"
MAX_POINTS = 400   # charts are thinned to this many points to stay fast
SIMULATED = "Simulated"
SOURCE_TEXT = {
    "live": "● LIVE from {detail}",
    "real": "Real ISS data",
    "part": "Measured dose + simulated history",
    "sim": "Simulated",
}

st.set_page_config(page_title="Astronaut Health Monitor", page_icon="🛰️", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading mission data...")
def load_mission(record_name: str | None) -> real_data.Mission:
    """Simulated mission, optionally with a real ISS record laid over it. Treat as read-only."""
    return real_data.build_mission(record_name)


@st.cache_data
def load_params():
    return status.load_params()


PARAMS = load_params()
CREW_NAMES = [a.name for a in simulator.CREW]
ROLES = {a.name: a.role for a in simulator.CREW}
LAST_HOUR = simulator.HOURS - 1

ss = st.session_state
RECORDS = real_data.available()
RECORD_OPTIONS = [SIMULATED] + RECORDS
# Use a real record by default when one has been downloaded.
ss.setdefault("record_choice", RECORDS[0] if RECORDS else SIMULATED)
if ss.record_choice not in RECORD_OPTIONS:
    ss.record_choice = SIMULATED
try:
    mission = load_mission(None if ss.record_choice == SIMULATED else ss.record_choice)
    load_error = None
except Exception as exc:  # noqa: BLE001 - a bad data file must not take the dashboard down
    mission, load_error = load_mission(None), f"Could not load {ss.record_choice}: {exc}"
crew_data, cabin_data, record = mission.crew, mission.cabin, mission.record


def fresh_live() -> dict[tuple[str, str], live_store.Reading]:
    """Live readings the dashboard can use: known crew member, known per-person parameter."""
    return {k: r for k, r in live_store.latest().items()
            if k[0] in CREW_NAMES and k[1] in PARAMS and PARAMS[k[1]].scope == "crew"}


def live_signature(live: dict) -> tuple:
    return tuple(sorted((a, k, round(r.timestamp, 1)) for (a, k), r in live.items()))


LIVE = fresh_live()
ss["live_sig"] = live_signature(LIVE)


# ----------------------------------------------------------------------------- clock
ss.setdefault("hour", 30 * 24 + 9)
ss.setdefault("playing", False)
ss.setdefault("astronaut", CREW_NAMES[0])


def set_hour(hour: int) -> None:
    ss.hour = int(min(max(hour, 0), LAST_HOUR))
    ss.day_slider = ss.hour // 24
    ss.hour_slider = ss.hour % 24


def on_slider_change() -> None:
    set_hour(ss.day_slider * 24 + ss.hour_slider)


def jump_to(day: int, hour: int) -> None:
    ss.playing = False
    set_hour(day * 24 + hour)


def select_astronaut(name: str) -> None:
    ss.astronaut = name


# While playing, move the clock forward once per rerun (before any widget is drawn).
if ss.playing and ss.get("_tick"):
    set_hour(ss.hour + ss.get("speed", 6))
    if ss.hour >= LAST_HOUR:
        ss.playing = False
ss["_tick"] = False
if "day_slider" not in ss:
    set_hour(ss.hour)
NOW = ss.hour


# ----------------------------------------------------------------------------- helpers
def pill(state: str, text: str | None = None) -> str:
    s = STATUS_STYLE[state]
    return (f"<span style='background:{s['color']};color:{s['text']};padding:2px 10px;"
            f"border-radius:999px;font-size:0.8rem;font-weight:600;white-space:nowrap'>"
            f"{s['icon']} {text or s['label']}</span>")


def frame_for(param: status.Param, astronaut: str) -> pd.DataFrame:
    return cabin_data if param.scope == "cabin" else crew_data[astronaut]


def current(param: status.Param, astronaut: str) -> tuple[float, str, str]:
    """(value, origin, detail) right now. origin is 'live', 'real', 'part' or 'sim'."""
    reading = LIVE.get((astronaut, param.key)) if param.scope == "crew" else None
    if reading is not None:
        return reading.value, "live", reading.source
    return float(frame_for(param, astronaut)[param.key].iloc[NOW]), mission.source(param.key, NOW), ""


def value_now(param: status.Param, astronaut: str) -> float:
    return current(param, astronaut)[0]


def source_text(param: status.Param, astronaut: str) -> str:
    _, origin, detail = current(param, astronaut)
    return SOURCE_TEXT[origin].format(detail=detail)


def statuses_for(scope: str, astronaut: str | None = None) -> dict[str, str]:
    """Status of every parameter in a scope at the current mission time."""
    return {k: p.classify(value_now(p, astronaut or CREW_NAMES[0]))
            for k, p in PARAMS.items() if p.scope == scope}


def count_text(states: dict[str, str]) -> str:
    red = sum(s == RED for s in states.values())
    yellow = sum(s == YELLOW for s in states.values())
    if not red and not yellow:
        return "All parameters in range"
    parts = []
    if red:
        parts.append(f"{red} alert{'s' if red > 1 else ''}")
    if yellow:
        parts.append(f"{yellow} to watch")
    return ", ".join(parts)


def add_bands(fig: go.Figure, param: status.Param, y0: float, y1: float) -> None:
    """Green / yellow / red background washes between the parameter's limits."""
    if param.info_only:
        return
    edges = [(None, param.red_low, RED), (param.red_low, param.yellow_low, YELLOW),
             (param.yellow_low, param.yellow_high, GREEN),
             (param.yellow_high, param.red_high, YELLOW), (param.red_high, None, RED)]
    for low, high, state in edges:
        # A band exists only if the limit that opens it is defined.
        if state == RED and low is None and high is None:
            continue
        if state == YELLOW and ((high == param.yellow_low and high is None)
                                or (low == param.yellow_high and low is None)):
            continue
        b0 = y0 if low is None else max(low, y0)
        b1 = y1 if high is None else min(high, y1)
        if b1 > b0:
            fig.add_hrect(y0=b0, y1=b1, fillcolor=STATUS_STYLE[state]["color"],
                          opacity=0.13, line_width=0, layer="below")


def _finish(fig: go.Figure, param: status.Param, x, y, title: str, x_title: str, hover_x: str,
            y0: float, y1: float) -> go.Figure:
    """Shared look for both chart kinds: the line, a status-coloured dot on the newest point, axes."""
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=LINE_COLOR, width=2),
        hovertemplate=hover_x + "<br>%{y:." + str(param.decimals) + "f} " + param.unit + "<extra></extra>",
    ))
    newest = param.classify(float(y[-1]))
    fig.add_trace(go.Scatter(
        x=[x[-1]], y=[y[-1]], mode="markers", hoverinfo="skip",
        marker=dict(size=10, color=STATUS_STYLE[newest]["color"], line=dict(color="white", width=2)),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=250, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
        xaxis=dict(title=x_title, showgrid=False),
        yaxis=dict(range=[y0, y1], gridcolor="rgba(128,128,128,0.18)"),
        hovermode="x",
    )
    return fig


def _y_range(values, param: status.Param) -> tuple[float, float]:
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = max((hi - lo) * 0.15, abs(hi) * 0.02, 10 ** -param.decimals)
    return lo - pad, hi + pad


def band_chart(param: status.Param, series: pd.Series, window_hours: int) -> go.Figure:
    """Mission history of one parameter, with a marker wherever real data starts or stops."""
    start = max(0, NOW - window_hours + 1)
    data = series.iloc[start:NOW + 1]
    step = max(1, len(data) // MAX_POINTS)
    if step > 1:   # thin long windows but always keep the newest point
        data = pd.concat([data.iloc[:-1:step], data.iloc[-1:]])
    x = data.index / 24.0
    y0, y1 = _y_range(data.values, param)

    fig = go.Figure()
    add_bands(fig, param, y0, y1)
    origin = mission.source(param.key, NOW)
    mask = mission.coverage.get(param.key)
    if mask is not None:
        window = mask[start:NOW + 1].astype(int)
        for i in np.flatnonzero(np.diff(window))[:4]:
            becomes_real = window[i + 1] == 1
            fig.add_vline(x=(start + i + 1) / 24.0, line_width=1, line_color="rgba(128,128,128,0.8)",
                          annotation_text="real data ▸" if becomes_real else "simulated ▸",
                          annotation_position="top left" if becomes_real else "top right",
                          annotation_font_size=11)
    suffix = {"real": " · real ISS data", "part": " · measured dose + simulated history"}.get(origin, "")
    return _finish(fig, param, x, data.values, f"{param.name} ({param.unit}){suffix}", "Mission day",
                   "Day %{x:.1f}", y0, y1)


def live_chart(param: status.Param, astronaut: str, source: str) -> go.Figure | None:
    """The last five minutes of a live tracker's readings."""
    points = live_store.history(astronaut, param.key)
    if not points:
        return None
    now = time.time()
    x = [t - now for t, _ in points]
    y = [v for _, v in points]
    y0, y1 = _y_range(y, param)
    fig = go.Figure()
    add_bands(fig, param, y0, y1)
    fig = _finish(fig, param, x, y, f"{param.name} ({param.unit}) · LIVE from {source}", "Seconds before now",
                  "%{x:.0f} s", y0, y1)
    fig.update_xaxes(range=[-live_store.HISTORY_WINDOW_S, 0])
    fig.update_traces(mode="lines+markers", marker=dict(size=4), selector=dict(mode="lines"))
    return fig


def tile(param: status.Param, value: float | None, state: str, source: str | None = None) -> None:
    with st.container(border=True):
        st.caption(param.name)
        st.markdown(f"<div style='font-size:1.6rem;font-weight:600;line-height:1.2'>{param.fmt(value)}"
                    f" <span style='font-size:0.85rem;font-weight:400;opacity:0.7'>{param.unit}</span></div>",
                    unsafe_allow_html=True)
        st.markdown(pill(state), unsafe_allow_html=True)
        limits = "Info only" if param.info_only else f"Green: {param.green_range_text()}"
        st.caption(f"{limits}  \n{source}" if source else limits)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Mission clock")
    st.toggle("▶ Play", key="playing", help="Advance mission time automatically")
    st.select_slider("Speed (mission hours per second)", options=[1, 3, 6, 12, 24], value=6, key="speed")
    st.slider("Mission day", 0, simulator.MISSION_DAYS - 1, key="day_slider", on_change=on_slider_change)
    st.slider("Hour of day", 0, 23, key="hour_slider", on_change=on_slider_change)

    st.subheader("Jump to an event")
    for label, (day, hour), description in mission.events:
        st.button(label, help=description, on_click=jump_to, args=(day, hour), width="stretch")

    st.divider()
    st.selectbox("Radiation and cabin data", RECORD_OPTIONS, key="record_choice",
                 format_func=lambda m: m if m == SIMULATED else f"Real ISS record: {m}",
                 help="Real records are measured on the ISS and published by NASA. "
                      "Download them with:  python fetch_real_data.py")
    if not RECORDS:
        st.caption("No real records downloaded yet. Run `python fetch_real_data.py`, then restart.")
    st.radio("Crew member", CREW_NAMES, key="astronaut",
             format_func=lambda n: f"{n} ({ROLES[n]})")
    window_label = st.selectbox("Chart window", list(WINDOWS), index=1)

    @st.fragment(run_every=2.0)
    def live_watch() -> None:
        """Checks for tracker data every 2 s and refreshes the page when something new arrives."""
        live_now = fresh_live()
        devices = sorted({r.source for r in live_now.values()})
        st.caption(f"Trackers live: {', '.join(devices)}" if devices else "Trackers: none connected")
        if live_signature(live_now) != ss.get("live_sig"):
            st.rerun()

    live_watch()

selected = ss.astronaut
window_hours = WINDOWS[window_label]

# ----------------------------------------------------------------------------- header
st.title("🛰️ Astronaut Health Monitor")
clock = f"**{simulator.MISSION_NAME}** · Mission time: **{simulator.fmt_time(NOW)}**"
if record is not None and any(mask[NOW] for mask in mission.coverage.values()):
    clock += f" · real record: {record.real_time(NOW):%d %b %Y, %H:%M} UTC"
st.markdown(clock)
if load_error:
    st.error(load_error)
if record is None:
    st.caption("Radiation and cabin data: simulated. Crew data: simulated unless a tracker is connected.")
else:
    st.caption(real_data.describe(record) + " Data courtesy of the NASA OSDR Environmental Data Application.")
    for note in record.notes:
        st.warning(note)

# ----------------------------------------------------------------------------- overview
crew_states = {name: statuses_for("crew", name) for name in CREW_NAMES}
cabin_states = statuses_for("cabin")

cards = st.columns(len(CREW_NAMES) + 1)
for col, name in zip(cards, CREW_NAMES):
    with col, st.container(border=True):
        st.markdown(f"**{name}**  \n{ROLES[name]}")
        st.markdown(pill(status.worst(crew_states[name].values())), unsafe_allow_html=True)
        live_count = sum(1 for (who, _key) in LIVE if who == name)
        st.caption(count_text(crew_states[name]) + (f" · {live_count} live" if live_count else ""))
        st.button("Selected" if name == selected else "View", key=f"view_{name}",
                  on_click=select_astronaut, args=(name,), disabled=name == selected, width="stretch")
with cards[-1], st.container(border=True):
    st.markdown("**Cabin**  \nShared habitat")
    st.markdown(pill(status.worst(cabin_states.values())), unsafe_allow_html=True)
    st.caption(count_text(cabin_states))

# ----------------------------------------------------------------------------- alerts
alerts = []
for who, states in [*crew_states.items(), ("Cabin", cabin_states)]:
    person = who if who != "Cabin" else CREW_NAMES[0]
    for key, state in states.items():
        if state in (RED, YELLOW):
            p = PARAMS[key]
            alerts.append({
                "_rank": 0 if state == RED else 1,
                "Status": f"{STATUS_STYLE[state]['icon']} {STATUS_STYLE[state]['label']}",
                "Who": who, "Parameter": p.name,
                "Value": f"{p.fmt(value_now(p, person))} {p.unit}",
                "Green range": p.green_range_text(), "Hazard": p.hazard,
                "Data": source_text(p, person).replace("● ", ""),
            })
alerts.sort(key=lambda a: a["_rank"])
with st.expander(f"Active alerts ({len(alerts)})", expanded=bool(alerts)):
    if alerts:
        st.dataframe(pd.DataFrame(alerts).drop(columns="_rank"), hide_index=True, width="stretch")
    else:
        st.write("Nothing to report: every parameter is in its green range.")

# ----------------------------------------------------------------------------- sections
section = st.segmented_control("Section", SECTIONS, default="Vitals", key="section",
                               label_visibility="collapsed") or "Vitals"

if section in HAZARD_SECTIONS:
    st.caption(SECTION_HELP[section])
    params = [p for p in PARAMS.values() if p.hazard == section and p.scope in ("crew", "cabin")]
    if any(p.scope == "crew" for p in params):
        st.subheader(f"{section} – {selected}")
    else:
        st.subheader(section)

    per_row = 4
    for i in range(0, len(params), per_row):
        for col, p in zip(st.columns(per_row), params[i:i + per_row]):
            with col:
                v = value_now(p, selected)
                tile(p, v, p.classify(v), source_text(p, selected))

    chart_cols = st.columns(2)
    for i, p in enumerate(params):
        with chart_cols[i % 2]:
            _, origin, detail = current(p, selected)
            fig = live_chart(p, selected, detail) if origin == "live" else None
            if fig is None:
                fig = band_chart(p, frame_for(p, selected)[p.key], window_hours)
            st.plotly_chart(fig, key=f"chart_{p.key}", config={"displayModeBar": False})
    with st.expander("Show this section as a table"):
        st.dataframe(pd.DataFrame([{
            "Parameter": p.name, "Value": p.fmt(value_now(p, selected)), "Unit": p.unit,
            "Status": STATUS_STYLE[p.classify(value_now(p, selected))]["label"],
            "Green range": p.green_range_text(), "Data": source_text(p, selected).replace("● ", ""),
            "Limit source": p.source,
        } for p in params]), hide_index=True, width="stretch")

elif section == "Camera":
    st.subheader("Camera: fatigue and contactless heart rate")
    st.caption("Live readings from a webcam: real data from whoever is in front of the camera. "
               "Start it with `--astronaut \"Name\"` to drive that crew member's tiles as well.")

    @st.fragment(run_every=1.0)
    def camera_panel() -> None:
        state = None
        if CAMERA_STATE.exists():
            try:
                state = json.loads(CAMERA_STATE.read_text())
            except (json.JSONDecodeError, OSError):
                state = None
        if not state or time.time() - state.get("timestamp", 0) > 5:
            st.info("Camera offline. In a second terminal, from the project folder, run:\n\n"
                    "`python camera_monitor.py`\n\nThen look at the screen for 5 seconds so it can calibrate.")
            return
        if not state["face_found"]:
            st.warning("Camera is running but no face is in view.")
            return

        perclos_p, hr_p = PARAMS["perclos"], PARAMS["heart_rate"]
        c1, c2, c3 = st.columns(3)
        with c1:
            if state["calibrated"]:
                tile(perclos_p, state["perclos"], perclos_p.classify(state["perclos"]))
            else:
                st.info("Calibrating: keep your eyes open and look at the screen.")
        with c2, st.container(border=True):
            st.caption("Blink rate")
            st.markdown(f"<div style='font-size:1.6rem;font-weight:600'>{state['blink_rate']:.0f}"
                        " <span style='font-size:0.85rem;font-weight:400;opacity:0.7'>per min</span></div>",
                        unsafe_allow_html=True)
            st.markdown(pill(INFO), unsafe_allow_html=True)
            st.caption("Typical at rest: roughly 10 to 20")
        with c3:
            reliable = state["heart_rate"] is not None and state["hr_quality"] >= 0.3
            if reliable:
                tile(hr_p, state["heart_rate"], hr_p.classify(state["heart_rate"]))
            else:
                with st.container(border=True):
                    st.caption("Heart rate (experimental)")
                    st.markdown("<div style='font-size:1.6rem;font-weight:600'>–</div>", unsafe_allow_html=True)
                    st.markdown(pill(INFO, "MEASURING"), unsafe_allow_html=True)
                    st.caption("Hold still in steady light for about 10 seconds")
        if state["microsleep"]:
            st.error(f"✖ Microsleep warning: eyes closed for {state['eyes_closed_s']:.1f} s")

    camera_panel()
    with st.expander("How it works"):
        st.markdown(
            "- **Fatigue**: the camera tracks six points around each eye and measures how open the "
            "eye is. PERCLOS is the percentage of the last minute the eyes were closed; a higher "
            "value means a drowsier person.\n"
            "- **Heart rate**: each heartbeat slightly changes how much green light the skin "
            "reflects. The camera averages the forehead and cheeks and finds the strongest rhythm "
            "between 45 and 180 beats per minute. This is experimental and is hidden when the "
            "signal is weak.")

elif section == "Trackers":
    st.subheader("Trackers")
    st.caption("Connect real devices. A live reading replaces the simulated value for that crew member, "
               "is tagged LIVE on its tile, and is checked against the same limits.")

    rows = live_store.sources()
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No trackers connected. Use one of the three routes below, or send a test reading.")

    left, right = st.columns(2)
    with left, st.form("test_reading", border=True):
        st.markdown("**Send a test reading**")
        crew_keys = [k for k, p in PARAMS.items() if p.scope == "crew" and not p.info_only]
        who = st.selectbox("Crew member", CREW_NAMES, index=CREW_NAMES.index(selected))
        key = st.selectbox("Parameter", crew_keys, format_func=lambda k: f"{PARAMS[k].name} ({PARAMS[k].unit})")
        number = st.number_input("Value", value=None, placeholder="Type a number", step=None, format="%g")
        if st.form_submit_button("Send", type="primary"):
            if number is None:
                st.warning("Type a value first.")
            else:
                live_store.write(who, {key: float(number)}, source="manual entry", ttl=live_store.MANUAL_TTL_S)
                st.rerun()
        st.caption("Stays live for two minutes, then the tile returns to its simulated value.")
    with right, st.container(border=True):
        st.markdown("**Ways to connect**")
        st.markdown(
            "1. **Bluetooth heart-rate tracker** (chest strap, arm band, or a sports watch in "
            "broadcast mode)  \n`python tracker_ble.py --astronaut \"Asha Verma\"`\n"
            "2. **Phone, script or any other device** over Wi-Fi  \n`python tracker_server.py --lan`  \n"
            "then open the address it prints, or POST JSON to `/reading`\n"
            "3. **Webcam** for heart rate and eye closure  \n"
            "`python camera_monitor.py --astronaut \"Asha Verma\"`")
        if st.button("Clear live readings"):
            live_store.clear()
            st.rerun()

else:  # Thresholds
    st.subheader("Thresholds")
    st.caption("Edit thresholds.csv to change any band; the dashboard picks it up on restart.")
    table = pd.read_csv(status.THRESHOLDS_FILE).drop(columns=["key", "decimals"])
    st.dataframe(table, hide_index=True, width="stretch")
    with st.expander("Where the data comes from"):
        st.markdown(
            "- **Limits**: NASA-STD-3001 Volumes 1 and 2 and the NHS NEWS2 score, as listed in the table.\n"
            "- **Real records**: cabin temperature, humidity and CO2, and daily radiation dose, measured on "
            "the International Space Station and published in NASA's Environmental Data Application "
            "(https://visualization.osdr.nasa.gov/eda/). CO2 is converted from ppm to mmHg at 760 mmHg "
            f"cabin pressure. Absorbed dose (mGy) is converted to dose equivalent (mSv) with quality "
            f"factors of {real_data.Q_GCR} for cosmic rays and {real_data.Q_SAA} for South Atlantic "
            "Anomaly radiation.\n"
            "- **Live readings**: whatever trackers are connected (see the Trackers section).\n"
            "- **Everything else** is simulated by simulator.py.")

# ----------------------------------------------------------------------------- playback
if ss.playing:
    time.sleep(1.0)
    ss["_tick"] = True
    st.rerun()
