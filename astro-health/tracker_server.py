"""Receive readings from any tracker, phone or script over HTTP.

    python tracker_server.py           # this computer only  -> http://localhost:8600
    python tracker_server.py --lan     # also phones and other laptops on the same Wi-Fi

Then either

  * open the address in a browser for a form to type readings in, or
  * have a device or script POST JSON to /reading:

        curl -X POST http://localhost:8600/reading \
             -H "Content-Type: application/json" \
             -d '{"astronaut": "Asha Verma", "source": "my-watch", "heart_rate": 72, "spo2": 98}'

    "astronaut"  one of the crew names in simulator.py
    "source"     any label for the device (shown on the dashboard)
    then any crew parameter keys from thresholds.csv with numeric values
    "ttl"        optional: seconds the reading stays live (default 120)

The reply says how each value was classified (green / yellow / red).

Endpoints:  GET /  form    GET /params  valid names    GET /status  live sources
            POST /reading

There is no password. Use --lan only on a network you trust (a phone hotspot is
ideal; university Wi-Fi often blocks devices from talking to each other anyway).
Needs only the Python standard library.
"""
from __future__ import annotations

import argparse
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import live_store
import simulator
import status

PORT = 8600
MAX_BODY = 10_000
PARAMS = {k: p for k, p in status.load_params().items() if p.scope == "crew" and not p.info_only}
CREW_NAMES = [a.name for a in simulator.CREW]
RESERVED = {"astronaut", "source", "ttl"}


def handle_reading(payload: dict) -> tuple[int, dict]:
    """Validate one reading and store it. Returns (HTTP status, reply). Kept separate so it can be tested."""
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "Send a JSON object."}
    astronaut = payload.get("astronaut")
    if astronaut not in CREW_NAMES:
        return 400, {"ok": False, "error": f"Unknown astronaut {astronaut!r}.", "valid": CREW_NAMES}
    values, problems = {}, []
    for key, raw in payload.items():
        if key in RESERVED:
            continue
        if key not in PARAMS:
            problems.append(f"unknown parameter {key!r}")
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            problems.append(f"{key} must be a number")
            continue
        if number != number or number in (float("inf"), float("-inf")):
            problems.append(f"{key} must be a finite number")
            continue
        values[key] = number
    if problems or not values:
        return 400, {"ok": False, "error": "; ".join(problems) or "No parameter values sent.",
                     "valid_parameters": sorted(PARAMS)}
    try:
        ttl = min(max(float(payload.get("ttl", live_store.MANUAL_TTL_S)), 1.0), 3600.0)
    except (TypeError, ValueError):
        ttl = live_store.MANUAL_TTL_S
    source = str(payload.get("source") or "http")[:40]
    live_store.write(astronaut, values, source=source, ttl=ttl)
    return 200, {"ok": True, "astronaut": astronaut, "source": source,
                 "status": {k: PARAMS[k].classify(v) for k, v in values.items()}}


def form_page() -> str:
    crew = "".join(f"<option>{n}</option>" for n in CREW_NAMES)
    fields = "".join(
        f'<label>{p.name} <small>({p.unit}; green: {p.green_range_text()})</small>'
        f'<input name="{k}" type="number" step="any" inputmode="decimal"></label>'
        for k, p in PARAMS.items())
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Send a reading</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:34rem;margin:1.5rem auto;padding:0 1rem;line-height:1.4}}
 label{{display:block;margin:.7rem 0 .2rem;font-weight:600}} small{{font-weight:400;color:#666}}
 input,select,button{{width:100%;box-sizing:border-box;font-size:1rem;padding:.55rem;margin-top:.2rem}}
 button{{margin-top:1.2rem;font-weight:600}} #out{{margin-top:1rem;white-space:pre-wrap}}
</style></head><body>
<h1>Send a reading</h1>
<p>Fill in any values you have and leave the rest blank. They appear on the dashboard as live readings for two minutes.</p>
<form id="f">
 <label>Crew member<select name="astronaut">{crew}</select></label>
 <label>Device or your name<input name="source" value="phone form" maxlength="40"></label>
 {fields}
 <button type="submit">Send to dashboard</button>
</form>
<div id="out" role="status"></div>
<script>
const words = {{green: "OK", yellow: "WATCH", red: "ALERT"}};
document.getElementById("f").addEventListener("submit", async (event) => {{
  event.preventDefault();
  const body = {{}};
  for (const [key, value] of new FormData(event.target)) if (value !== "") body[key] = value;
  const out = document.getElementById("out");
  try {{
    const reply = await (await fetch("/reading", {{method: "POST", headers: {{"Content-Type": "application/json"}}, body: JSON.stringify(body)}})).json();
    out.textContent = reply.ok
      ? "Sent.\\n" + Object.entries(reply.status).map(([k, s]) => k + ": " + words[s]).join("\\n")
      : "Not sent: " + reply.error;
  }} catch (err) {{ out.textContent = "Could not reach the receiver: " + err; }}
}});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "AstroHealthTracker/1.0"

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data) -> None:
        self._send(code, json.dumps(data).encode("utf-8"), "application/json")

    def do_OPTIONS(self) -> None:  # noqa: N802 - name required by http.server
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, form_page().encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/params":
            self._json(200, {"astronauts": CREW_NAMES,
                             "parameters": {k: {"name": p.name, "unit": p.unit} for k, p in PARAMS.items()}})
        elif path == "/status":
            self._json(200, live_store.sources())
        else:
            self._json(404, {"ok": False, "error": "Not found. Try /, /params, /status or POST /reading."})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/reading":
            self._json(404, {"ok": False, "error": "POST readings to /reading."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY:
            self._json(400, {"ok": False, "error": "Send a small JSON body."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"ok": False, "error": "Body is not valid JSON."})
            return
        code, reply = handle_reading(payload)
        self._json(code, reply)

    def log_message(self, fmt: str, *args) -> None:
        print(f"  {self.address_string()}  {fmt % args}")


def lan_address() -> str:
    """This computer's address on the local network (no data is actually sent)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 80))
        return probe.getsockname()[0]
    except OSError:
        return "this-computer"
    finally:
        probe.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive tracker readings over HTTP for the dashboard")
    parser.add_argument("--lan", action="store_true", help="accept readings from other devices on the network")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    server = ThreadingHTTPServer((host, args.port), Handler)
    print(f"Tracker receiver running.\n  On this computer:  http://localhost:{args.port}")
    if args.lan:
        print(f"  On your phone:     http://{lan_address()}:{args.port}   (same Wi-Fi or hotspot)")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
