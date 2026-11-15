"""Webcam monitor: fatigue (eye closure) and experimental contactless heart rate.

Run this in its own terminal, next to the dashboard:

    python camera_monitor.py                            # default webcam
    python camera_monitor.py --camera 1                 # a different webcam
    python camera_monitor.py --astronaut "Asha Verma"   # also drive that crew member's tiles

With --astronaut, the measured heart rate and eye closure replace that crew
member's simulated values on the dashboard (tagged LIVE) while the camera runs.

It opens a preview window and writes the latest readings to camera_state.json
once a second. The dashboard's "Camera" section reads that file.
Press q in the preview window to quit.

First run downloads Google's face-landmark model (~4 MB) into models/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

import live_store
import simulator
from vision_metrics import LEFT_EYE, RIGHT_EYE, FatigueTracker, estimate_heart_rate, eye_aspect_ratio

HERE = Path(__file__).parent
STATE_FILE = HERE / "camera_state.json"
MODEL_PATH = HERE / "models" / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
HR_WINDOW_S = 12.0        # seconds of video used for each heart-rate estimate
MICROSLEEP_S = 1.5        # eyes closed this long = microsleep warning


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(exist_ok=True)
    print(f"Downloading face landmark model to {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - tell the user exactly what to do
        sys.exit(f"Could not download the model ({exc}).\n"
                 f"Download it manually from:\n  {MODEL_URL}\nand save it as {MODEL_PATH}")
    return MODEL_PATH


def write_state(state: dict) -> None:
    """Write atomically so the dashboard never reads a half-written file."""
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    live_store._replace(tmp, STATE_FILE)


def skin_regions(xy: np.ndarray):
    """Forehead and cheek boxes (pixel coords) relative to the face bounding box."""
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    w, h = x1 - x0, y1 - y0
    boxes = [
        (x0 + 0.30 * w, y0 + 0.06 * h, x0 + 0.70 * w, y0 + 0.20 * h),   # forehead
        (x0 + 0.14 * w, y0 + 0.52 * h, x0 + 0.34 * w, y0 + 0.70 * h),   # cheek
        (x0 + 0.66 * w, y0 + 0.52 * h, x0 + 0.86 * w, y0 + 0.70 * h),   # other cheek
    ]
    return [tuple(int(v) for v in b) for b in boxes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--no-window", action="store_true", help="run without the preview window")
    parser.add_argument("--astronaut", choices=[a.name for a in simulator.CREW],
                        help="show these readings as this crew member's live values")
    args = parser.parse_args()

    # Imported here so the rest of the project works without these installed.
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
    )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}. Try --camera 1, and check "
                 "that no other app is using the webcam.")

    fatigue = FatigueTracker()
    green_trace: deque[tuple[float, float]] = deque()
    heart_rate, hr_quality = None, 0.0
    start = time.monotonic()
    last_write = last_hr = 0.0
    print("Camera running. Look at the screen for 5 s to calibrate. Press q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame not received - stopping.")
                break
            now = time.monotonic() - start
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(now * 1000))
            face_found = bool(result.face_landmarks)
            ear = None

            if face_found:
                h, w = frame.shape[:2]
                xy = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]])
                ear = (eye_aspect_ratio(xy[LEFT_EYE]) + eye_aspect_ratio(xy[RIGHT_EYE])) / 2
                fatigue.update(now, ear)

                greens = []
                for (bx0, by0, bx1, by1) in skin_regions(xy):
                    patch = frame[max(by0, 0):by1, max(bx0, 0):bx1, 1]   # channel 1 = green (BGR)
                    if patch.size:
                        greens.append(float(patch.mean()))
                        cv2.rectangle(frame, (bx0, by0), (bx1, by1), (0, 200, 0), 1)
                if greens:
                    green_trace.append((now, float(np.mean(greens))))
                for i in LEFT_EYE + RIGHT_EYE:
                    cv2.circle(frame, (int(xy[i, 0]), int(xy[i, 1])), 1, (255, 200, 0), -1)
            else:
                green_trace.clear()   # a gap would corrupt the pulse estimate

            while green_trace and now - green_trace[0][0] > HR_WINDOW_S:
                green_trace.popleft()
            if now - last_hr >= 1.0 and green_trace:
                last_hr = now
                times, values = zip(*green_trace)
                fps = len(times) / max(times[-1] - times[0], 1e-6)
                heart_rate, hr_quality = estimate_heart_rate(times, values, fs=min(max(fps, 10.0), 60.0))

            closed_for = fatigue.closed_for(now) if face_found else 0.0
            if now - last_write >= 1.0:
                last_write = now
                write_state({
                    "timestamp": time.time(),
                    "face_found": face_found,
                    "calibrated": fatigue.calibrated,
                    "ear": None if ear is None else round(ear, 3),
                    "perclos": round(fatigue.perclos, 1),
                    "blink_rate": round(fatigue.blink_rate, 1),
                    "eyes_closed_s": round(closed_for, 1),
                    "microsleep": closed_for >= MICROSLEEP_S,
                    "heart_rate": None if heart_rate is None else round(heart_rate, 0),
                    "hr_quality": round(hr_quality, 2),
                })
                if args.astronaut and face_found:
                    live = {}
                    if fatigue.calibrated:
                        live["perclos"] = round(fatigue.perclos, 1)
                    if heart_rate is not None and hr_quality >= 0.3:
                        live["heart_rate"] = round(heart_rate)
                    if live:
                        live_store.write(args.astronaut, live, source="camera")

            if not args.no_window:
                lines = [
                    "No face found" if not face_found else
                    ("Calibrating..." if not fatigue.calibrated else f"PERCLOS {fatigue.perclos:4.1f}%   blinks/min {fatigue.blink_rate:4.1f}"),
                    "Heart rate: measuring..." if heart_rate is None else
                    f"Heart rate ~{heart_rate:3.0f} bpm (quality {hr_quality:.2f})",
                ]
                if closed_for >= MICROSLEEP_S:
                    lines.append("MICROSLEEP WARNING")
                for i, text in enumerate(lines):
                    cv2.putText(frame, text, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 0, 255) if "WARNING" in text else (255, 255, 255), 2)
                cv2.imshow("Crew camera monitor (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        STATE_FILE.unlink(missing_ok=True)   # so the dashboard shows "camera offline"


if __name__ == "__main__":
    main()
