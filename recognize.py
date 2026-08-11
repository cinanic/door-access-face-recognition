"""
Live door-access recognition.

Reads the webcam/IP camera feed, matches every detected face against the
enrolled database (encodings.pkl + employees.json), and:

  - authorized match          -> opens the door, logs "granted"
  - recognized, not authorized -> keeps door locked, logs + sends warning
  - unknown face               -> keeps door locked, logs + sends warning

Every event is appended to access_log.json as one JSON object per line, e.g.:

    {"timestamp": "2026-07-27T10:15:32", "name": "Alice", "status": "granted", "confidence": 0.94, "reason": null}
    {"timestamp": "2026-07-27T10:16:05", "name": "Unknown", "status": "denied", "confidence": null, "reason": "unknown face", "snapshot": null}

By default this runs headless: no video window, no snapshot images -- just
the JSON log (and optional one-line console prints for each open/deny event).
Set config.SHOW_VIDEO_WINDOW / config.SAVE_SNAPSHOTS back to True if you want
those again.

Run:
    python recognize.py
Stop with Ctrl+C (or 'q' in the video window if SHOW_VIDEO_WINDOW is True).
"""

import csv
import json
import os
import pickle
import threading
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np

import config
import anti_spoof
from alerts import open_door, send_warning, init_gpio, cleanup_gpio
from enroll import get_face_app


def load_database():
    if not os.path.exists(config.ENCODINGS_FILE) or not os.path.exists(config.EMPLOYEES_JSON):
        raise FileNotFoundError(
            "No enrolled database found. Run enroll.py first "
            f"(expected {config.ENCODINGS_FILE} and {config.EMPLOYEES_JSON})."
        )
    with open(config.ENCODINGS_FILE, "rb") as f:
        encodings_db = pickle.load(f)
    with open(config.EMPLOYEES_JSON, "r") as f:
        employees_db = json.load(f)
    return encodings_db, employees_db


def flatten_encodings(encodings_db):
    """Turn {name: [enc, enc, ...]} into parallel lists for fast comparison."""
    names, encs = [], []
    for name, enc_list in encodings_db.items():
        for enc in enc_list:
            names.append(name)
            encs.append(enc)
    return names, np.array(encs) if encs else np.empty((0, 512))


def load_auth_csv():
    auth = {}
    if not os.path.exists(config.AUTH_CSV_FILE):
        return auth
    try:
        with open(config.AUTH_CSV_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "name" in row and "authorized" in row:
                    auth[row["name"].strip()] = row["authorized"].strip().lower() in ("true", "1", "yes", "on")
    except Exception as e:
        print(f"[AUTH] Error loading CSV: {e}")
    return auth


def is_authorized(name, employees_db, auth_csv):
    if config.AUTH_SOURCE == "csv" and auth_csv is not None:
        return auth_csv.get(name, config.DEFAULT_AUTHORIZED_IF_MISSING)
    return employees_db.get(name, {}).get("authorized", config.DEFAULT_AUTHORIZED_IF_MISSING)


def start_dashboard():
    if not config.DASHBOARD_ENABLED:
        return
    try:
        from dashboard import app
        def run():
            app.run(
                host=config.DASHBOARD_HOST,
                port=config.DASHBOARD_PORT,
                debug=False,
                use_reloader=False,
            )
        t = threading.Thread(target=run, daemon=True)
        t.start()
        print(f"[DASHBOARD] Web UI running on http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
    except Exception as e:
        print(f"[DASHBOARD] Could not start: {e}")


def log_event(entry):
    os.makedirs(os.path.dirname(config.ACCESS_LOG_JSON) or ".", exist_ok=True)
    line = json.dumps(entry)
    with open(config.ACCESS_LOG_JSON, "a") as f:
        f.write(line + "\n")


def match_face(face_embedding, known_names, known_encodings):
    """
    face_embedding and every row of known_encodings are L2-normalized 512-d
    InsightFace embeddings, so their dot product is the cosine similarity
    (1.0 = identical direction, higher = closer match) -- unlike the old
    face_recognition distance, higher is better here.
    """
    if len(known_encodings) == 0 or face_embedding is None:
        return None, None
    similarities = known_encodings @ face_embedding
    best_idx = int(np.argmax(similarities))
    best_similarity = float(similarities[best_idx])
    if best_similarity >= config.MATCH_TOLERANCE:
        return known_names[best_idx], round(best_similarity, 3)
    return None, None


def save_snapshot(frame, label):
    if not config.SAVE_SNAPSHOTS:
        return None
    os.makedirs(config.ALERTS_DIR, exist_ok=True)
    fname = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path = os.path.join(config.ALERTS_DIR, fname)
    cv2.imwrite(path, frame)
    return path


def main():
    encodings_db, employees_db = load_database()
    known_names, known_encodings = flatten_encodings(encodings_db)
    print(f"Loaded {len(encodings_db)} people, {len(known_names)} total face samples.")
    print(f"Mode: {'windowed' if config.SHOW_VIDEO_WINDOW else 'headless'}, "
          f"snapshots {'ON' if config.SAVE_SNAPSHOTS else 'OFF'}, "
          f"log -> {config.ACCESS_LOG_JSON}")

    face_app = get_face_app()  # loads/downloads the InsightFace model once, up front
    init_gpio()

    # Live auth reload setup
    auth_csv = load_auth_csv() if config.AUTH_SOURCE == "csv" else None
    last_auth_reload = time.time()

    start_dashboard()

    video = cv2.VideoCapture(config.CAMERA_SOURCE)
    if not video.isOpened():
        print(f"Could not open camera source: {config.CAMERA_SOURCE}")
        print("If this is an IP camera, double check the URL/port/path in a browser or VLC first.")
        return

    last_event_time = {}  # name -> last timestamp, for cooldown
    match_streak = defaultdict(int)  # key -> consecutive processed frames matched
    real_face_streak = defaultdict(int)  # key -> consecutive processed frames anti-spoof called "real"
    fake_face_streak = defaultdict(int)  # key -> consecutive processed frames anti-spoof called "paper"/"screen"
    frame_count = 0
    door_open_until = 0

    print("Starting recognition." + (" Press 'q' in the video window to quit." if config.SHOW_VIDEO_WINDOW else " Press Ctrl+C to quit."))
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                print("Lost camera stream, attempting to reconnect...")
                video.release()
                time.sleep(2)
                video = cv2.VideoCapture(config.CAMERA_SOURCE)
                continue

            frame_count += 1

            # Reload authorization CSV periodically so dashboard/hand edits apply live
            if config.AUTH_SOURCE == "csv" and (time.time() - last_auth_reload > config.AUTH_RELOAD_INTERVAL_SECONDS):
                auth_csv = load_auth_csv()
                last_auth_reload = time.time()

            display_frame = frame.copy() if config.SHOW_VIDEO_WINDOW else None

            if frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
                small = cv2.resize(frame, (0, 0), fx=config.FRAME_RESIZE_SCALE, fy=config.FRAME_RESIZE_SCALE)

                # InsightFace expects a BGR image (same as cv2.imread/VideoCapture
                # output), so no color-space conversion is needed here.
                faces = face_app.get(small)

                scale_back = 1.0 / config.FRAME_RESIZE_SCALE
                now = time.time()
                keys_seen_this_frame = set()

                if len(faces) > config.MAX_FACES_ALLOWED:
                    # Anti-tailgating: more than one person at the door -> never open,
                    # and wipe all in-progress streaks so nobody's partial progress
                    # carries over once the frame is back down to one person.
                    match_streak.clear()
                    for face in faces:
                        left, top, right, bottom = [int(v * scale_back) for v in face.bbox]
                        name, _ = match_face(face.normed_embedding, known_names, known_encodings)
                        if config.SHOW_VIDEO_WINDOW:
                            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 200), 2)
                            cv2.putText(display_frame, f"{name or 'Unknown'} - multiple people",
                                        (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

                    if now - last_event_time.get("multiple_people", 0) > config.COOLDOWN_SECONDS:
                        last_event_time["multiple_people"] = now
                        snapshot_path = save_snapshot(frame, "multiple_people")
                        send_warning("Multiple people", f"{len(faces)} faces detected at once - access denied")
                        log_event({
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "name": "Multiple", "status": "denied", "confidence": None,
                            "reason": f"{len(faces)} people detected at once",
                            "snapshot": snapshot_path,
                        })

                else:
                    for face in faces:
                        left, top, right, bottom = [int(v * scale_back) for v in face.bbox]

                        name, confidence = match_face(face.normed_embedding, known_names, known_encodings)
                        authorized = name is not None and is_authorized(name, employees_db, auth_csv)
                        key = name if authorized else (f"not_allowed:{name}" if name is not None else "unknown")

                        keys_seen_this_frame.add(key)
                        match_streak[key] += 1
                        streak_ok = match_streak[key] >= config.CONSECUTIVE_MATCHES_REQUIRED

                        if authorized:
                            if streak_ok:
                                is_real, spoof_conf, spoof_label = anti_spoof.check(
                                    frame, (left, top, right, bottom)
                                )
                                if is_real and spoof_conf >= config.ANTI_SPOOF_REAL_THRESHOLD:
                                    real_face_streak[key] += 1
                                    fake_face_streak[key] = 0
                                elif not is_real and spoof_conf >= config.ANTI_SPOOF_FAKE_THRESHOLD:
                                    fake_face_streak[key] += 1
                                    real_face_streak[key] = 0
                                # else: low-confidence/ambiguous frame -- leave both streaks as-is
                            else:
                                spoof_label = None

                            live = real_face_streak[key] >= config.ANTI_SPOOF_CONSECUTIVE_REAL_REQUIRED
                            spoof_suspected = fake_face_streak[key] >= config.ANTI_SPOOF_CONSECUTIVE_FAKE_TO_DENY

                            if config.SHOW_VIDEO_WINDOW:
                                box_color = (0, 200, 0)
                                if not streak_ok:
                                    suffix = f" [{match_streak[key]}/{config.CONSECUTIVE_MATCHES_REQUIRED}]"
                                elif not live:
                                    suffix = f" [anti-spoof: {spoof_label or '...'}]"
                                else:
                                    suffix = ""
                                label = f"{name} ({confidence}){suffix}"

                            if live and now - last_event_time.get(key, 0) > config.COOLDOWN_SECONDS:
                                last_event_time[key] = now
                                open_door()
                                door_open_until = now + config.DOOR_OPEN_SECONDS
                                log_event({
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                    "name": name, "status": "granted",
                                    "confidence": confidence, "reason": None,
                                })
                                real_face_streak[key] = 0
                                fake_face_streak[key] = 0
                            elif spoof_suspected and now - last_event_time.get(key, 0) > config.COOLDOWN_SECONDS:
                                last_event_time[key] = now
                                snapshot_path = save_snapshot(frame, f"spoof_suspected_{name}")
                                send_warning(name, f"face matched but anti-spoofing flagged as '{spoof_label}' - access denied")
                                log_event({
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                    "name": name, "status": "denied", "confidence": confidence,
                                    "reason": f"anti-spoof check flagged as {spoof_label}", "snapshot": snapshot_path,
                                })
                                real_face_streak[key] = 0
                                fake_face_streak[key] = 0
                        elif name is not None:
                            if config.SHOW_VIDEO_WINDOW:
                                box_color = (0, 0, 200)
                                label = f"{name} (not allowed here)"
                            if streak_ok and now - last_event_time.get(key, 0) > config.COOLDOWN_SECONDS:
                                last_event_time[key] = now
                                snapshot_path = save_snapshot(frame, f"not_allowed_{name}")
                                send_warning(name, "recognized employee, not authorized for this room")
                                log_event({
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                    "name": name, "status": "denied", "confidence": confidence,
                                    "reason": "recognized, not authorized for this room", "snapshot": snapshot_path,
                                })
                        else:
                            if config.SHOW_VIDEO_WINDOW:
                                box_color = (0, 0, 200)
                                label = "Unknown"
                            if streak_ok and now - last_event_time.get(key, 0) > config.COOLDOWN_SECONDS:
                                last_event_time[key] = now
                                snapshot_path = save_snapshot(frame, "unknown")
                                send_warning("Unknown", "unknown face")
                                log_event({
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                    "name": "Unknown", "status": "denied", "confidence": None,
                                    "reason": "unknown face", "snapshot": snapshot_path,
                                })

                        if config.SHOW_VIDEO_WINDOW:
                            cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)
                            cv2.putText(display_frame, label, (left, top - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                # Keys not seen this pass break their streak (blip/turned away/left frame)
                for k in list(match_streak.keys()):
                    if k not in keys_seen_this_frame:
                        match_streak[k] = 0
                        real_face_streak[k] = 0
                        fake_face_streak[k] = 0

            if config.SHOW_VIDEO_WINDOW:
                door_status = "OPEN" if time.time() < door_open_until else "LOCKED"
                cv2.putText(display_frame, f"Door: {door_status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 200, 0) if door_status == "OPEN" else (0, 0, 200), 2)
                cv2.imshow("Door Access - press q to quit", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")
    finally:
        video.release()
        cleanup_gpio()
        if config.SHOW_VIDEO_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
