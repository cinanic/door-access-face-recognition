"""
Live door-access recognition.

Reads the USB webcam feed, matches every detected face against the enrolled
database (encodings.pkl + employees.json), and:

  - authorized match      -> opens the door, logs "granted"
  - recognized but revoked -> keeps door locked, logs + sends warning
  - unknown face           -> keeps door locked, logs + sends warning, saves a snapshot

Every event is appended to access_log.json as one JSON object per entry, e.g.:

    {
      "timestamp": "2026-07-27T10:15:32",
      "name": "Alice",
      "status": "granted",
      "confidence": 0.94,
      "reason": null
    }
    {
      "timestamp": "2026-07-27T10:16:05",
      "name": "Unknown",
      "status": "denied",
      "confidence": null,
      "reason": "unknown face",
      "snapshot": "alerts/unknown_20260727_101605.jpg"
    }

Run:
    python recognize.py
Press 'q' in the video window to quit.
"""

import json
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np

import config
from alerts import open_door, send_warning
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
    os.makedirs(config.ALERTS_DIR, exist_ok=True)
    fname = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path = os.path.join(config.ALERTS_DIR, fname)
    cv2.imwrite(path, frame)
    return path


def main():
    encodings_db, employees_db = load_database()
    known_names, known_encodings = flatten_encodings(encodings_db)
    print(f"Loaded {len(encodings_db)} people, {len(known_names)} total face samples.")

    face_app = get_face_app()  # loads/downloads the InsightFace model once, up front

    video = cv2.VideoCapture(config.CAMERA_SOURCE)
    if not video.isOpened():
        print(f"Could not open camera source: {config.CAMERA_SOURCE}")
        print("If this is an IP camera, double check the URL/port/path in a browser or VLC first.")
        return

    last_event_time = {}  # name -> last timestamp, for cooldown
    match_streak = defaultdict(int)  # key -> consecutive processed frames matched
    frame_count = 0
    door_open_until = 0

    print("Starting recognition. Press 'q' to quit.")
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
            display_frame = frame.copy()

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
                        authorized = name is not None and employees_db.get(name, {}).get("authorized", False)
                        key = name if authorized else (f"not_allowed:{name}" if name is not None else "unknown")

                        keys_seen_this_frame.add(key)
                        match_streak[key] += 1
                        streak_ok = match_streak[key] >= config.CONSECUTIVE_MATCHES_REQUIRED

                        if authorized:
                            box_color = (0, 200, 0)
                            suffix = "" if streak_ok else f" [{match_streak[key]}/{config.CONSECUTIVE_MATCHES_REQUIRED}]"
                            label = f"{name} ({confidence}){suffix}"
                            if streak_ok and now - last_event_time.get(key, 0) > config.COOLDOWN_SECONDS:
                                last_event_time[key] = now
                                open_door()
                                door_open_until = now + config.DOOR_OPEN_SECONDS
                                log_event({
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                    "name": name, "status": "granted",
                                    "confidence": confidence, "reason": None,
                                })
                        elif name is not None:
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

                        cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)
                        cv2.putText(display_frame, label, (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                # Keys not seen this pass break their streak (blip/turned away/left frame)
                for k in list(match_streak.keys()):
                    if k not in keys_seen_this_frame:
                        match_streak[k] = 0

            door_status = "OPEN" if time.time() < door_open_until else "LOCKED"
            cv2.putText(display_frame, f"Door: {door_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 200, 0) if door_status == "OPEN" else (0, 0, 200), 2)

            cv2.imshow("Door Access - press q to quit", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
