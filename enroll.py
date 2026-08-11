"""
Build (or refresh) the face database from your employee images/videos.

Expected folder layout:

    data/employees/
        Alice/
            photo1.jpg
            photo2.png
            badge_video.mp4
            access.yaml       <- optional, controls door access for Alice
        Bob/
            id_photo.jpg
            access.yaml

Each employee's access.yaml just contains:

    authorized: true

(or false). If a folder has no access.yaml, that person defaults to
config.DEFAULT_AUTHORIZED_IF_MISSING (False by default -- fail closed, so
new hires are locked out until someone explicitly authorizes them).

Run:
    python enroll.py

Produces:
    encodings.pkl   -- {name: [list of 128-d face encodings]}
    employees.json  -- {name: {"authorized": true, "num_samples": N, "sources": [...]}}

employees.json is now a GENERATED file -- authorization is read fresh from
each employee's access.yaml every run, so employees.json always reflects
whatever access.yaml currently says. Don't hand-edit employees.json anymore;
edit access.yaml instead and re-run enroll.py.
"""

import json
import os
import pickle

import cv2
import yaml
from insightface.app import FaceAnalysis

import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

_face_app = None


def get_face_app():
    """
    Returns the shared InsightFace FaceAnalysis app, building it on first
    call (this is what triggers the one-time model download to
    ~/.insightface/models, so import time stays fast). recognize.py imports
    this same function so enrollment and live recognition share one loaded
    model instead of loading it twice.
    """
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(
            name=config.INSIGHTFACE_MODEL_NAME,
            allowed_modules=["detection", "recognition"],
        )
        _face_app.prepare(
            ctx_id=config.INSIGHTFACE_CTX_ID,
            det_size=config.INSIGHTFACE_DET_SIZE,
            det_thresh=config.INSIGHTFACE_DET_THRESH,
        )
    return _face_app


def _detect_encodings(bgr_image):
    """
    Runs detection + recognition on a BGR image (as returned by cv2.imread /
    VideoCapture.read) and returns the normalized 512-d embedding for every
    detected face that passes the minimum-size filter.
    """
    faces = get_face_app().get(bgr_image)
    encs = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        if (x2 - x1) < config.MIN_FACE_SIZE_PX:
            continue
        if face.normed_embedding is None:
            continue
        encs.append(face.normed_embedding)
    return encs


def encodings_from_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"  [!] could not read {path}, skipping")
        return []
    encs = _detect_encodings(image)
    if not encs:
        print(f"  [!] no face found in {path}, skipping")
    return encs


def encodings_from_video(path, num_samples=config.VIDEO_SAMPLE_FRAMES):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        print(f"  [!] could not read {path}, skipping")
        return []

    sample_indices = sorted(set(
        int(i * total_frames / max(num_samples, 1)) for i in range(num_samples)
    ))

    encs = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        encs.extend(_detect_encodings(frame))
    cap.release()
    return encs


def load_existing_employees_json():
    if os.path.exists(config.EMPLOYEES_JSON):
        with open(config.EMPLOYEES_JSON, "r") as f:
            return json.load(f)
    return {}


def read_authorized_flag(person_dir, name):
    """Reads <person_dir>/access.yaml -> {"authorized": true|false}."""
    path = os.path.join(person_dir, config.ACCESS_CONFIG_FILENAME)
    if not os.path.exists(path):
        print(f"  [!] no {config.ACCESS_CONFIG_FILENAME} for '{name}', "
              f"defaulting authorized={config.DEFAULT_AUTHORIZED_IF_MISSING}")
        return config.DEFAULT_AUTHORIZED_IF_MISSING

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        if "authorized" not in data:
            print(f"  [!] {path} has no 'authorized' key, "
                  f"defaulting authorized={config.DEFAULT_AUTHORIZED_IF_MISSING}")
            return config.DEFAULT_AUTHORIZED_IF_MISSING
        return bool(data["authorized"])
    except Exception as e:
        print(f"  [!] could not parse {path} ({e}), "
              f"defaulting authorized={config.DEFAULT_AUTHORIZED_IF_MISSING}")
        return config.DEFAULT_AUTHORIZED_IF_MISSING


def main():
    if not os.path.isdir(config.EMPLOYEES_DIR):
        print(f"Expected folder not found: {config.EMPLOYEES_DIR}")
        return

    all_encodings = {}
    employees_json = load_existing_employees_json()

    people = sorted(
        d for d in os.listdir(config.EMPLOYEES_DIR)
        if os.path.isdir(os.path.join(config.EMPLOYEES_DIR, d))
    )

    if not people:
        print(f"No employee subfolders found in {config.EMPLOYEES_DIR}")
        return

    for name in people:
        person_dir = os.path.join(config.EMPLOYEES_DIR, name)
        print(f"Enrolling '{name}'...")
        person_encodings = []
        sources = []

        for fname in sorted(os.listdir(person_dir)):
            fpath = os.path.join(person_dir, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext in IMAGE_EXTS:
                encs = encodings_from_image(fpath)
                if encs:
                    person_encodings.extend(encs)
                    sources.append(fname)
            elif ext in VIDEO_EXTS:
                encs = encodings_from_video(fpath)
                if encs:
                    person_encodings.extend(encs)
                    sources.append(fname)
            else:
                continue

        if not person_encodings:
            print(f"  [!] no usable faces found for '{name}', skipping entirely")
            continue

        all_encodings[name] = person_encodings

        authorized = read_authorized_flag(person_dir, name)
        employees_json[name] = {
            "authorized": authorized,
            "num_samples": len(person_encodings),
            "sources": sources,
        }
        print(f"  -> {len(person_encodings)} face samples collected, authorized={authorized}")

    with open(config.ENCODINGS_FILE, "wb") as f:
        pickle.dump(all_encodings, f)

    with open(config.EMPLOYEES_JSON, "w") as f:
        json.dump(employees_json, f, indent=2)

    print(f"\nDone. {len(all_encodings)} people enrolled.")
    print(f"Encodings saved to: {config.ENCODINGS_FILE}")
    print(f"Metadata saved to:  {config.EMPLOYEES_JSON}")
    print(f"To change who can open the door, edit each person's "
          f"{config.ACCESS_CONFIG_FILENAME} and re-run enroll.py.")


if __name__ == "__main__":
    main()
