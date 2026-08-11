"""
Passive face anti-spoofing (liveness) check using MiniFASNet -- a small
CNN trained specifically to tell a real face apart from a printed photo
or a phone/tablet screen held up to the camera.

Why this instead of blink detection: an earlier version of this project
tried spotting blinks via eye-cascade flicker, but a photo wobbling
slightly in someone's hand flickers the eye cascade too, so it wasn't
reliable enough. This uses an ensemble of two small CNNs (~1.7MB each),
the models behind the well-known Silent-Face-Anti-Spoofing project, which
were trained specifically to notice the texture/color/frequency artifacts
that give a print or screen away (moire patterns from screen refresh,
unnaturally flat/inaccurate color, paper or plastic texture instead of
skin) rather than relying on the subject doing something like blinking.
That makes it robust to a spoof attempt being held slightly unsteady --
it's judging the face itself, not motion.

Model credit: Silent-Face-Anti-Spoofing by Minivision AI (Apache-2.0)
https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
ONNX conversion: https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx

Requires the two .onnx files from that repo's onnx/ folder to be present
in config.ANTI_SPOOF_MODEL_DIR (shipped alongside this file), and:
    pip install onnxruntime
On 32-bit Raspberry Pi OS, onnxruntime doesn't always have a prebuilt
wheel on PyPI -- if `pip install onnxruntime` fails, either switch to
64-bit Raspberry Pi OS (recommended, and also just generally better for
the InsightFace side of this project) or install via piwheels.org.

Usage (see recognize.py):
    is_real, confidence, label = anti_spoof.check(frame, (left, top, right, bottom))
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

import config

_sessions = {}  # filename -> (onnxruntime session, scale)


def _load_sessions():
    if _sessions:
        return
    if not os.path.isdir(config.ANTI_SPOOF_MODEL_DIR):
        raise FileNotFoundError(
            f"Anti-spoof model directory not found: {config.ANTI_SPOOF_MODEL_DIR}\n"
            "Download the two .onnx files from https://github.com/QingHeYang/"
            f"Silent-Face-Anti-Spoofing-onnx/tree/main/onnx into {config.ANTI_SPOOF_MODEL_DIR}"
        )
    for fname in sorted(os.listdir(config.ANTI_SPOOF_MODEL_DIR)):
        if not fname.endswith(".onnx"):
            continue
        first_part = fname.split("_")[0]
        scale = 4.0 if first_part == "4" else float(first_part)
        path = os.path.join(config.ANTI_SPOOF_MODEL_DIR, fname)
        _sessions[fname] = (ort.InferenceSession(path, providers=["CPUExecutionProvider"]), scale)
    if not _sessions:
        raise FileNotFoundError(f"No .onnx models found in {config.ANTI_SPOOF_MODEL_DIR}")


def _expand_box(src_w, src_h, x, y, w, h, scale):
    """Same expansion logic the models were trained/converted with -- crop a
    square-ish region `scale`x the face box, centered on the face, clipped
    to the image."""
    scale = min((src_h - 1) / h, min((src_w - 1) / w, scale))
    new_w, new_h = w * scale, h * scale
    cx, cy = x + w / 2, y + h / 2
    l, t = cx - new_w / 2, cy - new_h / 2
    r, b = cx + new_w / 2, cy + new_h / 2
    if l < 0:
        r -= l
        l = 0
    if t < 0:
        b -= t
        t = 0
    if r > src_w - 1:
        l -= (r - src_w + 1)
        r = src_w - 1
    if b > src_h - 1:
        t -= (b - src_h + 1)
        b = src_h - 1
    return int(l), int(t), int(r), int(b)


def _preprocess(frame, bbox_xywh, scale):
    src_h, src_w = frame.shape[:2]
    x, y, w, h = bbox_xywh
    l, t, r, b = _expand_box(src_w, src_h, x, y, w, h, scale)
    crop = frame[t:b + 1, l:r + 1]
    if crop.size == 0:
        return None
    resized = cv2.resize(crop, (80, 80)).astype(np.float32)  # BGR, kept as-is (matches training)
    chw = np.transpose(resized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def check(frame, bbox_ltrb):
    """
    frame: the ORIGINAL (non-resized) BGR frame.
    bbox_ltrb: (left, top, right, bottom) in that frame's coordinates.

    Returns (is_real, confidence, label) where label is one of
    "real" / "paper" / "screen". If the models can't be loaded (e.g. the
    .onnx files are missing), this fails OPEN -- returns (True, 0.0,
    "unavailable") and prints a one-time warning -- so a broken/incomplete
    install doesn't lock everyone out of their own door. Fix the install
    if you see that warning; don't rely on the fail-open behavior.
    """
    try:
        _load_sessions()
    except FileNotFoundError as e:
        if not getattr(check, "_warned", False):
            print(f"[ANTI-SPOOF] Disabled: {e}")
            check._warned = True
        return True, 0.0, "unavailable"

    left, top, right, bottom = bbox_ltrb
    bbox_xywh = (max(left, 0), max(top, 0), right - left, bottom - top)
    if bbox_xywh[2] <= 0 or bbox_xywh[3] <= 0:
        return False, 0.0, "invalid-box"

    predictions = np.zeros((1, 3))
    for session, scale in _sessions.values():
        input_data = _preprocess(frame, bbox_xywh, scale)
        if input_data is None:
            return False, 0.0, "invalid-crop"
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_name: input_data})[0]
        exp = np.exp(output - np.max(output, axis=1, keepdims=True))
        softmax = exp / np.sum(exp, axis=1, keepdims=True)
        predictions += softmax

    predictions /= len(_sessions)
    label_idx = int(np.argmax(predictions[0]))
    scores = predictions[0]
    label_names = ["paper", "real", "screen"]  # fixed order the models were trained with
    return label_idx == 1, float(scores[label_idx]), label_names[label_idx]
