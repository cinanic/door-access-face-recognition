# Face Recognition Door Access

Enrolls employees from photos/videos, then watches a USB webcam feed and
opens the door only for authorized faces. Every attempt (granted or denied)
is logged to a JSON file, and denied attempts trigger a warning + snapshot.

## 1. Install

```bash
pip install -r requirements.txt
```

`face_recognition` depends on `dlib`, which needs `cmake` and a C++ compiler
to build. On Ubuntu/Debian:

```bash
sudo apt-get install -y cmake build-essential
```

On Windows, installing dlib via pip can be painful — using
`conda install -c conda-forge dlib` first is usually easier.

## 2. Add your data

```
data/employees/
    Alice/
        photo1.jpg
        photo2.png
        badge_video.mp4
    Bob/
        id_photo.jpg
```

- One folder per employee, folder name = display name.
- Mix images and short videos freely; more/varied samples = better accuracy
  (different angles, lighting, with/without glasses).
- Each image should contain exactly one clear face.

## 3. Enroll

```bash
python enroll.py
```

This creates:
- `encodings.pkl` — face encodings per person (not human-readable, used internally)
- `employees.json` — per-person metadata, e.g.:

```json
{
  "Alice": { "authorized": true, "num_samples": 42, "sources": ["photo1.jpg", "badge_video.mp4"] },
  "Bob":   { "authorized": true, "num_samples": 6,  "sources": ["id_photo.jpg"] }
}
```

To revoke someone's access (e.g. they left the company but you keep the
enrollment data for records), just set `"authorized": false` in
`employees.json`. Re-running `enroll.py` later will NOT overwrite that flag.

## 4. Enroll everyone, but the door only opens for allowed people

You can enroll your entire employee list — everyone gets recognized by name —
while only a subset can actually open this specific door. This is controlled
by an `access.yaml` file placed inside each employee's own folder:

```
data/employees/
    Alice/
        photo1.jpg
        access.yaml     <-- authorized: true
    Bob/
        id_photo.jpg
        access.yaml     <-- authorized: false
```

`access.yaml` contents:

```yaml
authorized: true
```

- `authorized: true` → door opens for them.
- `authorized: false` → they are still **recognized by name** (not treated
  as unknown), but the door stays locked and a warning fires with reason
  `"recognized, not authorized for this room"`. Useful for people who work
  in the building but shouldn't have access to this particular room.
- **No `access.yaml` at all** → defaults to `authorized: false` (fail
  closed) — a newly added employee folder is locked out until someone
  deliberately adds `access.yaml` with `authorized: true`. You can flip this
  default in `config.py` (`DEFAULT_AUTHORIZED_IF_MISSING`) if you'd rather
  new employees default to allowed.
- A face that matches no one at all is logged as `"unknown face"` instead.

`employees.json` is now a **generated** file — every run of `enroll.py`
re-reads each `access.yaml` and rebuilds `employees.json` to match. Don't
hand-edit `employees.json` anymore; edit `access.yaml` per person instead,
then re-run `enroll.py` to apply it.

## 5. Anti-tailgating (no shared entries)

If more than one face is detected in frame at the same time, the door will
**never** open — even if one of the faces is an authorized person. This stops
someone piggybacking in behind an authorized employee. Controlled by
`MAX_FACES_ALLOWED` in `config.py` (default `1`). Each such event is logged
with reason `"N people detected at once"` and a snapshot is saved.

## 6. Run live recognition

```bash
python recognize.py
```

- Green box = recognized + authorized -> door opens.
- Red box = recognized but revoked, OR unknown face -> door stays locked and
  a warning fires.
- Press `q` in the video window to quit.

## 7. Access log

Every event is appended as one JSON object per line to `access_log.json`:

```json
{"timestamp": "2026-07-27T10:15:32", "name": "Alice", "status": "granted", "confidence": 0.94, "reason": null}
{"timestamp": "2026-07-27T10:16:05", "name": "Unknown", "status": "denied", "confidence": null, "reason": "unknown face", "snapshot": "alerts/unknown_20260727_101605.jpg"}
```

Denied attempts also save a snapshot image under `alerts/`.

## 8. Wiring up real hardware / notifications

Edit `alerts.py`:
- `open_door()` — currently just prints. Example code for a Raspberry Pi GPIO
  relay or a USB/Arduino serial relay is included as comments — uncomment and
  adjust pins/port for your hardware.
- `send_warning()` — currently prints. Set `ENABLE_EMAIL_ALERTS = True` in
  `config.py` and fill in your SMTP details to also email security. You can
  swap in Slack/Telegram/SMS calls here instead just as easily.

## 9. Tuning

All thresholds live in `config.py`:
- `MATCH_TOLERANCE` — lower = stricter matching (fewer false accepts, more false rejects). Default 0.5.
- `COOLDOWN_SECONDS` — avoids re-triggering/re-alerting for the same person every frame.
- `PROCESS_EVERY_N_FRAMES` / `FRAME_RESIZE_SCALE` — trade accuracy for CPU/speed.

## Notes on accuracy

- Good, front-facing, well-lit enrollment photos matter more than model choice.
- If you get false accepts between look-alike employees, lower `MATCH_TOLERANCE`.
- If legitimate employees keep getting rejected, raise `MATCH_TOLERANCE` slightly
  or add more/varied enrollment samples for them.

### Only one person enrolled, but strangers sometimes get let in

With a single-person database there's nothing for the matcher to actively
tell "not you" apart from — it's just a distance threshold, so this is the
most false-accept-prone setup. Two things now help with this directly:

- `MATCH_TOLERANCE` was tightened to `0.45` (from the default `0.5`). If
  strangers still get matched, lower it further in `0.02` steps until they
  stop — just re-test that you still get recognized reliably each time.
- `CONSECUTIVE_MATCHES_REQUIRED` (default `4`) now requires several
  processed frames in a row to agree before the door opens, instead of
  trusting a single frame. This alone should remove most one-off misfires
  caused by a bad angle or lighting flicker.

Also worth doing: re-run `enroll.py` with a few more/varied photos or a
slightly longer video of yourself (different angles, with/without glasses,
different lighting) — more good samples of *you* makes your true matches
land closer to 0 distance, which widens the safety margin against strangers
without needing an ultra-strict tolerance.
