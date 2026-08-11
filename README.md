# Face Recognition Door Access

Enrolls employees from photos/videos, then watches a USB webcam feed and
opens the door only for authorized faces. Every attempt (granted or denied)
is logged to a JSON file, and denied attempts trigger a warning + snapshot.

**Quickest start:** once your data is in place (step 2 below),
```bash
python run.py
```
runs enrollment and then immediately starts live recognition in one command
— see section 6a. The steps below walk through what each part does, in case
you want to run enrollment and recognition separately (e.g. re-enrolling
without restarting the camera).

## 1. Install

```bash
pip install -r requirements.txt
```

Face detection/recognition is powered by
[InsightFace](https://github.com/deepinsight/insightface) (a lightweight
SCRFD detector + MobileFaceNet embeddings via ONNX Runtime) — no C++
compiler or `dlib` build step needed. The first time you run `enroll.py`,
`recognize.py`, or `run.py`, InsightFace automatically downloads the
`buffalo_s` model pack (~125 MB) from GitHub to `~/.insightface/models`, so
**that first run needs internet access**; every run after that uses the
local copy.
**Orange Pi Zero 2 W note (512MB RAM):** this repo is already configured for
`buffalo_s` (not the heavier `buffalo_l`) and a smaller 320x320 detector
size specifically because the Zero 2 W's RAM is tight enough that the
larger model gets killed by the OOM killer (you'll just see `Killed` printed
with no Python traceback — that's the OS, not a bug). If you still hit
that, add some swap so a memory spike gets paged instead of killing the
process:
```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```
Swap on an SD card is slow, so treat it as a safety net rather than a
performance fix — if enrollment/recognition is still crawling, that's the
signal to trim `INSIGHTFACE_DET_SIZE` further (e.g. `(256, 256)`) or enroll
fewer/smaller sample images at a time.

On the Orange Pi Zero 2W (aarch64), plain `pip install onnxruntime` may not
have a prebuilt wheel for your OS/architecture. If it fails, try
[piwheels](https://www.piwheels.org/project/onnxruntime/) or search for a
community-built aarch64 wheel matching your OS and Python version.

**Anti-spoofing model files:** the anti-spoofing check (section 5a) needs
two small `.onnx` files that aren't distributed via pip. Download them from
[Silent-Face-Anti-Spoofing-onnx](https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/tree/main/onnx)
into an `antispoof_models/` folder next to `config.py`:

```
antispoof_models/
    2.7_80x80_MiniFASNetV2.onnx
    4_0_0_80x80_MiniFASNetV1SE.onnx
```

If this folder/files are missing, `recognize.py` still runs, but the door
opens for anyone who matches a face regardless of photo/screen spoofing —
it prints a one-time `[ANTI-SPOOF] Disabled: ...` warning so you notice.

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

## 5a. Anti-spoofing (no photos/screens)

A face match alone just proves "this looks like an authorized person" — it
doesn't prove a live person is standing there. A printed photo or a phone/
tablet screen held up to the camera matches face encodings just as well as
the real person does.

To stop that, once a face's identity match is confirmed the frame is also
run through a small pretrained CNN ([MiniFASNet](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing),
Apache-2.0) that looks for the texture/color/frequency artifacts a print or
screen gives off — moire patterns from a screen's refresh, unnaturally flat
or inaccurate color from a printer, paper/plastic texture instead of skin —
rather than relying on the person doing something like blinking (which
turned out unreliable: a photo wobbling in someone's hand can flicker a
blink-detector too). See `anti_spoof.py` for details.

The door only opens once several consecutive frames agree the face is real
(`ANTI_SPOOF_CONSECUTIVE_REAL_REQUIRED`, default 3). If several consecutive
frames instead confidently flag it as a paper photo or screen
(`ANTI_SPOOF_CONSECUTIVE_FAKE_TO_DENY`, default 3), it's logged/alerted as a
denied attempt with reason `"anti-spoof check flagged as paper"` (or
`"screen"`), same as any other denial.

Tuning, in `config.py`:
- `ANTI_SPOOF_REAL_THRESHOLD` / `ANTI_SPOOF_FAKE_THRESHOLD` — confidence a
  single frame needs to count toward its streak. Lower `ANTI_SPOOF_REAL_THRESHOLD`
  if legitimate people are getting denied; raise it if spoofs are getting
  through.
- `ANTI_SPOOF_CONSECUTIVE_REAL_REQUIRED` / `ANTI_SPOOF_CONSECUTIVE_FAKE_TO_DENY`
  — how many frames in a row are needed either way.
- `ANTI_SPOOF_ENABLED = False` turns the whole check off if you want identity
  matching alone (not recommended for a door, but useful while testing).

**Known limitation:** this catches prints and static screen images. It does
not catch a video replay attack (e.g. a phone playing a video of the real
person's face). That needs depth or IR hardware, which is out of scope for
a single webcam.

## 6. Run live recognition

```bash
python recognize.py
```

By default this runs **headless**: no video window, no snapshot images —
just the JSON log (plus short one-line console prints for each
open/deny event). Stop it with `Ctrl+C`.

If you want the live video window with bounding boxes back (useful while
tuning), set in `config.py`:

```python
SHOW_VIDEO_WINDOW = True   # shows the annotated video feed, quit with 'q'
SAVE_SNAPSHOTS = True      # saves a .jpg per denied attempt under alerts/
```

`PRINT_EVENTS_TO_CONSOLE` (default `True`) controls the one-line console
prints independently of the above two — set it to `False` for a fully
silent run where only `access_log.json` records anything.

## 6a. One command instead of two: `run.py`

```bash
python run.py
```

Runs enrollment first, then immediately starts live recognition in the
same process — no separate `python enroll.py` step needed. This is what you
want for a systemd service or a startup script on the Pi: one command boots
the whole thing. If enrollment finds no usable employee data, it prints why
and exits instead of trying to start recognition against an empty database.

Use `enroll.py` and `recognize.py` separately instead when you want to
re-enroll (e.g. added a new employee) without restarting a recognition
session that's already running.

## 7. Access log

Every event is appended as one JSON object per line to `access_log.json`:

```json
{"timestamp": "2026-07-27T10:15:32", "name": "Alice", "status": "granted", "confidence": 0.94, "reason": null}
{"timestamp": "2026-07-27T10:16:05", "name": "Unknown", "status": "denied", "confidence": null, "reason": "unknown face", "snapshot": "alerts/unknown_20260727_101605.jpg"}
{"timestamp": "2026-07-27T10:18:41", "name": "Alice", "status": "denied", "confidence": 0.91, "reason": "anti-spoof check flagged as paper", "snapshot": "alerts/spoof_suspected_Alice_20260727_101841.jpg"}
```

Denied attempts also save a snapshot image under `alerts/`.

## 8. Wiring up real hardware / notifications

### Door relay via Orange Pi GPIO

This is built in and on by default (`config.GPIO_ENABLED = True`), matching
the two-relay sequence your electrical engineer specified: one relay pulses
to open the door, then after a hold the other relay pulses to close it.

```python
# config.py
GPIO_ENABLED = True
GPIO_MODE = "BCM"
GPIO_PIN_OPEN = 11       # BCM 11 == PI11 -- pulses HIGH to open
GPIO_PIN_CLOSE = 12      # BCM 12 == PI12 -- pulses HIGH to close

GPIO_OPEN_PULSE_SECONDS = 3    # how long the open pulse lasts
GPIO_HOLD_SECONDS = 6          # both pins LOW in between (door sits open)
GPIO_CLOSE_PULSE_SECONDS = 3   # how long the close pulse lasts
```

On a granted match, `open_door()` in `alerts.py` runs exactly this sequence:

```
GPIO_PIN_OPEN  -> HIGH for GPIO_OPEN_PULSE_SECONDS
both pins      -> LOW  for GPIO_HOLD_SECONDS
GPIO_PIN_CLOSE -> HIGH for GPIO_CLOSE_PULSE_SECONDS
both pins      -> LOW  (idle)
```

The whole ~12 second sequence runs on a **background thread**, so the
camera loop keeps reading frames the entire time instead of freezing. If a
new match comes in while a cycle is already running, it's ignored with a
log line (`"Door cycle already in progress"`) rather than starting a second
overlapping sequence on the same pins.

On the Orange Pi:
```bash
pip install OPi.GPIO
```
(Don't install `OPi.GPIO` on a regular laptop — it's Orange Pi-specific and
will fail there. It also needs the wiringOP driver present on the device.)
If it's not installed or you run `recognize.py` on a non-Orange Pi machine,
`open_door()` automatically falls back to printing what it would have done,
so you can develop/test everything else off-Pi first, then deploy to the
Orange Pi with zero code changes.

### USB relay / Arduino over serial

If you're not using GPIO directly — e.g. a USB relay board or an Arduino —
replace the GPIO calls in `alerts.py`'s `open_door()` with something like:

```python
import serial
ser = serial.Serial('/dev/ttyUSB0', 9600)
ser.write(b'OPEN\n')
```

### Email/notification alerts

`send_warning()` currently prints. Set `ENABLE_EMAIL_ALERTS = True` in
`config.py` and fill in your SMTP details to also email security. You can
swap in Slack/Telegram/SMS calls here instead just as easily.

## 9. Tuning

All thresholds live in `config.py`:
- `MATCH_TOLERANCE` — InsightFace embeddings are compared by cosine similarity, so **higher = stricter** matching here (fewer false accepts, more false rejects). Default 0.42.
- `COOLDOWN_SECONDS` — avoids re-triggering/re-alerting for the same person every frame.
- `PROCESS_EVERY_N_FRAMES` / `FRAME_RESIZE_SCALE` — trade accuracy for CPU/speed.
- `INSIGHTFACE_DET_SIZE` — detector input resolution; larger catches smaller/farther faces but costs more CPU per frame.
- `ANTI_SPOOF_REAL_THRESHOLD` / `ANTI_SPOOF_CONSECUTIVE_REAL_REQUIRED` — how strict the photo/screen spoof check is (see section 5a).

## Notes on accuracy

- Good, front-facing, well-lit enrollment photos matter more than model choice.
- If you get false accepts between look-alike employees, lower `MATCH_TOLERANCE`.
- If legitimate employees keep getting rejected, raise `MATCH_TOLERANCE` slightly
  or add more/varied enrollment samples for them.

### Only one person enrolled, but strangers sometimes get let in

With a single-person database there's nothing for the matcher to actively
tell "not you" apart from — it's just a similarity threshold, so this is the
most false-accept-prone setup. Two things now help with this directly:

- `MATCH_TOLERANCE` defaults to `0.42`. If strangers still get matched,
  raise it further in `0.02` steps until they stop — just re-test that you
  still get recognized reliably each time.
- `CONSECUTIVE_MATCHES_REQUIRED` (default `4`) now requires several
  processed frames in a row to agree before the door opens, instead of
  trusting a single frame. This alone should remove most one-off misfires
  caused by a bad angle or lighting flicker.

Also worth doing: re-run `enroll.py` with a few more/varied photos or a
slightly longer video of yourself (different angles, with/without glasses,
different lighting) — more good samples of *you* makes your true matches
land closer to 0 distance, which widens the safety margin against strangers
without needing an ultra-strict tolerance.
