"""
Warning / alert delivery and door actuation.

open_door() runs a two-relay open/close sequence on real Raspberry Pi GPIO
pins, matching this hardware design from the electrical engineer:

    GPIO_PIN_OPEN  -> HIGH for GPIO_OPEN_PULSE_SECONDS   (pulses the door open)
    both pins      -> LOW  for GPIO_HOLD_SECONDS          (door sits open)
    GPIO_PIN_CLOSE -> HIGH for GPIO_CLOSE_PULSE_SECONDS   (pulses the door closed)
    both pins      -> LOW                                 (idle)

The whole sequence runs on a background thread so it doesn't block the
camera loop for the ~12 seconds it takes. If OPi.GPIO isn't installed/
available -- e.g. you're testing this on a laptop before deploying to the
Pi -- it automatically falls back to just printing what it would have done,
so the rest of the code needs no changes between dev and the Pi.
"""

import smtplib
import threading
import time
from email.mime.text import MIMEText

import config

try:
    import OPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    # RuntimeError covers "not running on a Raspberry Pi" on some platforms
    _GPIO_AVAILABLE = False

_gpio_ready = False
_door_busy = threading.Event()  # true while an open/close cycle is running


def init_gpio():
    """Call once at program startup (recognize.py does this). Safe to call more than once."""
    global _gpio_ready
    if not config.GPIO_ENABLED or not _GPIO_AVAILABLE or _gpio_ready:
        return
    GPIO.setmode(GPIO.BCM if config.GPIO_MODE == "BCM" else GPIO.BOARD)
    GPIO.setup(config.GPIO_PIN_OPEN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(config.GPIO_PIN_CLOSE, GPIO.OUT, initial=GPIO.LOW)
    _gpio_ready = True
    print(f"[GPIO] Initialized pins open={config.GPIO_PIN_OPEN} "
          f"close={config.GPIO_PIN_CLOSE} ({config.GPIO_MODE}).")


def cleanup_gpio():
    """Call once at program shutdown (recognize.py does this in its finally block)."""
    if _GPIO_AVAILABLE and _gpio_ready:
        GPIO.cleanup()


def _run_door_cycle():
    try:
        GPIO.output(config.GPIO_PIN_OPEN, GPIO.HIGH)
        GPIO.output(config.GPIO_PIN_CLOSE, GPIO.LOW)
        time.sleep(config.GPIO_OPEN_PULSE_SECONDS)

        GPIO.output(config.GPIO_PIN_OPEN, GPIO.LOW)
        GPIO.output(config.GPIO_PIN_CLOSE, GPIO.LOW)
        time.sleep(config.GPIO_HOLD_SECONDS)

        GPIO.output(config.GPIO_PIN_OPEN, GPIO.LOW)
        GPIO.output(config.GPIO_PIN_CLOSE, GPIO.HIGH)
        time.sleep(config.GPIO_CLOSE_PULSE_SECONDS)

        GPIO.output(config.GPIO_PIN_OPEN, GPIO.LOW)
        GPIO.output(config.GPIO_PIN_CLOSE, GPIO.LOW)
    finally:
        _door_busy.clear()


def open_door():
    """Grant access: run the open -> hold -> close relay sequence, non-blocking."""
    if config.PRINT_EVENTS_TO_CONSOLE:
        print("[DOOR] Access granted -> unlocking door.")

    if not config.GPIO_ENABLED:
        return

    if not _GPIO_AVAILABLE:
        total = config.GPIO_OPEN_PULSE_SECONDS + config.GPIO_HOLD_SECONDS + config.GPIO_CLOSE_PULSE_SECONDS
        print(f"[GPIO] (simulated - OPi.GPIO not available) "
              f"open pin {config.GPIO_PIN_OPEN} pulse {config.GPIO_OPEN_PULSE_SECONDS}s -> "
              f"hold {config.GPIO_HOLD_SECONDS}s -> "
              f"close pin {config.GPIO_PIN_CLOSE} pulse {config.GPIO_CLOSE_PULSE_SECONDS}s "
              f"(total {total}s)")
        return

    init_gpio()  # no-op if already initialized

    if _door_busy.is_set():
        print("[GPIO] Door cycle already in progress, ignoring duplicate trigger.")
        return

    _door_busy.set()
    threading.Thread(target=_run_door_cycle, daemon=True).start()


def send_warning(name, reason):
    """
    Fire a warning for a denied access attempt.

    name   -- "Unknown" or the recognized-but-unauthorized employee's name
    reason -- short human-readable reason, e.g. "unknown face" / "access revoked"
    """
    message = f"[ALERT] Access denied for '{name}': {reason}"
    if config.PRINT_EVENTS_TO_CONSOLE:
        print(message)

    if config.ENABLE_EMAIL_ALERTS:
        _send_email_alert(name, reason)


def _send_email_alert(name, reason):
    try:
        msg = MIMEText(f"Access denied for '{name}'.\nReason: {reason}")
        msg["Subject"] = f"[Door Access] Denied entry: {name}"
        msg["From"] = config.SMTP_USER
        msg["To"] = ", ".join(config.ALERT_RECIPIENTS)

        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.sendmail(config.SMTP_USER, config.ALERT_RECIPIENTS, msg.as_string())
    except Exception as e:
        print(f"[ALERT] Failed to send email alert: {e}")
