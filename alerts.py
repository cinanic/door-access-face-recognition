"""
Warning / alert delivery and door actuation.

Both open_door() and send_warning() are intentionally simple so you can wire
them to your real hardware / notification stack without touching recognize.py.
"""

import smtplib
from email.mime.text import MIMEText

import config


def open_door():
    """
    Trigger whatever mechanism unlocks your door.

    Replace the print() below with your real integration, e.g.:

    Raspberry Pi GPIO relay:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(17, GPIO.OUT)
        GPIO.output(17, GPIO.HIGH)   # unlock
        time.sleep(config.DOOR_OPEN_SECONDS)
        GPIO.output(17, GPIO.LOW)    # re-lock

    USB relay / Arduino over serial:
        import serial
        ser = serial.Serial('/dev/ttyUSB0', 9600)
        ser.write(b'OPEN\\n')
    """
    print("[DOOR] Access granted -> unlocking door.")


def send_warning(name, reason):
    """
    Fire a warning for a denied access attempt.

    name   -- "Unknown" or the recognized-but-unauthorized employee's name
    reason -- short human-readable reason, e.g. "unknown face" / "access revoked"
    """
    message = f"[ALERT] Access denied for '{name}': {reason}"
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
