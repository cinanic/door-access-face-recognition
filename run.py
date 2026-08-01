"""
Single-command entry point: enrolls employees, then immediately starts live
recognition. Use this instead of running enroll.py and recognize.py
separately -- handy for a systemd service or a one-shot startup script on
the Raspberry Pi.

Usage:
    python run.py
Stop with Ctrl+C (recognize.py handles that cleanly, including GPIO cleanup).
"""

import os
import sys

import config
import enroll
import recognize


def main():
    print("=== Step 1/2: Enrolling employees ===")
    enroll.main()

    if not (os.path.exists(config.ENCODINGS_FILE) and os.path.exists(config.EMPLOYEES_JSON)):
        print(
            "\nEnrollment did not produce a database -- no employees were "
            f"found under {config.EMPLOYEES_DIR}. Add photos/videos there "
            "(one folder per person) and run again."
        )
        sys.exit(1)

    print("\n=== Step 2/2: Starting live recognition ===")
    recognize.main()


if __name__ == "__main__":
    main()
