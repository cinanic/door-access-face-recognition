import os
import enroll
import recognize
import config


def main():

    # Always refresh employee database before starting camera
    enroll.main()

    # Start live recognition
    recognize.main()


if __name__ == "__main__":
    main()
