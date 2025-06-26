import logging
from djitellopy import Tello
import time
from threading import Thread, Event
from utils import videoRecorder

# Prevent double logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("djitellopy").propagate = False
logging.getLogger("djitellopy").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def run():
    tello = Tello()
    tello.connect()
    logger.info("Drone battery: %s%%", tello.get_battery())
    tello.streamon()
    frame_read = tello.get_frame_read()
    keepRecording = Event()
    keepRecording.set()

    # Wait for the first valid frame
    while frame_read.frame is None or frame_read.frame.size == 0:
        time.sleep(0.1)

    recorder = Thread(target=videoRecorder, kwargs={
        "frame_read": frame_read,
        "keepRecording": keepRecording,
        "tello": tello,
        "save_frames": False,
        "save_folder": "recording"
    })
    recorder.start()

    try:
        tello.takeoff()
    except Exception as e:
        logger.warning(f"Could not take off: {e}")

    recorder.join()
    logger.info("Drone battery: %s%%", tello.get_battery())
    tello.end()
    logger.info("Drone landed, connection ended.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting drone control script.")
    run()