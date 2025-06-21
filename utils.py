import time
import cv2
import os
import logging
from ultralytics import YOLO
import torch

os.environ["YOLO_VERBOSE"] = "False"
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def videoRecorder(frame_read, keepRecording, tello, save_frames=False, save_folder="recording"):
    frame_count = 0
    if torch.cuda.is_available():
        model = YOLO("best.pt").cuda()
    else:
        model = YOLO("best.pt")
    last_direction = "-"
    if save_frames:
        os.makedirs(save_folder, exist_ok=True)
    else:
        logger.info("save_frames is False, frames will not be saved.")
    last_keepalive = time.time()
    logger.info("Drone video recorder started. Press q to stop.")
    while keepRecording.is_set():
        frame = frame_read.frame
        if frame is not None and frame.size != 0:
            if save_frames:
                frame_path = os.path.join(save_folder, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                logger.debug(f"Saved frame {frame_count} to {frame_path}")
                frame_count += 1
            coord, display_frame = objectDetection(frame, model)
            direction = getDirection(coord, frame.shape[1], frame.shape[0], last_direction)
            if len(direction) > 1:
                try:
                    tello.move(direction, 20)
                except Exception as e:
                    logger.error(f"Error during tello.move('{direction}', 20): {e}")
                    time.sleep(1)
            last_direction = direction if coord is not None else "-"
            cv2.imshow("Tello Stream", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            tello.land()
            keepRecording.clear()
            cv2.destroyAllWindows()
            logger.info("Video recording stopped by user.")
            break
        # Send keep-alive every 5 seconds
        if tello.is_flying and (time.time() - last_keepalive > 5):
            tello.send_rc_control(0, 0, 0, 0)
            logger.debug("Sent keep-alive rc_control to drone.")
            last_keepalive = time.time()
        time.sleep(1/4)

def objectDetection(frame, model=None):
    coord = None
    frame_with_box = frame.copy()
    results = model(frame)
    boxes = results[0].boxes  # get boxes from the first result

    # Find the first box of class 0
    for i in range(len(boxes.cls)):
        if int(boxes.cls[i]) == 0:
            coord = boxes.xywh[i].cpu().numpy()  # (x_center, y_center, w, h)
            x, y, w, h = coord
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
            break  # Only the first shirt detected
    logger.debug(f"Yellow shirt detected: {coord}")
    return coord, frame_with_box

def getDirection(coord, frame_width, frame_height, last_direction="-"):
    if coord is None:
        logger.info(f"No detection, returning last direction: {last_direction}")
        return last_direction

    box_x, box_y, box_w, box_h = coord
    center_x, center_y = frame_width / 2, frame_height / 2
    offset_x = box_x - center_x
    offset_y = box_y - center_y
    center_threshold_x = frame_width * 0.1
    center_threshold_y = frame_height * 0.1
    
    # If centered, check box size for forward/backward
    box_area = box_w * box_h
    frame_area = frame_width * frame_height
    ideal_area = frame_area / 20
    area_threshold = ideal_area * 0.5

    if box_area < ideal_area - area_threshold:
        logger.info("Box too small, moving forward.")
        return "forward"
    elif box_area > ideal_area + area_threshold:
        logger.info("Box too large, moving backward.")
        return "back"

    # Center horizontally
    if abs(offset_x) > center_threshold_x:
        direction = "right" if offset_x > 0 else "left"
        logger.info(f"Box not centered horizontally, direction: {direction}")
        return direction
    # Center vertically
    if abs(offset_y) > center_threshold_y:
        direction = "down" if offset_y > 0 else "up"
        logger.info(f"Box not centered vertically, direction: {direction}")
        return direction

    logger.info("Box centered and at ideal size, staying in place.")
    return "-"