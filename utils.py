import sys
import time
import cv2
import os
import logging
from ultralytics import YOLO
import torch
import numpy as np

sys.path.append(os.path.abspath("Depth-Anything-V2"))
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

FPS=10
TARGET_DEPTH=2.5 # Distance idéal entre l'objet et le drone mètre
MAX_SPEED=20
MAX_SPEED_DEPTH=5 # A partir de 5 mètre, le drone se déplace à sa vitesse max sur l'axe z

os.environ["YOLO_VERBOSE"] = "False"
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def videoRecorder(frame_read, keepRecording, tello, save_frames=False, save_folder="recording"):
    frame_count = 0
    device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
    )

    model_obj = YOLO("best.pt").to(device)
    # model_obj.eval()
    model_depth = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384]).to(device)
    model_depth.load_state_dict(torch.load('depth_anything_v2_metric_hypersim_vits.pth', map_location='cpu'))
    model_depth.eval()

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
            coord, display_frame = objectDetection(frame, model_obj)

            if coord is None:
                # tourne à droite avec une vitesse de 10
                tello.send_rc_control(0, 0, 0, 10)
                logger.info(f"No shirt, rotating")
            else:
                depth = depthEstimation(frame, model_depth, device)
                med_depth = objectDepth(coord, depth)
                lf_speed, fb_speed, ud_speed, y_speed = computeSpeed(frame, med_depth, coord)
                tello.send_rc_control(lf_speed, fb_speed, ud_speed, y_speed)
                logger.info(f"Centering shirt in the frame. Speeds : {lf_speed},{fb_speed},{ud_speed},{y_speed}")
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
        time.sleep(1/FPS)

def objectDetection(frame, model_obj=None):
    coord = None
    frame_with_box = frame.copy()
    results = model_obj(frame)
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

def depthEstimation(frame, model_depth=None, device=None):
    # Charger l'image
    raw_frame = frame

    # Passage en Tensor
    frame_tensor = model_depth.image2tensor(raw_frame, input_size=518)[0]
    frame_tensor = frame_tensor.to(device)

    # Inférence
    with torch.no_grad():
        depth = model_depth.forward(frame_tensor)
    
    depth = depth.squeeze().cpu().numpy()
    # Redimensionner à la taille d'origine
    depth = cv2.resize(depth, (raw_frame.shape[1], raw_frame.shape[0]))

    return depth

def objectDepth(coord, depth):
    x, y, w, h = coord
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    mask = np.zeros(depth.shape, dtype=bool)

    mask[y1:y2, x1:x2] = True

    depth_masked = depth[mask]

    # Filtrer les profondeurs valides (par exemple, > 0)
    depth_valid = depth_masked[depth_masked > 0]

    return np.median(depth_valid)

def computeSpeed(frame, med_depth, coord):
    frame_height, frame_width,_ = frame.shape
    x, y, w, h = coord
    print(med_depth)
    # vitesse pour la profondeur
    fb_speed = (med_depth - TARGET_DEPTH) * (MAX_SPEED/TARGET_DEPTH)

    if fb_speed < 0:
        fb_speed = max(fb_speed, -MAX_SPEED)
    else:
        fb_speed = min(fb_speed, MAX_SPEED)

    # vitesse de gauche à droite
    lf_speed = ((x - frame_width/2) / (frame_width/2)) * MAX_SPEED

    # vitesse de haut en bas
    ud_speed = ((y - frame_height/2) / (frame_height/2)) * MAX_SPEED


    return int(lf_speed), int(fb_speed), int(ud_speed), 0 # 0 pour la yaw speed