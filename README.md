# DroneTracking

Drone tracking project using the DJI Tello drone. Our goal here is to follow a person wearing a specific yellow shirt (but it doesn't really matter, as one can re train Yolo easily).

## Version 1 Features

- YOLOv11 model trained to detect a specific yellow shirt.
- Application functional with current specifications:
    - Image refresh rate set to 4 FPS. Higher rates are possible, but the drone may not reliably receive and act on commands, causing repetitive errors.
    - User should move slowly for optimal tracking.
    - In the current version, the drone moves a fixed distance per action (currently 20 cm); users should account for this.
    - Drone can move forward, backward, up, down, left, or right. Rotation is not yet implemented, so tracking efficiency is limited.
    - Forward/backward centering is prioritized over other directions. A future improvement could involve prioritizing the largest correction needed.

## Acknowledgements

Special thanks to the [DJITelloPy project](https://github.com/damiafuentes/DJITelloPy/tree/master) for implementing a Python control library.