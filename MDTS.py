import cv2
import numpy as np
import time
from datetime import datetime

camera = cv2.VideoCapture(0)
prev_time = time.time()
success, previous_frame = camera.read()

while True:

    success, current_frame = camera.read()

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_frame,
        cv2.COLOR_BGR2GRAY
    )

    previous_gray = cv2.GaussianBlur(
    previous_gray,
    (21,21),
    0
    )

    current_gray = cv2.GaussianBlur(
        current_gray,
        (21,21),
        0
    )
    difference = cv2.absdiff(
        previous_gray,
        current_gray
    )
    _, threshold = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((5,5), np.uint8)

    threshold = cv2.dilate(
        threshold,
        kernel,
        iterations = 2
    )
    
    contours, hierarchy = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    motion_status = "No Motion"

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000:
            continue
        motion_status = "Motion Detected"

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            current_frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
    )
        cv2.putText(
            current_frame,
            motion_status,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    current_time = time.time()
    fps = 1/(current_time - prev_time)

    prev_time = current_time
    fps = int(fps)

    timestamp = datetime.now().strftime(
    "%d %b %Y %I:%M:%S %p"
    )

    cv2.putText(
        current_frame,
        f"FPS: {fps}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,255),
        2
    )
    cv2.putText(
        current_frame,
        timestamp,
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )
        
    cv2.imshow("Motion Mask", threshold)
    cv2.imshow("Live Feed", current_frame)
    
    previous_frame = current_frame

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()