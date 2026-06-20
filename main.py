import cv2
import numpy as np

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = (100, 150 ,50)
    upper_blue = (140, 255 ,255)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((9,9), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        biggest_contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(biggest_contour)

        if area > 5000:
            
            x, y, w, h = cv2.boundingRect(biggest_contour)


            center_x = x + w//2
            center_y = y + h//2

# To green boundary around the blue image
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        
        )
# To add a red circle in center
        cv2.circle (
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"X={center_x} Y={center_y}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Area={int(area)}",
            (x, y+h+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()