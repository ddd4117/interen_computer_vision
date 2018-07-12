import datetime

import cv2
import numpy as np


def findColor(frame, ret):
    # grab the frame
    ilowH = 150 # yellow + red : 130 # only red : 150
    ihighH = 180 # yellow + red : 200 # only red : 180
    ilowS = 80 # dark red + white : 30 or 50 # without white : 80
    ihighS = 225
    ilowV = 80
    ihighV = 225

    y = 0.299 * 80 + 0.587 * 45 + 0.114 * 40
    u = (40 - y) * 0.565
    v = (80 - y) * 0.713
    Y = 0.299 * 225 + 0.587 * 85 + 0.114 * 120
    U = (120 - Y) * 0.565
    V = (225 - Y) * 0.713

    ori=frame.copy()

    start = datetime.datetime.now()

    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)


    Hmask1 = cv2.inRange(hsv1, np.array([0, 80, 80]), np.array(
        [10, 225, 255]))  # yellow + red : np.array([20 or 30, 225, 255])  # only red : np.array([10, 225, 255])
    Hmask2 = cv2.inRange(hsv2, np.array([ilowH, ilowS, ilowV]), np.array([ihighH, ihighS, ihighV]))

    Hmask = Hmask1 | Hmask2
    Ymask = cv2.inRange(yuv, np.array([0,130,130]), np.array([80,225,225])) # yellow + red : np.array([20 or 30, 225, 255])  # only red : np.array([10, 225, 255])


    Yresult = cv2.bitwise_and(frame, frame, mask=Ymask)
    Hresult = cv2.bitwise_and(frame, frame, mask=Hmask)

    #   hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)


    frame = Hresult|Yresult
    cv2.imshow("Red", frame)

    print("[INFO] detection took: {}s".format(
        (datetime.datetime.now() - start).total_seconds()))
    return frame

cap = cv2.VideoCapture('driving_triangle.mp4')

while(True):
    ret, frame = cap.read()
    findColor(frame, ret)
    k = cv2.waitKey(10) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break

cv2.destroyAllWindows()
cap.release()