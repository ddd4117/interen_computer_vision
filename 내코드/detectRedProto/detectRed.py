import datetime

import cv2
import numpy as np


def findColor(frame,ret):
    # grab the frame
    ilowH = 130
    ihighH = 200

    ilowS = 50
    ihighS = 225
    ilowV = 80
    ihighV = 225
    ori=frame.copy()

    start = datetime.datetime.now()

    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    Rmask1 = cv2.inRange(hsv1, np.array([0, 80, 80]), np.array([20, 225, 255]))
    Rmask2 = cv2.inRange(hsv2, np.array([ilowH, ilowS, ilowV]), np.array([ihighH, ihighS, ihighV]))

    mask = Rmask1 | Rmask2
    hsv=hsv1 | hsv2

    frame = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original", ori)
    cv2.imshow('red', frame)

    print("[INFO] detection took: {}s".format(
        (datetime.datetime.now() - start).total_seconds()))

cap = cv2.VideoCapture('driving2.mp4')



while(True):
    ret, frame = cap.read()
    findColor(frame,ret)
    k = cv2.waitKey(30) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break

cv2.destroyAllWindows()
cap.release()