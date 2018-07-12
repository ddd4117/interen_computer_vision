import datetime

import cv2
import numpy as np


def findColor(frame,ret):
    # grab the frame
    ilowH = 150 # yellow + red : 130 # only red : 150
    ihighH = 180 # yellow + red : 200 # only red : 180

    ilowS = 80 # dark red + white : 30 or 50 # without white : 80
    ihighS = 225
    ilowV = 80
    ihighV = 225

    ori=frame.copy()

    start = datetime.datetime.now()

    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    Rmask = cv2.inRange(frame, np.array([40, 45, 80]), np.array([120, 85, 255])) # yellow + red : np.array([20 or 30, 225, 255])  # only red : np.array([10, 225, 255])
    Ymask = cv2.inRange(ycc, np.array([60, 130, 130]), np.array(
        [100, 225, 160]))  # yellow + red : np.array([20 or 30, 225, 255])  # only red : np.array([10, 225, 255])
    Hmask1 = cv2.inRange(hsv1, np.array([0, 80, 80]), np.array(
        [10, 225, 255]))  # yellow + red : np.array([20 or 30, 225, 255])  # only red : np.array([10, 225, 255])
    Hmask2 = cv2.inRange(hsv2, np.array([ilowH, ilowS, ilowV]), np.array([ihighH, ihighS, ihighV]))

    Hmask = Hmask1 | Hmask2

    rresult = cv2.bitwise_and(frame, frame, mask=Rmask)
    yresult = cv2.bitwise_and(frame, frame, mask=Ymask)
    Hresult = cv2.bitwise_and(frame, frame, mask=Hmask)

    #   hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

    cv2.imshow('HR', rresult)

    frame = yresult | rresult | Hresult

    cv2.imshow("Original", ori)

    print("[INFO] detection took: {}s".format(
        (datetime.datetime.now() - start).total_seconds()))



cap = cv2.VideoCapture('driving2.mp4')

while(True):
    ret, frame = cap.read()
    findColor(frame,ret)
    k = cv2.waitKey(10) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break

cv2.destroyAllWindows()
cap.release()