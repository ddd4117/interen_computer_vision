import datetime

import cv2
import numpy as np


def findColor2(frame,ret):
    # grab the frame
    ilowH = 150 # yellow + red : 130 # only red : 150
    ihighH = 180 # yellow + red : 200 # only red : 180

    ilowS = 50 # dark red + white : 30 or 50 # without white : 80
    ihighS = 225
    ilowV = 80
    ihighV = 225
    ori=frame.copy()

    start = datetime.datetime.now()

    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
 #   hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

    Ymask = cv2.inRange(ycc, np.array([60, 130, 130]), np.array([100, 225, 160])) # yellow + red : np.array([20 or 30, 225, 255])  # only red : np.array([10, 225, 255])

    yresult = cv2.bitwise_and(frame, frame, mask=Ymask)

    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    Rmask1 = cv2.inRange(hsv1, np.array([0, 80, 80]), np.array([10, 225, 255]))  # yellow + red : np.array([20 or 30, 225, 255])  # only red : np.array([10, 225, 255])
    Rmask2 = cv2.inRange(hsv2, np.array([ilowH, ilowS, ilowV]), np.array([ihighH, ihighS, ihighV]))

    mask = Rmask1 | Rmask2
    hsv = hsv1 | hsv2

    cv2.imshow('YCrCb', yresult)
    Hresult = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('HSV', Hresult)

    frame =yresult

    cv2.imshow("Original", ori)
    cv2.imshow('red', frame)

    print("[INFO] detection took: {}s".format(
        (datetime.datetime.now() - start).total_seconds()))



cap = cv2.VideoCapture('driving.mp4')

while(True):
    ret, frame = cap.read()
    findColor2(frame,ret)
    k = cv2.waitKey(30) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break

cv2.destroyAllWindows()
cap.release()