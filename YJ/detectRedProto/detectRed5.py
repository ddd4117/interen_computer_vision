import datetime

import cv2
import numpy as np


def findColor(frame,ret):
    ori=frame.copy()

    start = datetime.datetime.now()

    B=frame[:,:,0]
    G=frame[:,:,1]
    R=frame[:,:,2]

#    Rmask = cv2.inRange(frame, np.array([40, 45, 80]), np.array([120, 85, 255])) # yellow + red : np.array([20 or 30, 225, 255])  # only red : np.array([10, 225, 255])

 #   rresult = cv2.bitwise_and(frame, frame, mask=Rmask)

    #   hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

    rresult=2*R-G-B
    _,thres = cv2.threshold(rresult, 0, 225, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    mask = cv2.inRange(frame, np.array([40, 45, 80]), np.array([120, 85, 255])) # yellow + red : np.array([20 or 30, 225, 255])  # only red : np.array([10, 225, 255])
    msk = cv2.bitwise_and(thres,thres, mask=mask)

    cv2.imshow("Original", ori)
    cv2.imshow('2rgb', rresult)
    cv2.imshow('thres', thres)

    print("[INFO] detection took: {}s".format(
        (datetime.datetime.now() - start).total_seconds()))



cap = cv2.VideoCapture('driving.mp4')

while(True):
    ret, frame = cap.read()
    findColor(frame,ret)
    k = cv2.waitKey(30) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break

cv2.destroyAllWindows()
cap.release()