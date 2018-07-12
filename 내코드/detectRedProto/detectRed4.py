import datetime

import cv2
import numpy as np


def findColor(frame,ret):
    # grab the frame

    y = 0.299*80 + 0.587*45 + 0.114 *40
    u= (40-y) * 0.565
    v= (80-y) * 0.713
    Y = 0.299*225 + 0.587*85 + 0.114 *120
    U= (120-Y) * 0.565
    V= (225-Y) * 0.713

    ori=frame.copy()

    start = datetime.datetime.now()

    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    Ymask = cv2.inRange(yuv, np.array([0,130,130]), np.array([80,225,225])) # yellow + red : np.array([20 or 30, 225, 255])  # only red : np.array([10, 225, 255])


    frame = cv2.bitwise_and(frame, frame, mask=Ymask)

    cv2.imshow("Original", ori)
    cv2.imshow('red', frame)

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