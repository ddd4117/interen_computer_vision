import datetime

import cv2
import numpy as np

def callback(x):
    pass

cap = cv2.VideoCapture('driving.mp4')

ilowH = 130
ihighH = 200

ilowS = 30
ihighS = 225
ilowV = 80
ihighV = 225

# create trackbars for color change

cnt=1

while(True):
    # grab the frame
    ret, frame = cap.read()
    ori=frame.copy()

    start = datetime.datetime.now()
    #################################################################################################33

    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    Rmask1 = cv2.inRange(hsv1, np.array([0, 80, 80]), np.array([20, 225, 255]))
    Rmask2 = cv2.inRange(hsv2, np.array([ilowH, ilowS, ilowV]), np.array([ihighH, ihighS, ihighV]))

    mask = Rmask1 | Rmask2
    hsv=hsv1 | hsv2

    frame = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('red', frame)

    print("[INFO] detection took: {}s".format(
        (datetime.datetime.now() - start).total_seconds()))

    """
     ## this part erode and dilate picture
     ## then cannot find small flag... 
    thres = frame.copy()
    kernel = np.ones((5, 5), np.uint8) 
    thres = cv2.erode(thres, kernel, iterations=cnt)
    thres = cv2.dilate(thres, kernel, iterations=cnt)

    thres = cv2.dilate(thres, kernel, iterations=cnt)
    thres = cv2.erode(thres, kernel, iterations=cnt)
    """

    #################################################################################################33

    canny = frame.copy()
    canny = cv2.Canny(canny, 100, 200)

    gray = canny.copy()
    gray, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)


    cv2.imshow("Original", ori)
    #    cv2.imshow("Thresholded Image", thres)  # show the thresholded image
    cv2.imshow("Canny", canny)
#    cv2.imshow("Result window", gray)
    # show thresholded image

    k = cv2.waitKey(10) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break

cv2.destroyAllWindows()
cap.release()