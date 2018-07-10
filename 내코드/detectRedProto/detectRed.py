import cv2
import numpy as np

def callback(x):
    pass

cap = cv2.VideoCapture('driving2.mp4')
cv2.namedWindow('image')

ilowH = 150
ihighH = 180

ilowS = 80
ihighS = 225
ilowV = 80
ihighV = 225

# create trackbars for color change
cv2.createTrackbar('lowH','image',ilowH,360,callback)
cv2.createTrackbar('highH','image',ihighH,360,callback)

cv2.createTrackbar('lowS','image',ilowS,255,callback)
cv2.createTrackbar('highS','image',ihighS,255,callback)

cv2.createTrackbar('lowV','image',ilowV,255,callback)
cv2.createTrackbar('highV','image',ihighV,255,callback)

cnt=1

while(True):
    # grab the frame
    ret, frame = cap.read()
    ori=frame.copy()

    # get trackbar positions
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')

    #################################################################################################33

    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    Rmask1 = cv2.inRange(hsv1, np.array([0, 80, 80]), np.array([10, 225, 255]))
    Rmask2 = cv2.inRange(hsv2, np.array([ilowH, ilowS, ilowV]), np.array([ihighH, ihighS, ihighV]))

    mask = Rmask1 | Rmask2
    hsv=hsv1 | hsv2

    frame = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('red', frame)

    #################################################################################################33

    thres = hsv.copy()
    kernel = np.ones((5, 5), np.uint8)
    thres = cv2.erode(thres, kernel, iterations=cnt)
    thres = cv2.dilate(thres, kernel, iterations=cnt)

    thres = cv2.dilate(thres, kernel, iterations=cnt)
    thres = cv2.erode(thres, kernel, iterations=cnt)

    canny = thres.copy()
    canny = cv2.Canny(canny, 100, 200)

    gray = canny.copy()
    gray, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    cv2.imshow("Original", ori)
    #    cv2.imshow("Thresholded Image", thres)  # show the thresholded image
#    cv2.imshow("Canny", canny)
#    cv2.imshow("Result window", gray)
    # show thresholded image

    k = cv2.waitKey(1000) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break

cv2.destroyAllWindows()
cap.release()