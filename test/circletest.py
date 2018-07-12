import cv2
import numpy as np
import datetime

###################
# Global Variable #
ilowH = 130
ihighH = 200

ilowS = 50
ihighS = 225
ilowV = 80
ihighV = 225

###################
def getCenter(_max_contour):
    cx = 0
    cy = 0
    A = 0
    for i in range(len(_max_contour) - 1):
        xi = _max_contour[i][0][0]
        yi = _max_contour[i][0][1]
        xii = _max_contour[i + 1][0][0]
        yii = _max_contour[i + 1][0][1]
        tmp = (xi * yii) - (xii * yi)
        cx += (xi + xii) * tmp
        cy += (yi + yii) * tmp

        A += (xi * yii) - (xii * yi)
    A /= 2
    cx /= 6 * A
    cy /= 6 * A
    # cv2.circle(img_out, (int(cx), int(cy)), 2, (0, 255, 255), -1)
    return cx, cy

def houghCircle(img1):
    img2=img1.copy()

    img2=cv2.GaussianBlur(img2, (3, 3), 0)
    imgray=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    circles=cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 15, param1=60, param2=33, minRadius=0, maxRadius=35)

    if circles is not None:
        circles=np.uint16(np.around(circles))

        for i in circles[0, :]:
            cv2.circle(img1, (i[0], i[1]), i[2], (255, 255, 0), 2)
            print('Circle')
            cv2.imshow('HoughCircles', img1)

def img_trim(img, x, y, w, h):
    img_trim = img[y:y + h, x:x + w]
    return img_trim

def get_max_area(img_in, frame, name):

    canny = frame.copy()
    canny = cv2.Canny(canny, 55, 195)

    gray = canny.copy()
    gray, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img_in

    # Find Maximum area in contours
    # max_index, max_area = sorted(enumerate([cv2.contourArea(x) for x in contours]), key=lambda x: x[1], reverse=True)[0]
    # max_contour = contours[max_index]
    img_out = img_in.copy()
    # cx, cy = getCenter(max_contour)s
    for idx, area in sorted(enumerate([cv2.contourArea(x) for x in contours]), key=lambda x: x[1], reverse=True):
        if area < 100:
            break
        _contour = contours[idx]
        l1 = []
        c = _contour.copy()
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        if extLeft[0]<10 or extTop[1]<10:
            #extLeft=extLeft[0:0]+(10,0)+extLeft[1:]
            #extTop=extTop[0:1]+(10, 0)+extTop[2:]
            cropped_img = img_trim(img_in, 0, 0, extRight[0] - extLeft[0] + 20,
                                   extBot[1] - extTop[1] + 20)
        else:
            cropped_img = img_trim(img_in, extLeft[0] - 10, extTop[1] - 10, extRight[0] - extLeft[0] + 20,
                                   extBot[1] - extTop[1] + 20)
        #cv2.drawContours(img_out, [_contour], 0, (0, 255, 100), 2)
        cv2.imshow("cropped", cropped_img)

        l1.append(cropped_img)
        houghCircle(cropped_img)
        #detectTriangle(cropped_img)

        cv2.waitKey(0)

        img2 = cv2.GaussianBlur(cropped_img, (3, 3), 0)
        g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(g, 127, 255, 1)

        image, con, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in con:
            approx = cv2.approxPolyDP(cnt, 0.025 * cv2.arcLength(cnt, True), True)
            #print(len(approx))
            if len(approx) == 3:
                print("triangle")
                #cv2.drawContours(cropped_img, [cnt], 0, 255, 2)

                cv2.imshow("triangle", cropped_img)

        cv2.drawContours(cropped_img, [_contour], 0, (0, 255, 100), 2)
        #cv2.imshow("cropped", cropped_img)

    return img_out

def get_red_color(frame, name):
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
    cv2.imshow(name, frame)
    cv2.waitKey(0)
    #print("[INFO] detection took: {}s".format(
    #   (datetime.datetime.now() - start).total_seconds()))
    return frame

#
# def get_red_color(frame, name):
#     ori = frame.copy()
#
#     start = datetime.datetime.now()
#     #################################################################################################
#
#     hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     Rmask1 = cv2.inRange(hsv1, np.array([0, 80, 80]), np.array([10, 225, 255]))
#     Rmask2 = cv2.inRange(hsv2, np.array([ilowH, ilowS, ilowV]), np.array([ihighH, ihighS, ihighV]))
#
#     mask = Rmask1 | Rmask2
#     hsv = hsv1 | hsv2
#
#     frame = cv2.bitwise_and(frame, frame, mask=mask)
#     frame = cv2.cv2.GaussianBlur(frame, (7, 7), 0)
#     cv2.imshow(name, frame)
#     # print("[INFO] detection took: {}s".format(
#     #     (datetime.datetime.now() - start).total_seconds()))
#
#     """
#      ## this part erode and dilate picture
#      ## then cannot find small flag...
#     thres = frame.copy()
#     kernel = np.ones((5, 5), np.uint8)
#     thres = cv2.erode(thres, kernel, iterations=cnt)
#     thres = cv2.dilate(thres, kernel, iterations=cnt)
#     thres = cv2.dilate(thres, kernel, iterations=cnt)
#     thres = cv2.erode(thres, kernel, iterations=cnt)
#     """
#
#     canny = frame.copy()
#     canny = cv2.Canny(canny, 55, 195)
#
#     gray = canny.copy()
#     gray, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     # gray = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
#     return contours

if __name__ == "__main__":
    cap = cv2.VideoCapture("driving_triangle.mp4")
    while cap.isOpened():
        ret, img_in = cap.read()
        w, h, c = img_in.shape  # img_in is the input images
        _trim_upper = img_trim(img_in, 100, 0, h - 100, w // 2)
        _trim_left = img_trim(img_in, 0, w // 2, h // 5, w - (w // 3) - 100)
        _trim_right = img_trim(img_in, h * 4 // 5, w // 2, h - (4 * h // 5), w - (w // 3) - 100)

        upper_contours = get_red_color(_trim_upper, "ur")
        left_contours = get_red_color(_trim_left, "lr")
        right_contours = get_red_color(_trim_right, "rr")
        upper = get_max_area(_trim_upper, upper_contours, "upper")
        left = get_max_area(_trim_left, left_contours, "left")
        right = get_max_area(_trim_right, right_contours, "right")
        # cv2.imshow("output", img_origin_out)
        cv2.imshow("upper", upper)
        cv2.imshow("left", left)
        cv2.imshow("right", right)
        # cv2.imshow("input", img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()