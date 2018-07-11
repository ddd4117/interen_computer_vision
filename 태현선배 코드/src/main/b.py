import cv2
import numpy as np
import datetime

###################
# Global Variable #
ilowH = 150
ihighH = 180

ilowS = 80
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


def get_max_area(img_in, contours, name):
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
        cv2.drawContours(img_out, [_contour], 0, (0, 255, 100), 2)
    return img_out

def img_trim(img, x, y, w, h):
    img_trim = img[y:y + h, x:x + w]
    return img_trim


def get_red_color(frame, name):
    ori = frame.copy()

    start = datetime.datetime.now()
    #################################################################################################33

    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    Rmask1 = cv2.inRange(hsv1, np.array([0, 80, 80]), np.array([10, 225, 255]))
    Rmask2 = cv2.inRange(hsv2, np.array([ilowH, ilowS, ilowV]), np.array([ihighH, ihighS, ihighV]))

    mask = Rmask1 | Rmask2
    hsv = hsv1 | hsv2

    frame = cv2.bitwise_and(frame, frame, mask=mask)
    frame = cv2.cv2.GaussianBlur(frame, (7, 7), 0)
    cv2.imshow(name, frame)
    # print("[INFO] detection took: {}s".format(
    #     (datetime.datetime.now() - start).total_seconds()))

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

    canny = frame.copy()
    canny = cv2.Canny(canny, 55, 195)

    gray = canny.copy()
    gray, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # gray = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    return contours


if __name__ == "__main__":
    cap = cv2.VideoCapture("./../resource/test2.mp4")
    while cap.isOpened():
        ret, img_in = cap.read()
        w, h, c = img_in.shape  # img_in is the input images
        _trim_upper = img_trim(img_in, 100, 0, h - 100, w // 2)
        _trim_left = img_trim(img_in, 0, w // 2, h // 5, w - (w // 3) - 100)
        _trim_right = img_trim(img_in, h * 4 // 5, w // 2, h - (4 * h // 5), w - (w // 3) - 100)

        upper_contours = get_red_color(_trim_upper, "ur")
        left_contours = get_red_color(_trim_left,"lr")
        right_contours = get_red_color(_trim_right, "rr")

        upper = get_max_area(_trim_upper, upper_contours, "upper")
        left = get_max_area(_trim_left, left_contours, "left")
        right = get_max_area(_trim_right, right_contours, "right")
        # cv2.imshow("output", img_origin_out)
        cv2.imshow("upper", upper)
        cv2.imshow("left", left)
        cv2.imshow("right", right)
        # cv2.imshow("input", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# img_in = cv2.imread("./../resource/test3.jpg")
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
# w, h, c = img_in.shape #img_in is the input image

# resize_coeff = 1
# img = cv2.resize(img_in, (int(resize_coeff*h), int(resize_coeff*w)))
# # 흑백사진
#
# gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
#
# #Blur
# img = cv2.medianBlur(gray, 9)
# # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,2)
# # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,35,2)
# # 외곽선 추출
# img = cv2.Canny(img, 55, 190)
#
# img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #Find Maximum area in contours
# max_index, max_area = max(enumerate([cv2.contourArea(x) for x in contours]), key = lambda x: x[1])
# max_contour = contours[max_index]
#
# img_out = cv2.resize(img_in, (int(resize_coeff*h), int(resize_coeff*w)))
# cx, cy = getCenter(max_contour)
# cv2.drawContours(img_out, [max_contour], 0, (0, 255, 100), 2)
#
#
# tmp = max_contour * 4 // 3
# _cx, _cy= getCenter(tmp)
# for i in range(len(tmp)):
#     tmp[i][0][0] = tmp[i][0][0] - (_cx - cx)
#     tmp[i][0][1] = tmp[i][0][1] - (_cy - cy)
# cv2.drawContours(img_out, [tmp], 0, (0, 255, 100), 2)
#
# cv2.imshow("output", img_out)
# cv2.imshow("input", img)
# cv2.waitKey(0)
