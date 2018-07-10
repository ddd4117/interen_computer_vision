import cv2
import numpy as np
import math


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


def get_max_area(img, img_in, name):
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img_in


    # Find Maximum area in contours
    # max_index, max_area = sorted(enumerate([cv2.contourArea(x) for x in contours]), key=lambda x: x[1], reverse=True)[0]
    # max_contour = contours[max_index]
    img_out = img_in.copy()
    # cx, cy = getCenter(max_contour)s
    for idx, area in sorted(enumerate([cv2.contourArea(x) for x in contours]), key=lambda x: x[1], reverse=True):

        if area < 400:
            break
        _contour = contours[idx]
        msk = np.zeros(img_in.shape[:2], np.uint8)

        cv2.drawContours(msk, [_contour], 0, 255, -1)
        b, g, r, c = cv2.mean(img_in, msk)

        if r > b >= g and  r - b > 20:

            cv2.drawContours(img_out, [_contour], 0, (0, 255, 100), 2)
            print(cv2.mean(img_in, msk), area)
            return img_out
    return img_in

        # cv2.imshow("image", black_image)

    # print(cv2.mean(img_in,mask=msk), cv2.mean(img_in), cv2.mean(msk))

    # cv2.drawContours(img_out, [max_contour], 0, (0, 255, 100), 2)
    # print(cv2.mean(img_in, mask=mask), cv2.mean(img_out))
    # print(cv2.mean(img_out, mask=max_contour[:,0,:]))

    # tmp = max_contour * 4 // 3
    # _cx, _cy = getCenter(tmp)
    # for i in range(len(tmp)):
    #     tmp[i][0][0] = tmp[i][0][0] - (_cx - cx)
    #     tmp[i][0][1] = tmp[i][0][1] - (_cy - cy)
    # cv2.drawContours(img_out, [tmp], 0, (0, 255, 100), 2)


def img_trim(img, x, y, w, h):
    img_trim = img[y:y + h, x:x + w]
    return img_trim


cap = cv2.VideoCapture("./../resource/test1.avi")
while cap.isOpened():
    ret, img_in = cap.read()
    w, h, c = img_in.shape  # img_in is the input images
    _trim_upper = img_trim(img_in, 100, 0, h - 100, w // 2)
    _trim_left = img_trim(img_in, 0, w // 2, h // 5, w - (w // 3) - 100)
    _trim_right = img_trim(img_in, h * 4 // 5, w // 2, h - (4 * h // 5), w - (w // 3) - 100)
    # cv2.imshow("upper", _trim_upper)
    # cv2.imshow("left", _trim_left)
    # cv2.imshow("right", _trim_right)
    resize_coeff = 1
    # img = cv2.resize(img_in, (int(resize_coeff * h), int(resize_coeff * w)))

    # 흑백사진
    # cv2.waitKey(0)
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    gray_upper = cv2.cvtColor(_trim_upper, cv2.COLOR_BGR2GRAY)
    gray_left = cv2.cvtColor(_trim_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(_trim_right, cv2.COLOR_BGR2GRAY)

    # Blur
    # img = cv2.medianBlur(gray, 9)
    img = cv2.GaussianBlur(gray, (9, 9), 0)
    img_upper = cv2.GaussianBlur(_trim_upper, (9, 9), 0)
    img_left = cv2.GaussianBlur(_trim_left, (9, 9), 0)
    img_right = cv2.GaussianBlur(_trim_right, (9, 9), 0)

    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,2)
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,35,2)
    # 외곽선 추출
    img = cv2.Canny(img, 55, 190)
    img_upper = cv2.Canny(img_upper, 55, 190)
    img_left = cv2.Canny(img_left, 55, 190)
    img_right = cv2.Canny(img_right, 55, 190)

    # img_origin_out = get_max_area(img, img_in, "origin")
    img_upper_out = get_max_area(img_upper, _trim_upper, "upper")
    img_left_out = get_max_area(img_left, _trim_left, "left")
    img_right_out = get_max_area(img_right, _trim_right, "right")

    # cv2.imshow("output", img_origin_out)
    cv2.imshow("upper", img_upper_out)
    cv2.imshow("left", img_left_out)
    cv2.imshow("right", img_right_out)
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
