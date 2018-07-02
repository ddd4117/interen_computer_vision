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


img_in = cv2.imread("./../resource/test4.jpg")
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
w, h, c = img_in.shape #img_in is the input image
cv2.namedWindow("output");
cv2.moveWindow("output", 400,300);
cv2.namedWindow("input");
cv2.moveWindow("input", 400 + h + 50,300);
resize_coeff = 1
img = cv2.resize(img_in, (int(resize_coeff*h), int(resize_coeff*w)))

gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY) # 흑백사진
img = cv2.medianBlur(gray, 5) #Blur
img = cv2.Canny(img, 55, 190) #외곽선 추출

img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)
max_index, max_area = max(enumerate([cv2.contourArea(x) for x in contours]), key = lambda x: x[1])
# print(max_area, max_index)
max_contour = contours[max_index]

img_out = cv2.resize(img_in, (int(resize_coeff*h), int(resize_coeff*w)))
# print([max_contour][0])
cx, cy= getCenter(max_contour)
cv2.drawContours(img_out, [max_contour], 0, (0, 255, 100), 2)


tmp = max_contour * 4 // 3
_cx, _cy= getCenter(tmp)
for i in range(len(tmp)):
    tmp[i][0][0] = tmp[i][0][0] - (_cx - cx)
    tmp[i][0][1] = tmp[i][0][1] - (_cy - cy)
cv2.drawContours(img_out, [tmp], 0, (0, 255, 100), 2)

cv2.imshow("output", img_out)
cv2.imshow("input", img)
cv2.waitKey(0)

