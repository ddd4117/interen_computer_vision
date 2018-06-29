import cv2

img_in = cv2.imread("./../resource/test5.jpg")
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
w, h, c = img_in.shape #img_in is the input image
resize_coeff = 1
img = cv2.resize(img_in, (int(resize_coeff*h), int(resize_coeff*w)))
gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY) # 흑백사진
img = cv2.medianBlur(gray, 5) #Blur
img = cv2.Canny(img, 55, 190) #외곽선 추출
img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
max_index, max_area = max(enumerate([cv2.contourArea(x) for x in contours]), key = lambda x: x[1])
max_contour = contours[max_index]

img_out = cv2.resize(img_in, (int(resize_coeff*h), int(resize_coeff*w)))
print(max_contour)
cx = 0
cy = 0
A = 0
for i in range(len(max_contour) - 1):

    xi = max_contour[i][0][0]
    yi = max_contour[i][0][1]
    xii = max_contour[i+1][0][0]
    yii = max_contour[i+1][0][1]
    # cv2.circle(img_out, (int(xi), int(yi)), 1, (0, 0, 255), -1)
    tmp = (xi*yii)-(xii*yi)
    cx += (xi + xii) * tmp
    cy += (yi + yii) * tmp
    A += (xi * yii) - (xii * yi)
A /= 2
cx /= 6*A
cy /= 6*A
print(cx, cy)
cv2.circle(img_out,(int(cx),int(cy)), 2, (0,255,255), -1)

cv2.drawContours(img_out, [max_contour], 0, (0, 255, 100), 2)

cv2.imshow("output", img_out)
cv2.imshow("input", img)
cv2.waitKey(0)