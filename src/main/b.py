import cv2

img_in = cv2.imread("./../resource/test3.jpg")
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
w, h, c = img_in.shape #img_in is the input image
resize_coeff = 1
img = cv2.resize(img_in, (int(resize_coeff*h), int(resize_coeff*w)))
gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY) # 흑백사진
img = cv2.medianBlur(gray, 17) #Blur
img = cv2.Canny(img, 55, 190) #외곽선 추출

img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
max_index, max_area = max(enumerate([cv2.contourArea(x) for x in contours]), key = lambda x: x[1])
max_contour = contours[max_index]

img_out = cv2.resize(img_in, (int(resize_coeff*h), int(resize_coeff*w)))
cv2.drawContours(img_out, [max_contour], 0, (0, 0, 255), 2)

cv2.imshow("output", img_out)
cv2.imshow("input", img)
cv2.waitKey(0)