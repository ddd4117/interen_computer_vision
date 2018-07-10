import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('./../resource/output.xml')

img = cv.imread("./../resource/test3.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 7)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()