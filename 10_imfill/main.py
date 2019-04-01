import numpy as np
import cv2 as cv

img=cv.imread("10.bmp")
gray=img[:,:,0]
src_gray=gray.copy()
_, contours, _ = cv.findContours(src_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# for cnt in contours:
cv.drawContours(src_gray,contours,-1,(255),thickness=cv.FILLED)

cv.imwrite("out/10.bmp",src_gray)
# cv.namedWindow('Contours',cv.WINDOW_NORMAL)
# cv.imshow('Contours', src_gray)
# cv.waitKey(0)