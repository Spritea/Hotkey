import numpy as np
import cv2 as cv
import util

img=cv.imread("9.bmp")
a=cv.imread("gt/9_gt.bmp")
util.print_result(img,a)
gray=img[:,:,0]
src_gray=gray.copy()
_, contours, _ = cv.findContours(src_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# for cnt in contours:
src=np.zeros((1500,1500,3),dtype=np.uint8)
cv.drawContours(src_gray,contours,-1,(255),thickness=cv.FILLED)
src[:,:,0]=src_gray
src[:,:,1]=src_gray
src[:,:,2]=src_gray
util.print_result(src,cv.imread("gt/8_gt.bmp"))
_, contours, _ = cv.findContours(src_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print("Original Contour Number: %d" % len(contours))
refine_contours = []
for cnt in contours:
    # print(cnt.size)
    area = abs(cv.contourArea(cnt))
    if area > 25 * 25:
        refine_contours.append(cnt)
print("Reduced: %d" % len(refine_contours))
rlt=np.zeros((1500,1500,3),dtype=np.uint8)
cv.drawContours(rlt,refine_contours,-1,(255,255,255),thickness=cv.FILLED)
# util.print_result(rlt,cv.imread("gt/8_gt.bmp"))
cv.imwrite("out/noise_9.bmp",rlt)


# cv.namedWindow('Contours',cv.WINDOW_NORMAL)
# cv.imshow('Contours', src_gray)
# cv.waitKey(0)