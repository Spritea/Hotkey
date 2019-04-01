import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random as rng

img = cv.imread("center_gt_edit.png")
# img[:,:,0]=img[:,:,2]!=img[:,:,1]
t1=img[:,:,0]
# gray=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
ret, th = cv.threshold(t1, 0, 255, cv.THRESH_OTSU)
cv.imwrite("center_gt_edit_final.png",th)
print("kk")