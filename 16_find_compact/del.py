import cv2 as cv
import numpy as np

p1=cv.imread("out.tif")
p2 = cv.imread("out5.tif")
print((p1==p2).all())
print("kk")