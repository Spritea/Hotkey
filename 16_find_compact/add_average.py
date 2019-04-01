import cv2 as cv
import numpy as np

p1=cv.imread("pred/p1/big/center_pred.png")
p2=cv.imread("time_series/scale/summer_3_4_cure.png")
p3=cv.imread("time_series/scale/summer_half_cure.png")
sum=p1.astype(np.int64)+p2+p3
# sole_chan=sum[:,:,0]
sum[np.where((sum==[0,0,0]).all(axis=2))]=[0,0,0]
sum[np.where((sum==[255,255,255]).all(axis=2))]=[0,0,0]
sum[np.where((sum==[510,510,510]).all(axis=2))]=[255,255,255]
sum[np.where((sum==[765,765,765]).all(axis=2))]=[255,255,255]

cv.imwrite("time_series/scale/ave.png",sum)
print("pp")