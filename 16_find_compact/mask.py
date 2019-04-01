import cv2 as cv
import numpy as np

mask = cv.imread("time_series/del_remove_small_only_2000.png")
pic = cv.imread("time_series/TSX_20130928_p1.tif")
# sole_chan=sum[:,:,0]
pic[np.where((mask == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
cv.imwrite("time_series/mask.png", pic)
print("pp")
