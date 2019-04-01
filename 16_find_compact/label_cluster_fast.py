import cv2 as cv
from pathlib import Path
import natsort
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy import spatial
import time
# for image after scale,
# label value would change

label_values = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
# label_values = np.array([[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]],dtype=np.uint8)
# pic=cv.imread("Postdam/scale/train24/scale075/train_gt_full/top_potsdam_2_10_label_075.tif")
pic=cv.imread("Postdam/scale/train24/scale125/train_gt_full/top_potsdam_2_10_label_125.tif")

pic=cv.cvtColor(pic, cv.COLOR_BGR2RGB)
t=time.time()
(h,w)=pic.shape[0:2]
img=pic.reshape(h*w,3)
new_img=np.zeros((h*w,3),dtype=np.uint8)

dist=cdist(img,label_values)
id=np.argmin(dist,axis=1)
# tree = spatial.cKDTree(label_values)
# dist, id = tree.query(img)

for i in range(h*w):
    new_img[i]=label_values[id[i]]
out_img=new_img.reshape([h,w,3])
out_img=cv.cvtColor(out_img, cv.COLOR_RGB2BGR)
cv.imwrite("out1.tif",out_img)

tt=time.time()-t
print(dist)
print("kk")
print("time: %f"%tt)
#a 读图时用pic=cv.cvtColor(pic, cv.COLOR_BGR2RGB)转成rgb的话，
#由于缩放后的label中存在rgb值相同的像素点，
#如rgb=(174,174,174),那么它对应（0,0,255）-rgb-蓝色
#b 不转的话，它对应（0,0,255）-bgr-红色
#实际效果图中，发现rgb值相同的像素点存在于红色区域周边，
#a会导致红色区域周边出现蓝色孤立像素点，视觉效果不好
#故采用b