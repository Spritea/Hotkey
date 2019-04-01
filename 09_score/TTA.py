import cv2 as cv
import numpy as np
from pathlib import Path
import natsort

IMG_Path = Path("tta/reverse")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.bmp")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))
out=np.zeros((1500,1500),dtype=np.float)
for j in range(len(IMG_Str)):
    pic = cv.imread(IMG_Str[j])
    gray=pic[:,:,0]
    out=out+gray
mask=out/6
out_gray=np.zeros((1500,1500),dtype=np.uint8)
comp=(mask>=127.5)
out_gray[np.where((comp))]=255
final=np.zeros((1500,1500,3),dtype=np.uint8)
final[:,:,0]=out_gray
final[:,:,1]=out_gray
final[:,:,2]=out_gray
cv.imwrite("average.bmp",final)
print("kk")