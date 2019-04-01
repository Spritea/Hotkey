import shutil
from pathlib import Path
import natsort
import random
import cv2 as cv
import numpy as np

IMG_Path = Path("out_crop30_aug/slice_ori")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.bmp")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

GT_Path = Path("out_crop30_aug/slice_gt")
GT_File = natsort.natsorted(list(GT_Path.glob("*.bmp")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

# for j in range(0,len(IMG_Str)):
#     stem=str(Path(IMG_Str[j]).stem)
#     new_name="out_crop30_aug/slice_ori_jpg_24bit/"+stem+".jpg"
#     img = cv.imread(IMG_Str[j], cv.IMREAD_COLOR)
#     # new_img=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
#     # new_img[:,:,0]=img
#     # new_img[:, :, 1] = img
#     # new_img[:, :, 2] = img
#     cv.imwrite(new_name,img)

for j in range(0,len(GT_Str)):
    stem=str(Path(GT_Str[j]).stem)
    new_name="out_crop30_aug/slice_gt_png/"+stem+".png"
    img=cv.imread(GT_Str[j],cv.IMREAD_COLOR)
    one_channel=img[:,:,0]
    comp = (one_channel == [255])
    one_channel[np.where(comp)] = [1]
    cv.imwrite(new_name,one_channel)



