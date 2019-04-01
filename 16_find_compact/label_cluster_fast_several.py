import cv2 as cv
from pathlib import Path
import natsort
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import time
# for image after scale,
# label value would change

label_values = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]

def change_one(pic):
    (h,w)=pic.shape[0:2]
    img=pic.reshape(h*w,3)
    new_img=np.zeros((h*w,3),dtype=np.uint8)

    dist=cdist(img,label_values)
    id=np.argmin(dist,axis=1)

    for i in range(h*w):
        new_img[i]=label_values[id[i]]
    out_img=new_img.reshape([h,w,3])
    return out_img

IMG_Path = Path("Vahingen/scale/train16/scale150/train_gt_full")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.tif")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

prefix = "Vahingen/scale/train16/scale150/train_gt_full_edit/"
for k in tqdm(range(len(IMG_Str))):
    in_path = IMG_Str[k]
    pic = cv.imread(in_path)
    out_img=change_one(pic)
    out_path=prefix+Path(in_path).name
    cv.imwrite(out_path,out_img)
# print(dist)
print("kk")
