import cv2 as cv
from pathlib import Path
import natsort
import numpy as np
from tqdm import tqdm
# for image after scale,
# label value would change

label_values = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
def compute_nearest(pix):
    distance_list=[]
    for i in range(len(label_values)):
        dis=abs(pix[0]-label_values[i][0])+abs(pix[1]-label_values[i][1])+abs(pix[2]-label_values[i][2])
        distance_list.append(dis)
    idx=distance_list.index(min(distance_list))
    pixel_change=label_values[idx]
    return pixel_change

IMG_Path = Path("Postdam/scale/train24/scale075/train_gt_full")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.tif")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

for j in tqdm(range(len(IMG_Str))):
    in_path=IMG_Str[j]
    pic=cv.imread(in_path)
    height=pic.shape[0]
    width=pic.shape[1]
    for row in range(height):
        for col in range(width):
            pix=pic[row,col]
            pix_change=compute_nearest(pix)
            pic[row,col]=pix_change
    prefix="Postdam/scale/train24/scale075/train_gt_full_edit/"
    out_path=prefix+Path(in_path).name
    cv.imwrite(out_path,pic)
