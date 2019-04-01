import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
import shutil

IMG_Path = Path("E:\code\hotkey\\01_random_crop\IMG_MY_random\slice_random_gt")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.bmp")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))
GT_Path=Path("off/gt/output")
GT_File = natsort.natsorted(list(GT_Path.glob("*gt*.bmp")), alg=natsort.PATH)
GT_Str = []
for j in GT_File:
    GT_Str.append(str(j))

pre=str(Path(IMG_Str[0]).parent)

for k in IMG_Str:
    file_name=(Path(k)).stem
    parts=file_name.split("-")
    back=parts[1]+"-"+parts[0]+"-"+parts[2]+".bmp"
    dst=pre+"\\"+back
    shutil.move(k,dst)

# for k in range(0,len(IMG_Str)):
#     pre="off/clear/ori/"
#     back=str(k+11)+".tiff"
#     dst=pre+back
#     shutil.move(IMG_Str[k],dst)
#
# for k in range(0,len(GT_Str)):
#     pre="off/clear/gt/"
#     back=str(k+11)+".bmp"
#     dst=pre+back
#     shutil.copy(GT_Str[k],dst)