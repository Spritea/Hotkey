import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
import util

IMG_Path = Path("out")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.bmp")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

for j in range(len(IMG_Str)):
    img = cv.imread(IMG_Str[j], cv.IMREAD_UNCHANGED)
    util.print_result(img, cv.imread("gt/"+str(8+j)+"_gt.bmp"))

    gray = img[:, :, 0]
    src_gray=gray.copy()
    _, contours, _ = cv.findContours(src_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rlt = np.zeros((1500, 1500, 3), dtype=np.uint8)
    for k in range(len(contours)):
        cnt=contours[k]
        approx=cv.approxPolyDP(cnt,10,True)
        cv.polylines(img,[approx],True,(0,0,255),2)
        cv.drawContours(rlt, [approx], -1, (255, 255, 255), thickness=cv.FILLED)

    util.print_result(rlt, cv.imread("gt/"+str(8+j)+"_gt.bmp"))

    # cv.imwrite("refine_10/"+str(Path(IMG_Str[j]).name),rlt)