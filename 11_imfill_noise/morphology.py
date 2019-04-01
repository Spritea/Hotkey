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
    print(str(Path(IMG_Str[j]).name))
    util.print_result(img, cv.imread("gt/" + str(8 + j) + "_gt.bmp"))

    gray = img[:, :, 0]
    dilatation_size = 1
    dilatation_type = 0
    element = cv.getStructuringElement(dilatation_type, (15 * dilatation_size + 1, 15 * dilatation_size + 1))
    # element = cv.getStructuringElement(dilatation_type, (30 * dilatation_size + 1, 30 * dilatation_size + 1))
    out = cv.morphologyEx(gray, cv.MORPH_CLOSE, element)
    # element2 = cv.getStructuringElement(dilatation_type, (5, 5))
    # blur = cv.morphologyEx(out, cv.MORPH_OPEN, element)

    rgb = np.zeros((1500, 1500, 3), dtype=np.uint8)
    rgb[:, :, 0] = out
    rgb[:, :, 1] = out
    rgb[:, :, 2] = out
    print("After CLOSE: ")
    util.print_result(rgb, cv.imread("gt/" + str(8 + j) + "_gt.bmp"))
    cv.imwrite("morp_out/" +"CLOSE15_" +str(Path(IMG_Str[j]).name), rgb)

    _, contours, _ = cv.findContours(out, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rlt = np.zeros((1500, 1500, 3), dtype=np.uint8)
    cv.drawContours(rlt, contours, -1, (255, 255, 255), thickness=cv.FILLED)
    print("After FILL: ")
    util.print_result(rlt, cv.imread("gt/" + str(8 + j) + "_gt.bmp"))
    cv.imwrite("morp_out/" +"fill_"+ "CLOSE15_"+str(Path(IMG_Str[j]).name), rlt)
