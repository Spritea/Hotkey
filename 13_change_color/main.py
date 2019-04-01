import cv2 as cv
import numpy as np
from pathlib import Path
import natsort

red_path = Path("slice_gt_png_256")
red_str_posix = natsort.natsorted(list(red_path.glob("*.png")), alg=natsort.PATH)
red_str = []
for i in red_str_posix:
    red_str.append(str(i))


def change_label(img_path):
    image = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    comp=(image==[255])
    image[np.where(comp)]=[1]
    # print("kk")
    # image[np.where((image == [128, 0, 0]).all(axis=2))] = [255, 255, 255]
    return image


for j in range(0, len(red_str)):
    a = red_str[j]
    out_img = change_label(a)
    cv.imwrite("label_256/" + Path(a).name, out_img)
