import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
import random


def random_crop(image, label, crop_height, crop_width, x, y):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    ori_set = {0, crop_height, crop_height * 2, image.shape[0] - crop_height}
    # This is a SET!
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        if x in ori_set:
            x = x - 1
        if y in ori_set:
            y = y - 1
        print("x: %d" % x)
        print("y: %d" % y)
        if len(label.shape) == 3:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width, :]
        else:
            return image[y:y + crop_height, x:x + crop_width], label[y:y + crop_height, x:x + crop_width]
    else:
        raise Exception('Crop shape exceeds image dimensions!')


IMG_Path = Path("opencv_aug/ori")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.bmp")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))
GT_Path=Path("opencv_aug/gt")
GT_File = natsort.natsorted(list(GT_Path.glob("*.bmp")), alg=natsort.PATH)
GT_Str = []
for j in GT_File:
    GT_Str.append(str(j))

id=0
crop_height = 512
crop_width = 512
random_num = 30
random.seed(15)
pic = cv.imread(IMG_Str[0], cv.IMREAD_UNCHANGED)
x = random.sample(range(1, pic.shape[1] - crop_width), random_num)
y = random.sample(range(1, pic.shape[0] - crop_height), random_num)
for k in range(0, len(IMG_Str)):
    a = cv.imread(IMG_Str[k], cv.IMREAD_UNCHANGED)
    b = cv.imread(GT_Str[k], cv.IMREAD_UNCHANGED)
    c = b[:, :, 0]
    id=k%6

    for m in range(0, len(x)):
        img_rd, label_rd = random_crop(a, c, crop_height, crop_width, x[m], y[m])
        first_name = str(Path(IMG_Str[k]).stem)
        sep = first_name.split(sep="_")
        sep[0] = sep[0].zfill(2)
        cv.imwrite(("opencv_aug/slice_ori/" + str(sep[0]) + "-%03d.bmp" % (m+54+id*30)), img_rd)
        cv.imwrite(("opencv_aug/slice_gt/" + "gt-"+str(sep[0]) + "-%03d.bmp" % (m+54+id*30)), label_rd)
