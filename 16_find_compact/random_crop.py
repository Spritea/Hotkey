import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
import random


def cord_to_index(x, y):
    width_number = 39
    # height_number = 49
    col = (int(x / crop_width)) + 1
    row = (int(y / crop_height)) + 1
    index = (row - 1) * width_number + col - 1
    return index


def random_crop(image, label, crop_height, crop_width, x, y):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
    is_repeat = 0
    not_in_index = 0
    index = cord_to_index(x, y)
    index_1 = cord_to_index(x + crop_width, y)
    index_2 = cord_to_index(x, y + crop_height)
    index_3 = cord_to_index(x + crop_width, y + crop_height)
    if x % crop_width == 0 and y % crop_height == 0:
        is_repeat = 1
        return is_repeat, 0, 0
    if index in Index and index_1 in Index and index_2 in Index and index_3 in Index:
        if len(label.shape) == 3:
            return 0, image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width, :]
        else:
            return 0, image[y:y + crop_height, x:x + crop_width], label[y:y + crop_height, x:x + crop_width]
    else:
        not_in_index = 1
        return not_in_index, 0, 0


IMG_Path = Path("random_crop/dataset_ori/IMG_MY_center/train")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.bmp")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))
Index = []
for j in range(len(IMG_Str)):
    name = Path(IMG_Str[j]).stem
    number = name.split("-")[1]
    Index.append(int(number))

sum = 0
crop_height = 512
crop_width = 512
random_num = 3000
total=500
random.seed(15)
pic = cv.imread("p1.tif")
gt = cv.imread("p1_gt.tif")
x = random.sample(range(1, pic.shape[1] - crop_width), random_num)
y = random.sample(range(1, pic.shape[0] - crop_height), random_num)

for m in range(0, len(x)):
    pointer, img_rd, label_rd = random_crop(pic, gt, crop_height, crop_width, x[m], y[m])
    if pointer == 1:
        continue
    cv.imwrite(("random_crop/dataset_aug/p1/train/" + "crop" + "-%05d.bmp" % sum), img_rd)
    cv.imwrite(("random_crop/dataset_aug/p1/train_labels/" + "crop" + "-%05d.png" % sum), label_rd)
    sum+=1
    if sum==total:
        break
print(sum)
#random_num:3000 sum:690
#random_num:2000 sum:433
#random_num:2500 sum:558