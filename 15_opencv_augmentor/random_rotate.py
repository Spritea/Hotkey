import cv2 as cv
import numpy as np
import random
from pathlib import Path
import natsort
import math


def augment(input_image, output_image, h_flip=0, v_flip=0, brightness=0, rotation=0):
    # Data augmentation

    if h_flip:
        input_image = cv.flip(input_image, 1)
        output_image = cv.flip(output_image, 1)
    if v_flip:
        input_image = cv.flip(input_image, 0)
        output_image = cv.flip(output_image, 0)
    if brightness:
        factor = 1.0 + random.uniform(-1.0 * brightness, brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv.LUT(input_image, table)
    if rotation:
        # angle = random.uniform(-1 * rotation, rotation)
        M = cv.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), rotation, 1.0)
        # // means divide result is int
        width=input_image.shape[1]
        height=input_image.shape[0]
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_width = int(width * cos + height * sin)
        new_height = int(width * sin + height * cos)
        M[0,2]+=(new_width-width)//2
        M[1,2]+=(new_height-height)//2
        input_image = cv.warpAffine(input_image, M, (new_width, new_height),
                                    flags=cv.INTER_NEAREST)
        output_image = cv.warpAffine(output_image, M, (new_width, new_height),
                                     flags=cv.INTER_NEAREST)

    return input_image, output_image


IMG_Path = Path("off")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.tiff")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

GT_Path = Path("off")
GT_File = natsort.natsorted(list(GT_Path.glob("*.bmp")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

h_flip = 0
v_flip = 0
brightness = 0
rotation = 90

for k in range(len(IMG_Str)):
    ori_out, gt_out = augment(cv.imread(IMG_Str[k]), cv.imread(GT_Str[k]), h_flip=h_flip, v_flip=v_flip,
                              brightness=brightness, rotation=rotation)

    if h_flip > 0:
        ori_name = "angle/ori/" + Path(IMG_Str[k]).stem + "_h" + ".bmp"
        gt_name = "angle/gt/" + Path(GT_Str[k]).stem + "_h" + ".bmp"
    elif v_flip > 0:
        ori_name = "angle/ori/" + Path(IMG_Str[k]).stem + "_v" + ".bmp"
        gt_name = "angle/gt/" + Path(GT_Str[k]).stem + "_v" + ".bmp"
    elif brightness > 0:
        ori_name = "angle/ori/" + Path(IMG_Str[k]).stem + "_b" + str(brightness) + ".bmp"
        gt_name = "angle/gt/" + Path(GT_Str[k]).stem + "_b" + str(brightness) + ".bmp"
    elif rotation > 0:
        ori_name = "angle/ori/" + Path(IMG_Str[k]).stem + "_r" + str(rotation) + ".bmp"
        gt_name = "angle/gt/" + Path(GT_Str[k]).stem + "_r" + str(rotation) + ".bmp"
    else:
        print("Choose one and only one ops! ")

    cv.imwrite(ori_name, ori_out)
    cv.imwrite(gt_name,gt_out)
