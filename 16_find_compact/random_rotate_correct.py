import cv2 as cv
import numpy as np
import random
from pathlib import Path
import natsort


def augment(input_image, output_image, h_flip=0, v_flip=0, brightness=0, rotation=0):
    # Data augmentation

    if h_flip:
        # 向右翻
        input_image = cv.flip(input_image, 1)
        output_image = cv.flip(output_image, 1)
    if v_flip:
        # 向下翻
        input_image = cv.flip(input_image, 0)
        output_image = cv.flip(output_image, 0)
    if brightness:
        factor = 1.0 + random.uniform(-1.0 * brightness, brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv.LUT(input_image, table)
    if rotation:
        # 这种旋转方式没有黑边，绝对正确
        # 这里的input_image和output_image都是之前操作的输出
        trans_in = cv.transpose(input_image)
        trans_out = cv.transpose(output_image)
        if rotation == 90:
            input_image = cv.flip(trans_in, 1)
            output_image = cv.flip(trans_out, 1)
        elif rotation == 270:
            input_image = cv.flip(trans_in, 0)
            output_image = cv.flip(trans_out, 0)
        elif rotation == 180:
            # -1表示同时水平和竖直翻转
            # 此处要用原图
            input_image = cv.flip(input_image, -1)
            output_image = cv.flip(output_image, -1)
        else:
            print("Unsupported Angle!!")
    return input_image, output_image


IMG_Path = Path("ceshi/ori/train")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

GT_Path = Path("ceshi/ori/train_gt_full")
GT_File = natsort.natsorted(list(GT_Path.glob("*.png")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

h_flip_list = [0, 0, 0, 0, 1, 1, 1, 1]
v_flip = 0
brightness = 0
rotation_list = [0, 90, 180, 270, 0, 90, 180, 270]

for j in range(len(h_flip_list)):
    flag = 0
    h_flip = h_flip_list[j]
    rotation = rotation_list[j]
    prefix_ori = 'ceshi/aug/train/'
    prefix_gt = 'ceshi/aug/train_gt_full/'
    if h_flip > 0:
        prefix_ori = prefix_ori + 'h'
        prefix_gt = prefix_gt + 'h'
    if v_flip > 0:
        prefix_ori = prefix_ori + 'v'
        prefix_gt = prefix_gt + 'v'
    if brightness > 0:
        prefix_ori = prefix_ori + 'b' + str(brightness)
        prefix_gt = prefix_gt + 'b' + str(brightness)
    if rotation > 0:
        prefix_ori = prefix_ori + 'r' + str(rotation)
        prefix_gt = prefix_gt + 'r' + str(rotation)
    if h_flip+v_flip+brightness+rotation==0:
        flag = 1
        print("Copy origin images!!")
    for k in range(len(IMG_Str)):
        ori_out, gt_out = augment(cv.imread(IMG_Str[k]), cv.imread(GT_Str[k]), h_flip=h_flip, v_flip=v_flip,
                                  brightness=brightness, rotation=rotation)
        suffix = str(k).zfill(5) + '.png'
        if flag == 0:
            ori_name = prefix_ori + '-' + suffix
            gt_name = prefix_gt + '-' + suffix
        else:
            ori_name = prefix_ori + suffix
            gt_name = prefix_gt + suffix
        cv.imwrite(ori_name, ori_out)
        cv.imwrite(gt_name, gt_out)
