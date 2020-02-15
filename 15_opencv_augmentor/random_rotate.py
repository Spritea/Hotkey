import cv2 as cv
import numpy as np
import random
from pathlib import Path
import natsort
from tqdm import tqdm

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


IMG_Path = Path("../18_preprocess/GID_name_in_order/7class/train38_val3/train")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

GT_Path = Path("../18_preprocess/GID_name_in_order/7class/train38_val3/train_gt")
GT_File = natsort.natsorted(list(GT_Path.glob("*.png")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

h_flip = 0
v_flip = 1
brightness = 0
rotation = 0

IMG_Path_out='../15_opencv_augmentor/GID/train38_val3/train/'
GT_Path_out='../15_opencv_augmentor/GID/train38_val3/train_gt/'

IMG_out_type='.png'
GT_out_type='.png'


for k in tqdm(range(len(IMG_Str))):
    ori_out, gt_out = augment(cv.imread(IMG_Str[k]), cv.imread(GT_Str[k],-1), h_flip=h_flip, v_flip=v_flip,
                              brightness=brightness, rotation=rotation)
    #-1用来保证单通道进去的标签别自动变成3通道
    if h_flip > 0:
        ori_name = IMG_Path_out + Path(IMG_Str[k]).stem + "_h" + IMG_out_type
        gt_name = GT_Path_out + Path(GT_Str[k]).stem + "_h" + GT_out_type
    elif v_flip > 0:
        ori_name = IMG_Path_out + Path(IMG_Str[k]).stem + "_v" + IMG_out_type
        gt_name = GT_Path_out + Path(GT_Str[k]).stem + "_v" + GT_out_type
    elif brightness > 0:
        ori_name = IMG_Path_out + Path(IMG_Str[k]).stem + "_b" + str(brightness) + IMG_out_type
        gt_name = GT_Path_out + Path(GT_Str[k]).stem + "_b" + str(brightness) + GT_out_type
    elif rotation > 0:
        ori_name = IMG_Path_out + Path(IMG_Str[k]).stem + "_r" + str(rotation) + IMG_out_type
        gt_name = GT_Path_out + Path(GT_Str[k]).stem + "_r" + str(rotation) + GT_out_type
    else:
        print("Choose one and only one ops! ")

    cv.imwrite(ori_name, ori_out)
    cv.imwrite(gt_name,gt_out)
