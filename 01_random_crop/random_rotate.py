import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
import random

def presave(IMG_Str):
    pic = cv.imread(IMG_Str[1], cv.IMREAD_UNCHANGED)
    cand_rect=[]
    image=pic.copy()
    gray = image[:, :, 0]
    gray_top = gray[0, :] if np.nonzero(gray[0, :])[0].size > 0 else gray[1, :]
    gray_bottom = gray[2048, :] if np.nonzero(gray[2048, :])[0].size > 0 else gray[2047, :]
    cord_top = np.nonzero(gray_top)[0][0]
    cord_bottom=np.nonzero(gray_bottom)[0][0]
    cord_left=cord_bottom
    cord_right=cord_top
    angle = 30 if cord_top > 1000 else 60
    big_rect = ((2049 // 2, 2049 // 2), (1500, 1500), -angle)
    # pts=np.array([[cord_top,0],[0,cord_left],[cord_bottom,2048],[2048,cord_right]],dtype=np.int64)
    # pts=pts.reshape((-1,1,2))
    # cv.polylines(image,[pts],True,(0,0,255))
    # small_rect = ((222 + crop_width // 2, 914 + crop_height // 2), (crop_width, crop_height), 0)
    # ppt=cv.boxPoints(big_rect).astype(np.int64)
    # # ppt = ppt.reshape((-1, 1, 2))
    # cv.polylines(image,[ppt],True,(0,0,255),thickness=5)
    # inter_type = cv.rotatedRectangleIntersection(big_rect, small_rect)
    # print(inter_type[0])
    # # cv.rectangle(image,(222,914),(734,1426),(0,0,255),thickness=5)
    # cv.imwrite("test3.bmp",image)
    for x in range(1, pic.shape[1] - crop_width):
        for y in range(1, pic.shape[0] - crop_height):

            small_rect = ((x + crop_width // 2, y + crop_height // 2), (crop_width, crop_height), 0)
            inter_type = cv.rotatedRectangleIntersection(big_rect, small_rect)
            if inter_type[0] == 2:
                cand_rect.append([x, y])
    cand_np = np.asarray(cand_rect)
    np.save("cand_angle_60.npy", cand_np)


def random_crop(image, label, crop_height, crop_width, x, y):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):

        print("x: %d" % x)
        print("y: %d" % y)
        if len(label.shape) == 3:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width, :]
        else:
            return image[y:y + crop_height, x:x + crop_width], label[y:y + crop_height, x:x + crop_width]
    else:
        raise Exception('Crop shape exceeds image dimensions!')


IMG_Path = Path("opencv_aug_random_rotate/ori")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.bmp")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))
GT_Path = Path("opencv_aug_random_rotate/gt")
GT_File = natsort.natsorted(list(GT_Path.glob("*.bmp")), alg=natsort.PATH)
GT_Str = []
for j in GT_File:
    GT_Str.append(str(j))

id = 0
crop_height = 512
crop_width = 512
random_num = 30
random.seed(15)

# presave(IMG_Str)

cand_angle_30 = np.load("cand_angle_30.npy")
cand_angle_60 = np.load("cand_angle_60.npy")
cand_rect_30 = cand_angle_30.tolist()
cand_rect_60 = cand_angle_60.tolist()



for k in range(0, len(IMG_Str)):
    a = cv.imread(IMG_Str[k], cv.IMREAD_UNCHANGED)
    b = cv.imread(GT_Str[k], cv.IMREAD_UNCHANGED)
    # c = b[:, :, 0]
    id = k % 6

    gray = a[:, :, 0]
    gray_top = gray[0, :] if np.nonzero(gray[0, :])[0].size > 0 else gray[1, :]
    cord_top = np.nonzero(gray_top)[0][0]
    angle = 30 if cord_top > 1000 else 60
    if angle == 30:
        cord = random.sample(cand_rect_30, random_num)
    else:
        cord = random.sample(cand_rect_60, random_num)

    for m in range(0, len(cord)):
        img_rd, label_rd = random_crop(a, b, crop_height, crop_width, cord[m][0], cord[m][1])
        first_name = str(Path(IMG_Str[k]).stem)
        sep = first_name.split(sep="_")
        sep[0] = sep[0].zfill(2)
        cv.imwrite(("opencv_aug_random_rotate/slice_ori/" + str(sep[0]) + "-%03d.bmp" % (m + 234 + id * 30)), img_rd)
        cv.imwrite(("opencv_aug_random_rotate/slice_gt/" + "gt-" + str(sep[0]) + "-%03d.bmp" % (m + 234 + id * 30)),
                   label_rd)
