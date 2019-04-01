import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
from PIL import Image

Out_path = "pred/"


def combine_one(imgs_list, img_path, imgwidth, imgheight):

    im = Image.fromarray(cv.imread(imgs_list[0]))
    width, height = im.size
    row_res = imgheight % height
    col_res = imgwidth % width
    img_row = int(imgheight / height) if row_res == 0 else int(imgheight / height) + 1
    # every row in big image contains img_row images
    img_col = int(imgwidth / width) if col_res == 0 else int(imgwidth / width) + 1
    blank = Image.new("RGB", (imgwidth, imgheight))
    for k in range(len(imgs_list)):
        name = Path(imgs_list[k]).stem
        number = name.split("-")[1]
        Index = (int(number))
        p=Image.fromarray(cv.imread(imgs_list[k]))
        width_number = 39
        height_number = 49
        row = int((Index + 1) / width_number)
        # start from 0,so +1
        remain = (Index + 1) % width_number
        row_final = row if remain == 0 else row + 1
        tl_y = (row_final - 1) * height
        tl_x = (remain - 1) * width
        blank.paste(p, (tl_x, tl_y))
    if Path(Out_path).is_dir():
        pass
    else:
        Path(Out_path).mkdir()
    out_path = Out_path + "/" + Path(img_path).stem + ".tif"
    blank.save(out_path)


IMG_Path = Path("rotate/dataset_ori/IMG_MY_center/train_labels")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))
# pic_small=[]
# for j in range(0,len(IMG_Str)):
#     pic_small.append(cv.imread(IMG_Str[j],cv.IMREAD_COLOR))
combine_one(IMG_Str,"p1_gt",20408,25373)

