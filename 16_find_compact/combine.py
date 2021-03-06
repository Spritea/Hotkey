import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
from PIL import Image

#包括边边也拼起来
def combine_one(imgs_list, img_path, imgwidth, imgheight):
    im = Image.fromarray(imgs_list[0])
    width, height = im.size
    row_res = imgheight % height
    col_res = imgwidth % width
    img_row = int(imgheight / height) if row_res == 0 else int(imgheight / height) + 1
    # every row in big image contains img_row images
    img_col = int(imgwidth / width) if col_res == 0 else int(imgwidth / width) + 1
    blank = Image.new("RGB", (imgwidth, imgheight))
    for k in range(img_row):
        for j in range(img_col):
            p = Image.fromarray(imgs_list[j + k * img_col])
            if j + 1 == img_col and k + 1 < img_row and col_res > 0:
                box = (width - col_res, 0, width, height)
                p = p.crop(box)
            elif j + 1 < img_col and k + 1 == img_row and row_res > 0:
                box = (0, height - row_res, width, height)
                p = p.crop(box)
            elif j + 1 == img_col and k + 1 == img_row and col_res > 0 and row_res > 0:
                box = (width - col_res, height - row_res, width, height)
                p = p.crop(box)
            blank.paste(p, (width * j, height * k))
    if Path(Out_path).is_dir():
        pass
    else:
        Path(Out_path).mkdir()
    out_path = Out_path + "/" + Path(img_path).stem + ".png"
    blank.save(out_path)


IMG_Path = Path("E:\code\hotkey\\17_post_proc\Vaihingen\\from-pytorch-train\\train16-val17\mv3_1_true_2_res50_data09\pred")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*area4_*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))
pic_small=[]
for j in range(0,len(IMG_Str)):
    pic_small.append(cv.cvtColor(cv.imread(IMG_Str[j],cv.IMREAD_COLOR),cv.COLOR_BGR2RGB))

Out_path = "ceshi"
combine_one(pic_small,"area4_pred",1887,2557)

