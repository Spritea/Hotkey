import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
from PIL import Image
from tqdm import tqdm

# 包括边边也拼起来
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
    if Path(out_path_prefix).is_dir():
        pass
    else:
        print("Out path is empty!!")
        exit(0)
    out_path = out_path_prefix + "\\" + img_path
    blank.save(out_path)

one_large_contain_small = 210
# 这个是一张大图包括多少张小图
IMG_Path = Path("GID/07_train38_v2_4band/CAN50/small")
refer_large_img = cv.imread("GID\\06_train38_v2\\test_gt\GF2_PMS1_E113.9_N30.8_20150902_L1A0001015646-MSS1_label.png")
height, width, _ = refer_large_img.shape
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))
large_number = int(len(IMG_Str) / one_large_contain_small)
#这个是一共能拼成几张大图

id_start=0
for k in tqdm(range(large_number)):
    pic_small = []
    id_stop=id_start+one_large_contain_small
    for j in range(id_start,id_stop):
        pic_small.append(cv.cvtColor(cv.imread(IMG_Str[j], cv.IMREAD_COLOR), cv.COLOR_BGR2RGB))
    id_start+=one_large_contain_small
    out_path_prefix = "GID/07_train38_v2_4band/CAN50/large/"
    out_name = str(k)+ '_pred.png'
    combine_one(pic_small, out_name, width, height)
