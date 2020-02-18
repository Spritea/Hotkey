import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
from PIL import Image
from tqdm import tqdm
import util

#以下是combine_several_SCPA.py直接复制的
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

one_large_contain_small = 49
# 这个是一张大图包括多少张小图
IMG_Path = Path("SCPA_WC/small/frrnb")
refer_large_img = cv.imread("SCPA_WC/test_gt/color/2002-test.png")
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
    out_path_prefix = "SCPA_WC/large/color/frrnb/"
    out_name = str(k)+ '_pred.png'
    combine_one(pic_small, out_name, width, height)

#以下是18_preprocee里precode.py直接复制的
def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image

GT_Path = Path("../16_find_compact/SCPA_WC/large/color/frrnb")
GT_File = natsort.natsorted(list(GT_Path.glob("*.png")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

out_prefix="../16_find_compact/SCPA_WC/large/precode/frrnb/"
label_values_RGB_SCPA_WC = [[0,0,0], [128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128]]
for k in tqdm(range(len(GT_Str))):
    gt=load_image(GT_Str[k])
    out=util.reverse_one_hot(util.one_hot_it(gt,label_values_RGB_SCPA_WC))
    out_str=out_prefix+Path(GT_Str[k]).name
    cv.imwrite(out_str,out)
    # print("kk")