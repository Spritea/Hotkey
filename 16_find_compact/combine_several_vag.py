import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
from PIL import Image
from tqdm import tqdm

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
    if Path(out_path_prefix).is_dir():
        pass
    else:
        print("Out path is empty!!")
        exit(0)
    out_path = out_path_prefix + "\\" + img_path
    blank.save(out_path)

#Vahingen train23
id_list=['16','20','22','24','27','29','31','33','35','38']
#Vahingen train16-benchmark
# id_list=['2','4','6','8','10','12','14','16','20','22','24','27','29','31','33','35','38']

IMG_Path = Path("E:\code\hotkey\\17_post_proc\Vaihingen\\from-pytorch-train\\train23-val10\dlv3\pred")
Large_Path = Path("E:\code\hotkey\\17_post_proc\Vaihingen\\from-pytorch-train\\train23-val10\\val_gt_full")
Large_File = natsort.natsorted(list(Large_Path.glob("*.tif")), alg=natsort.PATH)
Large_Str = []
for j in Large_File:
    Large_Str.append(str(j))

for k in tqdm(range(len(id_list))):
    glob_target='*area'+id_list[k]+'_*.png'
    IMG_File = natsort.natsorted(list(IMG_Path.glob(glob_target)), alg=natsort.PATH)
    IMG_Str = []
    for i in IMG_File:
        IMG_Str.append(str(i))
    pic_small=[]
    for j in range(0,len(IMG_Str)):
        pic_small.append(cv.cvtColor(cv.imread(IMG_Str[j], cv.IMREAD_COLOR), cv.COLOR_BGR2RGB))
    large_img=cv.imread(Large_Str[k])
    height,width,_=large_img.shape

    out_path_prefix = "E:\code\hotkey\\17_post_proc\Vaihingen\\from-pytorch-train\\train23-val10\dlv3\\test"
    out_name='area'+id_list[k]+'_pred.png'
    combine_one(pic_small,out_name,width,height)

