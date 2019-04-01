from PIL import Image
import os
from pathlib import Path
import natsort

def slice_slide_one(img_path):
    out_height = 400
    out_width = 400
    overlap_height=100
    overlap_width=100
    slide_height=out_height-overlap_height
    slide_width=out_width-overlap_width
    # slide_height=100
    # slide_width=100
    k = 0
    end=0
    im = Image.open(img_path)
    imgwidth, imgheight = im.size
    for i in range(0, imgheight, slide_height):
        for j in range(0, imgwidth, slide_width):
            if j + out_width <= imgwidth and i + out_height <= imgheight:
                box = (j, i, j + out_width, i + out_height)
            elif j + out_width > imgwidth and i + out_height <= imgheight:
                box = (imgwidth - out_width, i, imgwidth, i + out_height)
                end=1
            elif j + out_width <= imgwidth and i + out_height > imgheight:
                box = (j, imgheight - out_height, j + out_width, imgheight)
            else:
                box = (imgwidth - out_width, imgheight - out_height, imgwidth, imgheight)
                end=2
            a = im.crop(box)
            first_name=str(Path(img_path).stem).zfill(4)
            a.save(os.path.join("Vahingen/test/", first_name+"_%04d.png" % k))
            k += 1


IMG_Path = Path("F:\图像数据集\ISPRS遥感图像分割\Vaihingen\自己整理的\\train_8_edit\\train_gt_full")
IMG_File = natsort.natsorted(list(IMG_Path.glob('*.tif')))
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

for j in IMG_Str:
    slice_slide_one(j)



