from PIL import Image
import os
from pathlib import Path
import natsort
from tqdm import tqdm

#把多的边边单独割出来
def slice_slide_one(img_path):
    out_height = 512
    out_width = 512
    #for validate set, overlap=0
    overlap_height=0
    overlap_width=0
    slide_height=out_height-overlap_height
    slide_width=out_width-overlap_width
    # slide_height=100
    # slide_width=100
    k = 0
    im = Image.open(img_path)
    imgwidth, imgheight = im.size
    for i in range(0, imgheight, slide_height):
        for j in range(0, imgwidth, slide_width):
            if j + out_width <= imgwidth and i + out_height <= imgheight:
                box = (j, i, j + out_width, i + out_height)
            elif j + out_width > imgwidth and i + out_height <= imgheight:
                box = (j, i, imgwidth, i + out_height)
            elif j + out_width <= imgwidth and i + out_height > imgheight:
                box = (j, i, j + out_width, imgheight)
            else:
                box = (j, i, imgwidth, imgheight)
            a = im.crop(box)
            first_name=str(Path(img_path).stem).zfill(4)
            a.save(os.path.join("Postdam/train_18_val_20/different_shape_val/", first_name+"_%04d.png" % k))
            k += 1


IMG_Path = Path("E:\code\hotkey\\17_post_proc\Postdam\\from-pytorch-train\data14\\val")
IMG_File = natsort.natsorted(list(IMG_Path.glob('*.tif')))
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

for j in tqdm(IMG_Str):
    slice_slide_one(j)



