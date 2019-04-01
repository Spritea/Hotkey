import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
from PIL import Image

def slice_one(img_path):
    height = 50
    width = 50
    k = 0
    im = Image.open(img_path)
    imgwidth, imgheight = im.size
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            if j + width <= imgwidth and i + height <= imgheight:
                box = (j, i, j + width, i + height)
            elif j + width > imgwidth and i + height <= imgheight:
                box = (imgwidth - width, i, imgwidth, i + height)
            elif j + width <= imgwidth and i + height > imgheight:
                box = (j, imgheight - height, j + width, imgheight)
            else:
                box = (imgwidth - width, imgheight - height, imgwidth, imgheight)
            a = im.crop(box)
            first_name = str(Path(img_path).stem).zfill(2)
            a.save("slice_rlt" + "/" + first_name + "-%03d.bmp" % k)
            k += 1