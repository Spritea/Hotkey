from PIL import Image
import os
from pathlib import Path
import natsort

from multiprocessing.dummy import Pool as ThreadPool

def slice_slide_one(img_path):
    out_height = 512
    out_width = 512
    #for validate set, overlap must=0
    overlap_height=0
    overlap_width=0
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
            a.save(os.path.join("GID\\7class\\train38_val3_v2_small\\val\\", first_name+"%04d.png" % k))
            k += 1


IMG_Path = Path("GID\\7class\\train38_val3_v2\\val")
IMG_File = natsort.natsorted(list(IMG_Path.glob('*.tif')))
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

pool=ThreadPool(4)
#单参数用 pool.map
pool.map(slice_slide_one,IMG_Str)
pool.close()
pool.join()




