from PIL import Image
# from pathlib import Path
import natsort
import os

global k
k = 0


def slice_one(img_path):
    height = 512
    width = 512
    global k
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
            # first_name = str(Path(img_path).stem)
            # sep = first_name.split(sep="_")
            # sep[0] = sep[0].zfill(2)
            # a.save(("tt/gt-" + str(sep[0]) + "-%03d.bmp" % k))
            a.save(os.path.join(dir, "TDX-p1-%05d.bmp" % k))
            k += 1
            # if k == 54:
            #     k = 0
            
dir="TX/small/summer_3_4/"
slice_one("TX/TDX_20131225_p1_3_4.tif")


# IMG_Path = Path("opencv_aug/gt")
# IMG_File = natsort.natsorted(list(IMG_Path.glob('*.bmp')), alg=natsort.PATH)
# IMG_Str = []
# for i in IMG_File:
#     IMG_Str.append(str(i))
#
# for j in IMG_Str:
#     slice_one(j)
