from pathlib import Path
import natsort
import util
from PIL import Image

def load_image(path):
    image = Image.open(path)
    return image

GT_Path = Path("F:\\图像数据集\GID40张\label")
GT_File = natsort.natsorted(list(GT_Path.glob("*.tif")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

class_num=9
color_all=[0]*class_num
for k in range(len(GT_Str)):
    gt=load_image(GT_Str[k])
    color_one_img=gt.getcolors()
    for j in range(len(color_one_img)):
        color_one_cls=color_one_img[j]
        color_all[color_one_cls[1]]+=color_one_cls[0]
print(color_all)


