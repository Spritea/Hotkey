from pathlib import Path
import natsort
import util
from PIL import Image

def load_image(path):
    image = Image.open(path)
    return image

GT_Path = Path("GID/7class/label_precode")
GT_File = natsort.natsorted(list(GT_Path.glob("*.png")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

class_num=8
color_all=[0]*class_num
total_pixel=7200*6800*41
for k in range(len(GT_Str)):
    gt=load_image(GT_Str[k])
    color_one_img=gt.getcolors()
    color_single=[0]*class_num
    for j in range(len(color_one_img)):
        color_one_cls=color_one_img[j]
        color_all[color_one_cls[1]]+=color_one_cls[0]
        color_single[color_one_cls[1]]+=color_one_cls[0]
    # color_one_ratio = [y / total_pixel for y in color_single]
    # print('path: %s', Path(GT_Str[k]).name)
    # print(color_one_ratio)
color_all_ratio=[x/total_pixel for x in color_all]
print(color_all)
print(color_all_ratio)


