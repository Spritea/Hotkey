import cv2 as cv
from pathlib import Path
import natsort
import util
from tqdm import tqdm
from PIL import Image
import itertools

#输入：SCPA结果图
#输出：这张图里面有哪些变化类型
#过程：先precode，再输出每个label代表的具体变化类型


def takefirst(elem):
    return elem[0]

def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image

GT_Path = Path("test")
GT_File = natsort.natsorted(list(GT_Path.glob("*.png")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

out_prefix="test/out/"
# label_values_RGB_SCPA_WC = [[0,0,0], [128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128]]
candy_color_list = [(0, 0, 0), (255, 250, 250), (248, 248, 255), (211, 211, 211),
                    (255 ,99 ,71), (255, 250, 240), (139 ,69 ,19), (250, 240, 230),
                    (0, 206, 209), (255,215,0), (205,92,92), (255, 228, 196),
                    (255, 218, 185), (255, 222, 173), (175, 238, 238), (255, 248, 220),
                    (47, 79, 79), (255, 250, 205), (255, 245, 238), (240, 255, 240),
                    (245, 255, 250), (240, 255, 255), (240, 248, 255), (230, 230, 250),
                    (255, 240, 245), (255, 228, 225), (255,255,240), (105, 105, 105),
                    (112, 128, 144), (190, 190, 190), (245,245,245), (100, 149, 237),
                    (65, 105, 225), (0, 191, 255), (135, 206, 250), (70, 130, 180),
                    (255,228,181), (250,235,215), (95, 158, 160), (0, 250, 154),
                    (255, 255, 0), (255, 239, 213), (255, 235, 205)]
for k in tqdm(range(len(GT_Str))):
    gt=load_image(GT_Str[k])
    # out=util.reverse_one_hot(util.one_hot_it(gt,label_values_RGB_SCPA_WC))
    out = util.reverse_one_hot(util.one_hot_it(gt, candy_color_list))
    out_str=out_prefix+Path(GT_Str[k]).name
    cv.imwrite(out_str,out)
    # print("kk")

image = Image.open(out_str)
a = image.getcolors()
a.sort(key=takefirst, reverse=True)
print(a)
print("color kind: %d" % len(a))

image = Image.open(GT_Str[0])
b = image.getcolors()
b.sort(key=takefirst, reverse=True)
print(b)

land_class_list = [0, 1, 2, 3, 4, 5, 6]
class_list = list(itertools.product(land_class_list, repeat=2))
class_list_reduce = class_list.copy()
for k in range(len(class_list)):
    if class_list[k][0] == class_list[k][1] and class_list[k][0] > 0:
        class_list_reduce.remove(class_list[k])

original_change_type=[]
for i in range(len(a)):
    change_type_label=a[i][1]
    original_change_type.append(class_list_reduce[change_type_label])
print("original change type")
print(original_change_type)