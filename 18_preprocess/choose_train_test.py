from pathlib import Path
import natsort
import util
from PIL import Image
import itertools
from tqdm import tqdm
import numpy as np

#用来划分数据集-找各个类别占对应类别总数最少的几张图片构成test集

def load_image(path):
    image = Image.open(path)
    return image

# a=np.load('ratio.npy')
GT_Path = Path("GID/7class/label_precode")
GT_File = natsort.natsorted(list(GT_Path.glob("*.png")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

class_num=8
img_num=41
test_num=3
pixels_each_img=[]
color_all=[0]*class_num

for k in range(len(GT_Str)):
    gt=load_image(GT_Str[k])
    color_one_img=gt.getcolors()
    color_single=[0]*class_num
    for j in range(len(color_one_img)):
        color_one_cls=color_one_img[j]
        color_single[color_one_cls[1]]+=color_one_cls[0]
        color_all[color_one_cls[1]] += color_one_cls[0]
    pixels_each_img.append(color_single)

combines=list(itertools.combinations(range(img_num),test_num))
print('kk')

ratio_all=[]

for j in tqdm(range(len(combines))):
    situation_single=combines[j]
    pixels_test = [0] * class_num
    ratio_single_situation = []
    for m in range(len(situation_single)):
        img_index=situation_single[m]
        color_single_2=pixels_each_img[img_index]
        pixels_test=[pixels_test[p]+color_single_2[p] for p in range(len(color_single_2))]
    for n in range(len(pixels_test)):
        ratio_single_situation.append(pixels_test[n]/color_all[n])
    ratio_single_situation_sum=sum(ratio_single_situation)
    # print(ratio_single_situation_sum)
    ratio_all.append(ratio_single_situation_sum)

print(ratio_all)
combine_min=ratio_all.index(min(ratio_all))
combine_rlt=combines[combine_min]
print(combine_min)
print(combine_rlt)
for q in range(len(combine_rlt)):
    print(GT_Str[combine_rlt[q]])
ratio_file=np.array(ratio_all)
# np.save('ratio.npy',ratio_file)
print('kk')
