import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import numpy as np
import util


def draw_bar(num_list1, num_list2, label_list, xlabel, ylabel, title):
    """
    绘制条形图
    x:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """
    x = np.arange(len(num_list1))
    bar_width=0.45
    rects1 = plt.bar(x - bar_width/2, height=num_list1, width=bar_width, color= '#4472C4', label="Source")
    rects2 = plt.bar(x + bar_width/2, height=num_list2, width=bar_width, color='#ED7D31', label="Destination")
    plt.ylim(0, 0.4)
    # y轴取值范围
    plt.ylabel(ylabel,fontsize=25)
    plt.yticks(fontsize=20)
    plt.xticks(x, label_list,fontsize=25)
    #设置x轴中点坐标和显示值
    plt.xlabel(xlabel,fontsize=25)
    plt.title(title, fontsize=25)
    plt.legend(fontsize=25)
    # 设置题注
    for rect in rects1:
        height = rect.get_height()
        # height_ratio=format(height/sum(num_list1)*100,'.2f')
        height_ratio=format(height*100,'.2f')
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.001, str(height_ratio)+'%', ha="center", va="bottom",fontsize=17)
    for rect in rects2:
        height = rect.get_height()
        # height_ratio=format(height/sum(num_list2)*100,'.2f')
        height_ratio = format(height * 100, '.2f')
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.001, str(height_ratio)+'%', ha="center", va="bottom",fontsize=17)
    plt.show()

def takesecond(elem):
    return elem[1]
def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image

def compute_one(img_path):
    label_values_RGB_SCPA_WC = [[0,0,0], [128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128]]
    gt = load_image(img_path)
    out = util.reverse_one_hot(util.one_hot_it(gt, label_values_RGB_SCPA_WC))
    out_str = 'out.png'
    cv.imwrite(out_str,out)
    image = Image.open(out_str)
    a = image.getcolors()
    a.sort(key=takesecond)
    print(a)
    print("color kind: %d" % len(a))
    return a

src_path='2002ps.png'
dst_path='2009ps.png'
src_list=compute_one(src_path)
dst_list=compute_one(dst_path)
src_list_no_label=[]
dst_list_no_label=[]
for i in range(len(src_list)):
    src_list_no_label.append(src_list[i][0])
for i in range(len(dst_list)):
    dst_list_no_label.append(dst_list[i][0])
src_list_ratio=[]
dst_list_ratio=[]
for i in range(len(src_list_no_label)):
    decimal=src_list_no_label[i]/sum(src_list_no_label)
    src_list_ratio.append(decimal)
for i in range(len(dst_list_no_label)):
    decimal=dst_list_no_label[i]/sum(dst_list_no_label)
    dst_list_ratio.append(decimal)

print(src_list_ratio)
print(dst_list_ratio)
xlabel = "Land class"
ylabel = "Proportion"
title = "Land class distribution"
label_list = ['Back', 'Farm', 'Bare', 'Indust','Parking','Res','Water']
# 横坐标刻度显示值
draw_bar(src_list_ratio,dst_list_ratio,label_list,xlabel,ylabel,title)