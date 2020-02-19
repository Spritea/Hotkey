import cv2 as cv
import numpy as np
import util
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

#注意：下面这个绘图在y轴的yticks这个函数上有问题，上下会被截去一半
#在ubuntu系统下跑就可以
def img_to_label_list(GT_path,PRED_path):
    GT_img = cv.cvtColor(cv.imread(GT_path, 1), cv.COLOR_BGR2RGB)
    PRED_img = cv.cvtColor(cv.imread(PRED_path, 1), cv.COLOR_BGR2RGB)
    label_values_RGB = [(0, 0, 0), (255, 250, 250), (248, 248, 255), (211, 211, 211),
                        (255, 99, 71), (255, 250, 240), (139, 69, 19), (250, 240, 230),
                        (0, 206, 209), (255, 215, 0), (205, 92, 92), (255, 228, 196),
                        (255, 218, 185), (255, 222, 173), (175, 238, 238), (255, 248, 220),
                        (47, 79, 79), (255, 250, 205), (255, 245, 238), (240, 255, 240),
                        (245, 255, 250), (240, 255, 255), (240, 248, 255), (230, 230, 250),
                        (255, 240, 245), (255, 228, 225), (255, 255, 240), (105, 105, 105),
                        (112, 128, 144), (190, 190, 190), (245, 245, 245), (100, 149, 237),
                        (65, 105, 225), (0, 191, 255), (135, 206, 250), (70, 130, 180),
                        (255, 228, 181), (250, 235, 215), (95, 158, 160), (0, 250, 154),
                        (255, 255, 0), (255, 239, 213), (255, 235, 205)]

    GT_precode = util.reverse_one_hot(util.one_hot_it(GT_img, label_values_RGB))
    PRED_precode = util.reverse_one_hot(util.one_hot_it(PRED_img, label_values_RGB))
    GT_flat = GT_precode.flatten()
    PRED_flat = PRED_precode.flatten()
    return GT_flat,PRED_flat

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的标签文本
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i, format(cm[i, j], fmt),
        #          horizontalalignment="center",
        #          color="white" if cm[i, j] > thresh else "black")
        plt.text(j, i, format(cm[i, j], fmt),va="center",ha="center",
                          color="white" if cm[i, j] > thresh else "black")
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

GT_path='gt.png'
PRED_path='pred.png'
gt,pred=img_to_label_list(GT_path,PRED_path)
cm=confusion_matrix(gt,pred)
class_list=range(cm.shape[0])
str_class_list=[]
for i in range(len(class_list)):
    str_class_list.append(str(class_list[i]))
attack_types =str_class_list
# plot_confusion_matrix(cm,classes=attack_types)
plot_confusion_matrix(cm,classes=attack_types)
plt.show()