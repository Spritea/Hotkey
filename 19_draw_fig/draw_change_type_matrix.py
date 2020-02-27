import cv2 as cv
import numpy as np
import util
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import time

#注意：下面这个绘图在y轴的yticks这个函数上有问题，上下会被截去一半
#在ubuntu系统下跑就可以
def img_to_label_list_not_for_change_type(GT_path,PRED_path,label_values_RGB):
    GT_img = cv.cvtColor(cv.imread(GT_path, 1), cv.COLOR_BGR2RGB)
    PRED_img = cv.cvtColor(cv.imread(PRED_path, 1), cv.COLOR_BGR2RGB)
    GT_precode = util.reverse_one_hot(util.one_hot_it(GT_img, label_values_RGB))
    PRED_precode = util.reverse_one_hot(util.one_hot_it(PRED_img, label_values_RGB))
    GT_flat = GT_precode.flatten()
    PRED_flat = PRED_precode.flatten()
    return GT_flat,PRED_flat

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Land class change matrix', cmap=plt.cm.viridis):
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
    plt.title(title,fontsize=20)
    # plt.colorbar()
    cb=plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes,fontsize=20)
    plt.yticks(tick_marks, classes,fontsize=20)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #下面这个搭配cmap=plt.cm.Blues使用
        # plt.text(j, i, format(cm[i, j], fmt),va="center",ha="center",
        #                   color="white" if cm[i, j] > thresh else "black")
        # 下面这个搭配cmap=plt.cm.viridis使用
        plt.text(j, i, format(cm[i, j], fmt), va="center", ha="center",
                 color="black" if cm[i, j] == cm.max() else "white", fontsize=16)

    # plt.tight_layout()
    plt.ylabel('Source image',fontsize=20)
    plt.xlabel('Destination image',fontsize=20)

t=time.time()
GT_path='../16_find_compact/SCPA_WC/large/color/fcn8s/0_pred.png'
PRED_path='../16_find_compact/SCPA_WC/large/color/fcn8s/1_pred.png'
label_values_RGB_SCPA_WC=[[0,0,0], [128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128]]
gt,pred=img_to_label_list_not_for_change_type(GT_path,PRED_path,label_values_RGB_SCPA_WC)
cm=confusion_matrix(gt,pred)
class_str_list=['Background','Farmland','Bare land','Industrial','Parking','Residential','Water']
plot_confusion_matrix(cm,classes=class_str_list)
tt=time.time()-t
print('Time: %f'%tt)
plt.show()