import cv2 as cv
import numpy as np
import util
import time
from sklearn.metrics import f1_score, accuracy_score, \
    jaccard_score, precision_score, recall_score,confusion_matrix

np.set_printoptions(suppress=True)
#print时不用科学计数法输出
# IoU的学名叫做jaccard系数
#sklearn的混淆矩阵也是左边是gt，右上角是pred
#要看分类别IoU的话，还是要用17_post_proc/score_several.py或者17_post_proc/score_one.py
# 因为sklearn的混淆矩阵会把None的行、列去掉，就不晓得类别和class IoU的对应关系
GT_img = cv.cvtColor(cv.imread('../21_determine_change_type/binary_change_type/out_test_filter_ps.png', 1), cv.COLOR_BGR2RGB)
PRED_img = cv.cvtColor(cv.imread('../21_determine_change_type/binary_change_type/out_refine50.png', 1), cv.COLOR_BGR2RGB)
# GT_img = cv.cvtColor(cv.imread('../19_draw_fig/gt.png', 1), cv.COLOR_BGR2RGB)
# PRED_img = cv.cvtColor(cv.imread('../19_draw_fig/pred.png', 1), cv.COLOR_BGR2RGB)
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

# label_values_RGB=[(0,0,0),(255,255,255)]
gt = []
pred = []
t=time.time()
GT_precode = util.reverse_one_hot(util.one_hot_it(GT_img, label_values_RGB))
PRED_precode = util.reverse_one_hot(util.one_hot_it(PRED_img, label_values_RGB))
gt = GT_precode.flatten()
pred = PRED_precode.flatten()

Acc = accuracy_score(gt, pred)
F1 = f1_score(gt, pred, average=None)
mean_F1 = np.nanmean(F1)
IoU = jaccard_score(gt, pred, average=None)
IoU_format=np.around(100*IoU,4)
mean_IoU = np.nanmean(IoU)
tt=time.time()-t
print("class F1")
print(F1)
print("class IoU")
print(IoU_format)
print("overall accuracy: %f"%Acc)
print("mean F1: %f" % mean_F1)
print("mean IoU: %f" % mean_IoU)
print("time:%f"%tt)

