import cv2 as cv
import numpy as np
import util
from metrics_my import runningScore
import time

np.set_printoptions(suppress=True)
#print时不用科学计数法输出

def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image
running_metrics_val = runningScore(2)
# label_values = [(0, 0, 0), (255, 250, 250), (248, 248, 255), (211, 211, 211),
#                     (255 ,99 ,71), (255, 250, 240), (139 ,69 ,19), (250, 240, 230),
#                     (0, 206, 209), (255,215,0), (205,92,92), (255, 228, 196),
#                     (255, 218, 185), (255, 222, 173), (175, 238, 238), (255, 248, 220),
#                     (47, 79, 79), (255, 250, 205), (255, 245, 238), (240, 255, 240),
#                     (245, 255, 250), (240, 255, 255), (240, 248, 255), (230, 230, 250),
#                     (255, 240, 245), (255, 228, 225), (255,255,240), (105, 105, 105),
#                     (112, 128, 144), (190, 190, 190), (245,245,245), (100, 149, 237),
#                     (65, 105, 225), (0, 191, 255), (135, 206, 250), (70, 130, 180),
#                     (255,228,181), (250,235,215), (95, 158, 160), (0, 250, 154),
#                     (255, 255, 0), (255, 239, 213), (255, 235, 205)]

label_values=[(0,0,0),(255,255,255)]
t = time.time()

def compute_one(img_path,gt_path):
    out = load_image(img_path)
    gt = load_image(gt_path)
    # val_gt_erode paired with [0,0,0]label value
    # label order: R G B
    # num_classes = len(label_values)
    gt = util.reverse_one_hot(util.one_hot_it(gt, label_values))
    output_image = util.reverse_one_hot(util.one_hot_it(out, label_values))
    running_metrics_val.update(gt, output_image)
IMG_Path='../21_determine_change_type/binary_change_type/bs_12/out_can.png'
GT_Path='../21_determine_change_type/binary_change_type/out_test_filter_ps.png'
compute_one(IMG_Path,GT_Path)
acc, cls_pre, cls_rec, cls_f1, cls_iu, hist,my_f1 = running_metrics_val.get_scores()
tt = time.time() - t
print("cls f1")
print(cls_f1)
print("cls iu")
print(cls_iu)
# print(hist)
print("mean F1 classes： %f" % np.nanmean(cls_f1))
print("mIoU: %f" % np.nanmean(cls_iu))
print("all acc: %f" % acc)
print("time: %f" % tt)
# print("confusion matrix")
# print(hist)
print("my f1 matrix")
print(my_f1)
print("my f1: %f" % np.nanmean(my_f1))
