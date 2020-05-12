import cv2 as cv
import util
import numpy as np
import time
from pathlib import Path
import natsort
from tqdm import tqdm

from metrics_my import runningScore


# 注意：这个里面平均f1计算有问题，因为precision或recall等于0的时候，
# 即混淆矩阵中某行或某列为空，f1就成None了
# 而不是像IoU一样，只有当行与列同时为空的时候，IoU才变成None
# 这样会导致算平均f1的时候，总类数变少了，从而结果偏大
# 要改正的话，算f1也应该用2TP/2TP+FP+FN来计算，而不是通过precision和recall来计算
# 这也是sklearn采用的方式，即只有某行或某列为空时，f1给0值，保证总类数不因此减少

def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image

def compute_one(img_path, gt_path):
    out = load_image(img_path)
    gt = load_image(gt_path)
    # val_gt_erode paired with [0,0,0]label value
    # label order: R G B
    # num_classes = len(label_values)
    gt = util.reverse_one_hot(util.one_hot_it(gt, label_values))
    output_image = util.reverse_one_hot(util.one_hot_it(out, label_values))
    running_metrics_val.update(gt, output_image)

IMG_Path = Path("../16_find_compact/GID/07_train38_v2_4band/CAN50/large")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

GT_Path = Path("..\\16_find_compact\GID\\06_train38_v2\\test_gt")
GT_File = natsort.natsorted(list(GT_Path.glob("*.png")), alg=natsort.PATH)
GT_Str = []
for j in GT_File:
    GT_Str.append(str(j))
t = time.time()
running_metrics_val = runningScore(8)
# label_values = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
# label_values = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0], [0, 0, 0]]
# label_scpa_segment_RGB = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128]]
label_GID_7class_RGB=[[0,0,0],[150,250,0],[0,200,0],[200,0,200],[250,200,0],[200,0,0],[250,150,150],[0,0,200]]
label_values=label_GID_7class_RGB

for k in tqdm(range(len(IMG_Str))):
    compute_one(IMG_Str[k], GT_Str[k])

acc, cls_pre, cls_rec, cls_f1, cls_iu, hist, my_f1 = running_metrics_val.get_scores()
tt = time.time() - t
print("cls f1")
print(cls_f1)
print("my f1 matrix")
print(my_f1)
print("cls iu")
print(cls_iu)
# print(hist)
print("mean F1： %f" % np.nanmean(cls_f1))
print("mIoU: %f" % np.nanmean(cls_iu))
print("all acc: %f" % acc)
print("time: %f" % tt)
