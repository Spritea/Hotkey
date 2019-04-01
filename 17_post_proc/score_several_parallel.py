import cv2 as cv
import util
import numpy as np
import time

from pathlib import Path
import natsort

from metrics_my import runningScore
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool


def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image

IMG_Path = Path("Postdam/from-pytorch-train/data14/mv3_1_true_2_res50/big")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

GT_Path = Path("Postdam/from-pytorch-train/data14/val_gt_full")
GT_File = natsort.natsorted(list(GT_Path.glob("*.tif")), alg=natsort.PATH)
GT_Str = []
for j in GT_File:
    GT_Str.append(str(j))
t = time.time()
running_metrics_val = runningScore(6)
label_values = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
# label_values = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0], [0, 0, 0]]

def compute_one(img_path,gt_path):
    out = load_image(img_path)
    gt = load_image(gt_path)
    # val_gt_erode paired with [0,0,0]label value
    # label order: R G B
    # num_classes = len(label_values)
    gt = util.reverse_one_hot(util.one_hot_it(gt, label_values))
    output_image = util.reverse_one_hot(util.one_hot_it(out, label_values))
    running_metrics_val.update(gt, output_image)

pool=ThreadPool(6)
#单参数用 pool.map
pool.starmap(compute_one,zip(IMG_Str,GT_Str))
pool.close()
pool.join()

acc, cls_pre, cls_rec, cls_f1, cls_iu, hist = running_metrics_val.get_scores()
tt = time.time() - t
print("cls f1")
print(cls_f1)
print("cls iu")
print(cls_iu)
# print(hist)
print("mean F1-6 classes： %f" % np.nanmean(cls_f1))
print("mean F1-5 classes: %f" % np.nanmean(cls_f1[0:5]))
print("mIoU-5: %f" % np.nanmean(cls_iu[0:5]))
print("all acc: %f" % acc)
print("time: %f" % tt)
