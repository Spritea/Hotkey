import cv2 as cv
import util
import numpy as np
import time

from pathlib import Path
import natsort

from metrics_my import runningScore
from tqdm import tqdm


def load_index_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    index_grey_image=image[:,:,0]
    return index_grey_image

def load_image(path):
    # image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    image =cv.imread(path, 1)
    return image

IMG_Path = Path("C:\\Users\dell\Desktop\\refine")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.bmp")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

GT_Path = Path("I:\\DVS_dataset\dataset_final\\01_big\\val_labels_grey")
# GT_Path = Path("I:\\DVS_dataset\scnn_result\\vgg_SCNN_merge\merge")
GT_File = natsort.natsorted(list(GT_Path.glob("*.bmp")), alg=natsort.PATH)
GT_Str = []
for j in GT_File:
    GT_Str.append(str(j))
t = time.time()
running_metrics_val = runningScore(5)
label_values = [[0, 0, 0], [100, 100, 100], [150, 150, 150], [200, 200, 200], [250, 250,250]]

def compute_one(img_path,gt_path):
    out = load_image(img_path)
    # gt = load_image(gt_path)
    # 不要用interpolation=cv.INTER_NEAREST,不然结果不一样，估计opencv bug
    gt = cv.resize(load_image(gt_path),(512,256),cv.INTER_NEAREST)
    # val_gt_erode paired with [0,0,0]label value
    # label order: R G B
    # num_classes = len(label_values)

    gt = util.reverse_one_hot(util.one_hot_it(gt, label_values))
    output_image = util.reverse_one_hot(util.one_hot_it(out, label_values))
    running_metrics_val.update(gt, output_image)
    # running_metrics_val.update(gt, out)

for k in tqdm(range(len(IMG_Str))):
    compute_one(IMG_Str[k],GT_Str[k])

acc, cls_pre, cls_rec, cls_f1, cls_iu, hist = running_metrics_val.get_scores()
# np.save("confusion_matrix_np/hist_scnn.npy",hist)
tt = time.time() - t
print("cls f1")
print(cls_f1)
print("cls iu")
print(cls_iu)
# print(hist)
print("mean F1-5 classes： %f" % np.nanmean(cls_f1))
print("mean F1-4 classes: %f" % np.nanmean(cls_f1[1:5]))
print("mIoU-5: %f" % np.nanmean(cls_iu))
print("mIoU-4: %f" % np.nanmean(cls_iu[1:5]))
print("all acc: %f" % acc)
print("time: %f" % tt)
