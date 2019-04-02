import cv2 as cv
from pathlib import Path
import natsort
import util

def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image

GT_Path = Path("E:\code\hotkey\\16_find_compact\Vahingen\Augment\\train_8_val_8_edit\\train_gt_full")
GT_File = natsort.natsorted(list(GT_Path.glob("*.png")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

out_prefix="test/"
label_values = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
for k in range(len(GT_Str)):
    gt=load_image(GT_Str[k])
    out=util.reverse_one_hot(util.one_hot_it(gt,label_values))
    out_str=out_prefix+Path(GT_Str[k]).name
    cv.imwrite(out_str,out)
    # print("kk")