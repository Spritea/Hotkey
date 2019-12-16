import cv2 as cv
from pathlib import Path
import natsort
import util

def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image

GT_Path = Path("C:\\Users\dell\Desktop\edge\\train\gt")
GT_File = natsort.natsorted(list(GT_Path.glob("*.png")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

out_prefix="C:\\Users\dell\Desktop\edge\\train\gt_precode\\"
label_values = [[0,0,0], [255, 255, 255]]
for k in range(len(GT_Str)):
    gt=load_image(GT_Str[k])
    out=util.reverse_one_hot(util.one_hot_it(gt,label_values))
    out_str=out_prefix+Path(GT_Str[k]).name
    cv.imwrite(out_str,out)
    # print("kk")