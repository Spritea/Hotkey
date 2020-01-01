import cv2 as cv
from pathlib import Path
import natsort
import numpy as np
from tqdm import tqdm
# This is to remove extra classes in the label image.

def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image
def cut_class(label):
    height, width, channel = label.shape
    for i in range(height):
        for j in range(width):
            pixel = label[i, j].tolist()
            if pixel == [0, 150, 200] or pixel == [0, 200, 250]:
                label[i, j] = [0, 0, 200]
            elif pixel == [200, 200, 0]:
                label[i, j] = [250, 200, 0]
            elif pixel == [150, 250, 0] or pixel == [150, 200, 150]:
                label[i, j] = [0, 200, 0]
            elif pixel == [150, 0, 250] or pixel == [150, 150, 250]:
                label[i, j] = [200, 0, 200]
    return label

GT_Path = Path("label_copy")
GT_File = natsort.natsorted(list(GT_Path.glob("*.tif")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

out_prefix="out\\"
for k in tqdm(range(len(GT_Str))):
    label_ori=load_image(GT_Str[k])
    label_cut=cut_class(label_ori)
    label_out = cv.cvtColor(label_cut, cv.COLOR_RGB2BGR)
    out_str=out_prefix+Path(GT_Str[k]).stem+'.png'
    cv.imwrite(out_str,label_out)
    print('kk')

