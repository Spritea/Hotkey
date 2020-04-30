import cv2 as cv
from pathlib import Path
import natsort
import util
from tqdm import tqdm

def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image

GT_Path = Path("C://Users/think\Desktop/train_samesize\clean_data/test_gt")
GT_File = natsort.natsorted(list(GT_Path.glob("*.png")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

out_prefix="C:\\Users\\think\Desktop\\train_samesize\clean_data\\test_gt_precode\\"
# label_values_GID_9 = [[0,0,0], [0,0,200],[250,150,150],[250,200,0],[0,200,0],[200,0,200],[200,0,0],[250,0,150],[200,150,150]]
# label_values_GID_16=[[0,0,0],[0,200,0],[150,250,0],[150,200,150],[200,0,200],[150,0,250],[150,150,250],[250,200,0],[200,200,0],[200,0,0],[250,0,150],[200,150,150],[250,150,150],[0,0,200],[0,150,200],[0,200,250]]
# label_values_GID_8 = [[0,0,0], [150,250,0],[0,200,0],[200,0,200],[250,200,0],[200,0,0],[250,150,150],[0,0,200]]
# label_values_RGB_SCPA_WC = [[0,0,0], [128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128]]
label_values_binary=[[0,0,0],[255,255,255]]
for k in tqdm(range(len(GT_Str))):
    gt=load_image(GT_Str[k])
    out=util.reverse_one_hot(util.one_hot_it(gt,label_values_binary))
    out_str=out_prefix+Path(GT_Str[k]).name
    cv.imwrite(out_str,out)
    # print("kk")

