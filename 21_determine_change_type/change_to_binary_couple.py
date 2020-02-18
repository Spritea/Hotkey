import cv2 as cv
from pathlib import Path
import natsort
from tqdm import tqdm

GT_Path = Path("change_type_candy")
GT_File = natsort.natsorted(list(GT_Path.glob("*.png")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

out_prefix="binary_change_type/"
for k in tqdm(range(len(GT_Str))):
    pic=cv.imread(GT_Str[k])
    grey=cv.cvtColor(pic,cv.COLOR_BGR2GRAY)
    ret, th = cv.threshold(grey, 0, 255, cv.THRESH_BINARY)
    out_str=out_prefix+Path(GT_Str[k]).name
    cv.imwrite(out_str,th)