import cv2 as cv
import time
from pathlib import Path
import natsort

IMG_Path = Path("off")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.tiff")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

for j in range(0,len(IMG_Str)):
    img=cv.imread(IMG_Str[j],cv.IMREAD_UNCHANGED)
    t1=time.time()
    dst=cv.xphoto.bm3dDenoising(src=img,h=25,templateWindowSize=8,searchWindowSize=39,groupSize=16,slidingStep=3)
    t2=time.time()-t1
    print(t2)
    first_name = str(Path(IMG_Str[j]).stem)
    cv.imwrite("filter/"+first_name+".tiff",dst)