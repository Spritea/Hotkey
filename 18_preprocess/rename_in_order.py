from pathlib import Path
import natsort
import shutil

IMG_Path = Path("../05_slice_with_surround/GID/7class/train38_val3_small/val_gt")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

for k in range(len(IMG_Str)):
    name_ori=IMG_Str[k]
    name='GID_name_in_order/7class/train38_val3/val_gt/'+"%05d.png" % k
    shutil.copy(name_ori,name)