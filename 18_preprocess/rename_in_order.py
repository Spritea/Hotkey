from pathlib import Path
import natsort
import shutil

IMG_Path = Path("../05_slice_with_surround/GID/train_33_val_8_4band_small/train")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.tif")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

for k in range(len(IMG_Str)):
    name_ori=IMG_Str[k]
    name='GID_name_in_order/4band/train/'+"%05d.tif" % k
    shutil.copy(name_ori,name)