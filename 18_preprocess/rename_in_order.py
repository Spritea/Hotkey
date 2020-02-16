from pathlib import Path
import natsort
import shutil

IMG_Path = Path("../05_slice_with_surround/SCPA-WC/source_small/val_gt")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

for k in range(len(IMG_Str)):
    name_ori=IMG_Str[k]
    name='SCPA-WC_name_in_order/val_gt/'+"%04d.png" % k
    shutil.copy(name_ori,name)