from pathlib import Path
import natsort
import shutil

IMG_Path = Path("Postdam/train_18_val_20/precode_val_label")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

connector='_'
for k in range(len(IMG_Str)):
    name_ori=IMG_Str[k]
    name=name_ori.split(connector)
    name[4]='IRRG'
    name=connector.join(name)
    shutil.move(name_ori,name)