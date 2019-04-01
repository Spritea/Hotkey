from pathlib import Path
import natsort
import shutil


IMG_Path = Path("E:\code\hotkey\\18_preprocess\Postdam\\train_26_val_12\precode_val_label")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

for j in range(len(IMG_Str)):
    str=Path(IMG_Str[j])
    name=str.name
    parts=name.split("_")
    parts[4]='IRRG'
    icon='_'
    new_name=icon.join(parts)
    new_full_name="E:\code\hotkey\\18_preprocess\Postdam\\train_26_val_12\precode_val_label"+'\\'+new_name
    shutil.move(IMG_Str[j],new_full_name)