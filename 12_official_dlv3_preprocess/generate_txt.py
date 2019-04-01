from pathlib import Path
import natsort

IMG_Path = Path("E:\code\hotkey\\16_find_compact\Vahingen\Augment\\train_23_val_10\\train")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

VAL_Path = Path("E:\code\hotkey\\05_slice_with_surround\Vahingen\\train_23_val_10\small_val_nolap_edit")
VAL_File = natsort.natsorted(list(VAL_Path.glob("*.png")), alg=natsort.PATH)
VAL_Str = []
for i in VAL_File:
    VAL_Str.append(str(i))


with open('Vahingen/data08/train.txt','w') as f:
    for i in range(len(IMG_Str)):
        # S=str(c[i].name)
        # S=str(d[i])
        # f.write('/'+S+'\n')
        S=str(Path(IMG_Str[i]).stem)
        f.write(S+'\n')



