import shutil
from pathlib import Path
import natsort

IMG_Path = Path("E:\code\hotkey\\17_post_proc\Vaihingen\\from-train\\train8-edit\\03-02continue-rms-lr-e-4")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*pred.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

for j in range(0,len(IMG_Str)):
    shutil.move(IMG_Str[j],"E:\code\hotkey\\17_post_proc\Vaihingen\\from-train\\train8-edit\\03-02continue-rms-lr-e-4\pred\\"+Path(IMG_Str[j]).name)