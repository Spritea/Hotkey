import shutil
from pathlib import Path
import natsort
import random
import numpy as np

IMG_Path = Path("all_small_edit")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))
# print("kk")
Index = []
for j in range(len(IMG_Str)):
    name = Path(IMG_Str[j]).stem
    number = name.split("-")[1]
    Index.append(int(number))
# print("ll")
width = 10
height = 10
Head_1 = []
mat = []
for k in range(len(Index)):
    a = Index[k]
    for j in range(height):
        r = [x for x in range(a + j * 39, a + j * 39 + width)]
        mat.extend(r)
    if set(mat) < set(Index):
        Head_1.append(a)
    mat.clear()
np.save("center",np.array(Head_1))
print("k")
print(len(Head_1))
