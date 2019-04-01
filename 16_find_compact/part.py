import shutil
from pathlib import Path
import natsort
import random
import numpy as np

center=np.load("center.npy")
# head=int(center[0])
head=1149
width = 10
height = 10
mat = []
a = head
for j in range(height):
    r = [x for x in range(a + j * 39, a + j * 39 + width)]
    mat.extend(r)

# print("kk")
for k in range(len(mat)):
    elt=mat[k]
    core="SAR01-"+"%05d.png"%elt
    str_full="all_small_edit/"+core
    dest_full="rlt/p3/center/"+core
    shutil.move(str_full,dest_full)