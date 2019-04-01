import cv2 as cv
import numpy as np
from pathlib import Path
import natsort


def deal_one(img_path):
    pic = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    pic = cv.cvtColor(pic, cv.COLOR_BGR2RGB)
    r = pic[:, :, 0]
    r_mean = np.mean(r)
    g_mean = np.mean(pic[:, :, 1])
    b_mean = np.mean(pic[:, :, 2])
    r_std=np.std(pic[:, :, 0])
    g_std = np.std(pic[:, :, 1])
    b_std = np.std(pic[:, :, 2])
    mean_list=np.array([r_mean,g_mean,b_mean])
    std_list=np.array([r_std,g_std,b_std])
    return mean_list,std_list


IMG_Path = Path("E:\code\hotkey\\05_slice_with_surround\Vahingen\\train_8_val_8_edit\small_train")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))
mean_sum=np.zeros(3)
std_sum=np.zeros(3)
for k in range(len(IMG_Str)):
    mean_list,std_list=deal_one(IMG_Str[k])
    mean_sum+=mean_list
    std_sum+=std_list
mean_all=mean_sum/len(IMG_Str)
std_all=std_sum/len(IMG_Str)
mean_all_norm=mean_all/255
std_all_norm=std_all/255
print(mean_all)
print(std_all)
print(mean_all_norm)
print(std_all_norm)
# print(mean_sum)
# print(std_sum)