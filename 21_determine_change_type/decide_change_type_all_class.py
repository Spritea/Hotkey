import cv2 as cv
import itertools
import numpy as np
import time

# 注意
# 这个是把不变的类别进一步作了区分的
# 总共有7*7=49个类
time_A = time.time()
source_path = "../16_find_compact/SCPA_WC/bs12/large/precode/refine50/0_pred.png"
dest_path = "../16_find_compact/SCPA_WC/bs12/large/precode/refine50/1_pred.png"
source_img = cv.cvtColor(cv.imread(source_path,-1), cv.COLOR_BGR2RGB)
dest_img = cv.cvtColor(cv.imread(dest_path,-1), cv.COLOR_BGR2RGB)
height, width, _ = source_img.shape

land_class_list = [0, 1, 2, 3, 4, 5, 6]
color_list = list(itertools.product([0, 100, 150, 200], repeat=3))
class_list = list(itertools.product(land_class_list, repeat=2))
change_type_count = len(land_class_list) * len(land_class_list)

# 这里没变的rgb是直接用的分割的rgb
# 就是这7个-(0,0,0),(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128),(0,128,128)
candy_color_list = [(0, 0, 0), (255, 250, 250), (248, 248, 255), (211, 211, 211),
                    (255, 99, 71), (255, 250, 240), (139, 69, 19), (250, 240, 230),
                    (128, 0, 0), (0, 206, 209), (255, 215, 0), (205, 92, 92), (255, 228, 196),
                    (255, 218, 185), (255, 222, 173), (175, 238, 238), (0, 128, 0), (255, 248, 220),
                    (47, 79, 79), (255, 250, 205), (255, 245, 238), (240, 255, 240),
                    (245, 255, 250), (240, 255, 255), (128, 128, 0), (240, 248, 255), (230, 230, 250),
                    (255, 240, 245), (255, 228, 225), (255, 255, 240), (105, 105, 105),
                    (112, 128, 144), (0, 0, 128), (190, 190, 190), (245, 245, 245), (100, 149, 237),
                    (65, 105, 225), (0, 191, 255), (135, 206, 250), (70, 130, 180),
                    (128, 0, 128), (255, 228, 181), (250, 235, 215), (95, 158, 160), (0, 250, 154),
                    (255, 255, 0), (255, 239, 213), (255, 235, 205), (0, 128, 128)]
key_list = []
for i in range(len(class_list)):
    change_type = str(class_list[i][0]) + str(class_list[i][1])
    key_list.append(change_type)
candy_change_type_dict = {}
for j in range(change_type_count):
    candy_change_type_dict[key_list[j]] = candy_color_list[j]

source_img_one_channel = source_img[:, :, 0]
dest_img_one_channel = dest_img[:, :, 0]
empty = np.zeros((height, width, 3), np.uint8)

for i in range(height):
    for j in range(width):
        pixel_source = int(source_img_one_channel[i][j])
        pixel_dest = int(dest_img_one_channel[i][j])
        pixel_change_type = str(pixel_source) + str(pixel_dest)
        pixel_color = candy_change_type_dict[pixel_change_type]
        empty[i, j] = pixel_color
time_B = time.time() - time_A
print("time: %f" % time_B)
cv.imwrite("change_type_candy/bs_12_all_class/out_refine50.png", cv.cvtColor(empty, cv.COLOR_RGB2BGR))
