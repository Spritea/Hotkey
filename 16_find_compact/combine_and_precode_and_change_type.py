import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
from PIL import Image
from tqdm import tqdm
import itertools
import util


# 以下是combine_several_SCPA.py直接复制的
# 包括边边也拼起来
def combine_one(imgs_list, img_path, imgwidth, imgheight):
    im = Image.fromarray(imgs_list[0])
    width, height = im.size
    row_res = imgheight % height
    col_res = imgwidth % width
    img_row = int(imgheight / height) if row_res == 0 else int(imgheight / height) + 1
    # every row in big image contains img_row images
    img_col = int(imgwidth / width) if col_res == 0 else int(imgwidth / width) + 1
    blank = Image.new("RGB", (imgwidth, imgheight))
    for k in range(img_row):
        for j in range(img_col):
            p = Image.fromarray(imgs_list[j + k * img_col])
            if j + 1 == img_col and k + 1 < img_row and col_res > 0:
                box = (width - col_res, 0, width, height)
                p = p.crop(box)
            elif j + 1 < img_col and k + 1 == img_row and row_res > 0:
                box = (0, height - row_res, width, height)
                p = p.crop(box)
            elif j + 1 == img_col and k + 1 == img_row and col_res > 0 and row_res > 0:
                box = (width - col_res, height - row_res, width, height)
                p = p.crop(box)
            blank.paste(p, (width * j, height * k))
    if Path(out_path_prefix).is_dir():
        pass
    else:
        print("Out path is empty!!")
        exit(0)
    out_path = out_path_prefix + "\\" + img_path
    blank.save(out_path)


one_large_contain_small = 49
# 这个是一张大图包括多少张小图

model_folder_name='can'
IMG_Path = Path("SCPA_WC/bs12/small/"+model_folder_name)
refer_large_img = cv.imread("SCPA_WC/test_gt/color/2002-test.png")
height, width, _ = refer_large_img.shape
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.png")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))
large_number = int(len(IMG_Str) / one_large_contain_small)
# 这个是一共能拼成几张大图

large_path='SCPA_WC/bs12/large/color/'+model_folder_name+'/'
id_start = 0
for k in tqdm(range(large_number)):
    pic_small = []
    id_stop = id_start + one_large_contain_small
    for j in range(id_start, id_stop):
        pic_small.append(cv.cvtColor(cv.imread(IMG_Str[j], cv.IMREAD_COLOR), cv.COLOR_BGR2RGB))
    id_start += one_large_contain_small
    out_path_prefix = large_path
    out_name = str(k) + '_pred.png'
    combine_one(pic_small, out_name, width, height)


# 以下是18_preprocee里precode.py直接复制的
def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image


GT_Path = Path(large_path)
GT_File = natsort.natsorted(list(GT_Path.glob("*.png")), alg=natsort.PATH)
GT_Str = []
for i in GT_File:
    GT_Str.append(str(i))

out_prefix = "SCPA_WC/bs12/large/precode/"+model_folder_name+'/'
label_values_RGB_SCPA_WC = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                            [0, 128, 128]]
for k in tqdm(range(len(GT_Str))):
    gt = load_image(GT_Str[k])
    out = util.reverse_one_hot(util.one_hot_it(gt, label_values_RGB_SCPA_WC))
    out_str = out_prefix + Path(GT_Str[k]).name
    cv.imwrite(out_str, out)
    # print("kk")

# 以下是21_determine_change_type/decide_change_type.py直接复制的
source_path = out_prefix+'0_pred.png'
dest_path = out_prefix+'1_pred.png'
source_img = cv.cvtColor(cv.imread(source_path, ), cv.COLOR_BGR2RGB)
dest_img = cv.cvtColor(cv.imread(dest_path, -1), cv.COLOR_BGR2RGB)
height, width, _ = source_img.shape

land_class_list = [0, 1, 2, 3, 4, 5, 6]
color_list = list(itertools.product([0, 100, 150, 200], repeat=3))
class_list = list(itertools.product(land_class_list, repeat=2))
change_type_count = len(land_class_list) * (len(land_class_list) - 1) + 1
class_list_reduce = class_list.copy()
for k in range(len(class_list)):
    if class_list[k][0] == class_list[k][1] and class_list[k][0] > 0:
        class_list_reduce.remove(class_list[k])
color_list_reduce = color_list[0:change_type_count]
candy_color_list = [(0, 0, 0), (255, 250, 250), (248, 248, 255), (211, 211, 211),
                    (255, 99, 71), (255, 250, 240), (139, 69, 19), (250, 240, 230),
                    (0, 206, 209), (255, 215, 0), (205, 92, 92), (255, 228, 196),
                    (255, 218, 185), (255, 222, 173), (175, 238, 238), (255, 248, 220),
                    (47, 79, 79), (255, 250, 205), (255, 245, 238), (240, 255, 240),
                    (245, 255, 250), (240, 255, 255), (240, 248, 255), (230, 230, 250),
                    (255, 240, 245), (255, 228, 225), (255, 255, 240), (105, 105, 105),
                    (112, 128, 144), (190, 190, 190), (245, 245, 245), (100, 149, 237),
                    (65, 105, 225), (0, 191, 255), (135, 206, 250), (70, 130, 180),
                    (255, 228, 181), (250, 235, 215), (95, 158, 160), (0, 250, 154),
                    (255, 255, 0), (255, 239, 213), (255, 235, 205)]
key_list = []
for i in range(len(class_list_reduce)):
    change_type = str(class_list_reduce[i][0]) + str(class_list_reduce[i][1])
    key_list.append(change_type)
change_type_color_dict = {}
candy_change_type_dict = {}
for j in range(change_type_count):
    change_type_color_dict[key_list[j]] = color_list_reduce[j]
    candy_change_type_dict[key_list[j]] = candy_color_list[j]

source_img_one_channel = source_img[:, :, 0]
dest_img_one_channel = dest_img[:, :, 0]
empty = np.zeros((height, width, 3), np.uint8)

for i in range(height):
    for j in range(width):
        pixel_source = int(source_img_one_channel[i][j])
        pixel_dest = int(dest_img_one_channel[i][j])
        if pixel_source == pixel_dest:
            empty[i, j] = (0, 0, 0)
        else:
            pixel_change_type = str(pixel_source) + str(pixel_dest)
            # pixel_color = change_type_color_dict[pixel_change_type]
            pixel_color = candy_change_type_dict[pixel_change_type]
            empty[i, j] = pixel_color

out_change_type_path= '../21_determine_change_type/change_type_candy/bs_12/out_'+model_folder_name+'.png'
cv.imwrite(out_change_type_path, cv.cvtColor(empty, cv.COLOR_RGB2BGR))
