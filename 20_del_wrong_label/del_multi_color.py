import cv2 as cv
import numpy as np
from tqdm import tqdm
#这个用来去除封闭的黑色条缝

def get_right_point(empty,one_contour_reshape):
    height, width = empty.shape
    point_number=one_contour_reshape.shape[0]
    x_coor_list=[]
    for i in range(point_number):
        x_coor_list.append(one_contour_reshape[i][0])
    id=x_coor_list.index(max(x_coor_list))
    point_right=one_contour_reshape[id]
    y_coor_max_right=[]
    if point_right[0]+1==width:
        for j in range(point_number):
            if one_contour_reshape[j][0]==point_right[0]:
                #看最右边是只有一个像素点还是有个竖线
                y_coor_max_right.append(one_contour_reshape[j][1])
        y_coor_max_right.sort()
        y_min=y_coor_max_right[0]
        if y_min>0:
            fill_color_coor=[point_right[0],y_min-1]
        else:
            y_max=y_coor_max_right[len(y_coor_max_right)-1]
            fill_color_coor=[point_right[0],y_max+1]
    else:
        fill_color_coor = [point_right[0] + 1, point_right[1]]

    return fill_color_coor

def del_one_color(pic,color):
    small_area_count=0
    height,width,channel=pic.shape
    empty=np.zeros((height,width),np.uint8)
    for i in range(height):
        for j in range(width):
            pixel=pic[i][j]
            if (pixel==color).all():
                empty[i][j]=255
    empty_color=np.zeros((height,width,3),np.uint8)
    empty_color[:,:,0]=empty
    empty_color[:,:,1]=empty
    empty_color[:,:,2]=empty

    ret, th = cv.threshold(empty, 0, 255,cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(th, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    print("Original Contour Number: %d" % len(contours))

    for cnt in contours:
        # print(cnt.size)
        a=np.reshape(cnt,[cnt.shape[0],cnt.shape[2]])
        area = abs(cv.contourArea(cnt))
        # print(area)
        if area < 50:
            small_area_count+=1
            fill_color_coor=get_right_point(empty,a)
            fill_color=pic[fill_color_coor[1],fill_color_coor[0]]
            fill_color_tuple=(int(fill_color[0]),int(fill_color[1]),int(fill_color[2]))
            if a.shape[0]==1:
                pic[a[0][1]][a[0][0]]=fill_color
            else:
                # cv.fillPoly(empty_color, [a], (0, 0, 0))
                cv.fillPoly(pic, [a], fill_color_tuple)
    print("small contour: %d" % small_area_count)


pic=cv.imread("../21_determine_change_type/change_type_candy/out_train.png",cv.IMREAD_UNCHANGED)
# color_list_BGR=[(0,0,0),(0,0,128),(0,128,0),(0,128,128),(128,0,0),(128,0,128),(128,128,0)]
color_list_RGB = [(0, 0, 0), (255, 250, 250), (248, 248, 255), (211, 211, 211),
                    (255 ,99 ,71), (255, 250, 240), (139 ,69 ,19), (250, 240, 230),
                    (0, 206, 209), (255,215,0), (205,92,92), (255, 228, 196),
                    (255, 218, 185), (255, 222, 173), (175, 238, 238), (255, 248, 220),
                    (47, 79, 79), (255, 250, 205), (255, 245, 238), (240, 255, 240),
                    (245, 255, 250), (240, 255, 255), (240, 248, 255), (230, 230, 250),
                    (255, 240, 245), (255, 228, 225), (255,255,240), (105, 105, 105),
                    (112, 128, 144), (190, 190, 190), (245,245,245), (100, 149, 237),
                    (65, 105, 225), (0, 191, 255), (135, 206, 250), (70, 130, 180),
                    (255,228,181), (250,235,215), (95, 158, 160), (0, 250, 154),
                    (255, 255, 0), (255, 239, 213), (255, 235, 205)]
color_list_BGR=[]
for i in range(len(color_list_RGB)):
    color_list_BGR.append((color_list_RGB[i][2],color_list_RGB[i][1],color_list_RGB[i][0]))

for color in tqdm(color_list_BGR):
    del_one_color(pic,color)
cv.imwrite('out_train_filter.png',pic)

