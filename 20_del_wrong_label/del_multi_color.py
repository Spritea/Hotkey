import cv2 as cv
import numpy as np
from tqdm import tqdm
#这个用来去除封闭的黑色条缝

def get_right_point(one_contour_reshape):
    point_number=one_contour_reshape.shape[0]
    x_coor_list=[]
    for i in range(point_number):
        x_coor_list.append(one_contour_reshape[i][0])
    id=x_coor_list.index(max(x_coor_list))
    point_right=one_contour_reshape[id]
    fill_color_coor=[point_right[0]+1,point_right[1]]
    return fill_color_coor

def del_several_color(pic,color):
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
            fill_color_coor=get_right_point(a)
            fill_color=pic[fill_color_coor[1],fill_color_coor[0]]
            fill_color_tuple=(int(fill_color[0]),int(fill_color[1]),int(fill_color[2]))
            if a.shape[0]==1:
                pic[a[0][1]][a[0][0]]=fill_color
            else:
                # cv.fillPoly(empty_color, [a], (0, 0, 0))
                cv.fillPoly(pic, [a], fill_color_tuple)

pic=cv.imread("label2.png",cv.IMREAD_UNCHANGED)
color_list_BGR=[(0,0,0),(0,0,128),(0,128,0),(0,128,128),(128,0,0),(128,0,128),(128,128,0)]
for color in tqdm(color_list_BGR):
    del_several_color(pic,color)
cv.imwrite('out3.png',pic)

