import cv2 as cv
import numpy as np
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

pic=cv.imread("test1.png",cv.IMREAD_UNCHANGED)
height,width,channel=pic.shape
empty=np.zeros((height,width),np.uint8)
# empty[:,:]=255
black=np.array([0,0,0])
for i in range(height):
    for j in range(width):
        pixel=pic[i][j]
        if (pixel==black).all():
            empty[i][j]=255
# cv.imshow('kk',empty)
# cv.waitKey(0)
empty_color=np.zeros((height,width,3),np.uint8)
empty_color[:,:,0]=empty
empty_color[:,:,1]=empty
empty_color[:,:,2]=empty

ret, th = cv.threshold(empty, 0, 255,cv.THRESH_BINARY)
# cv.imshow('kk',th)
# cv.waitKey(0)
contours, hierarchy = cv.findContours(th, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print("Original Contour Number: %d" % len(contours))
refine_contours = []

for cnt in contours:
    # print(cnt.size)
    a=np.reshape(cnt,[cnt.shape[0],cnt.shape[2]])
    # b=np.intp(a)
    area = abs(cv.contourArea(cnt))
    print(area)
    if area < 50:
        fill_color_coor=get_right_point(a)
        fill_color=pic[fill_color_coor[1],fill_color_coor[0]]
        #设某个点的坐标为[i,j],那么从pic上取它应该写成pic[j,i],注意是反的
        fill_color_tuple=(int(fill_color[0]),int(fill_color[1]),int(fill_color[2]))
        if a.shape[0]==1:
            pic[a[0][1]][a[0][0]]=fill_color
        # elif a.shape[0]==2
        else:
            # cv.fillPoly(empty_color, [a], (0, 0, 0))
            cv.fillPoly(pic, [a], fill_color_tuple)

# cv.drawContours(empty_color,contours,-1,(0,0,255),2)

cv.imwrite('out_fcn.png',pic)
# cv.imshow('kk',empty_2)
# cv.waitKey(0)

