import cv2

ori=cv2.imread()
#(image_width,image_height)
#用最近邻才不会出现标签杂值问题，用线性插值会有杂值问题
#之前的vaihingen和postdam都是用的线性插值，然后额外校正的标签
pic=cv2.resize(ori,(512,256),cv2.INTER_NEAREST)

