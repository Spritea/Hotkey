import cv2 as cv
import scipy.misc as m
import numpy as np
from PIL import Image

p1=cv.imread("summer_1024/test.png")
p2=cv.cvtColor(p1,cv.COLOR_BGR2RGB)
# print((img==p1).all())
# print((img==p2).all())
# p2=cv.imread("p1/center_pred.png")
pil = Image.open("summer_1024/test.png")
test=pil.convert('RGB')
test_np=np.asarray(test)
print((p2==test_np).all())
print("kk")
# result=cv.subtract(p1,p2)
# cv.imwrite("test.png",result)