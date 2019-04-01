import cv2 as cv

def image_his_old(image):
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])

    return hist
    print("kk")
    # plt.show()
def image_his(image):
    hist=cv.calcHist([image],[0],None,[256],[0,256])
    return hist

src = cv.imread('center_gt_edit_final.png')
b=src[:,:,0]
g=src[:,:,1]
r=src[:,:,2]
print((b==g).all())
print((g==r).all())

h_b=image_his(b)
h_g=image_his(g)
h_r=image_his(r)
print("pp")
# bg=h_b-h_g
br=h_b-h_r
print("kk")