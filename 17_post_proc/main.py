import cv2 as cv
import numpy as np


def remove_small(image):
    height, width, channel = image.shape
    img = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
    gray = img[:, :, 0]
    pp = gray.copy()
    src_gray = gray.copy()
    # _, contours, _ = cv.findContours(src_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(src_gray, contours, -1, (255), thickness=cv.FILLED)
    _, all_contours, _ = cv.findContours(src_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    refine_contours = []
    # rlt = np.zeros((height, width, 3), dtype=np.uint8)

    for cnt in all_contours:
        area = abs(cv.contourArea(cnt))
        # remove tiny area
        if area < 500:
            # if area > 200:
            cv.drawContours(pp, [cnt], 0, 0, thickness=cv.FILLED)
        # imfill bigger area, leave large area containing gaps(roads)
        elif area < 20000:
            cv.drawContours(pp, [cnt], 0, 255, thickness=cv.FILLED)

    return pp
def remove_small_only(image):
    height, width, channel = image.shape
    img = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
    gray = img[:, :, 0]
    pp = gray.copy()
    src_gray = gray.copy()
    # _, contours, _ = cv.findContours(src_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(src_gray, contours, -1, (255), thickness=cv.FILLED)
    _, all_contours, _ = cv.findContours(src_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    refine_contours = []
    # rlt = np.zeros((height, width, 3), dtype=np.uint8)

    for cnt in all_contours:
        area = abs(cv.contourArea(cnt))
        # remove tiny area
        if area < 2000:
            # if area > 200:
            cv.drawContours(pp, [cnt], 0, 0, thickness=cv.FILLED)
        # imfill bigger area, leave large area containing gaps(roads)
        # elif area < 20000:
        #     cv.drawContours(pp, [cnt], 0, 255, thickness=cv.FILLED)

    return pp

def imfill_noise(image):
    height, width, channel = image.shape
    img = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
    gray = img[:, :, 0]
    src_gray = gray.copy()
    _, contours, _ = cv.findContours(src_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(src_gray, contours, -1, (255), thickness=cv.FILLED)
    _, all_contours, _ = cv.findContours(src_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    refine_contours = []
    rlt = np.zeros((height, width, 3), dtype=np.uint8)
    # for cnt in all_contours:
    #     area = abs(cv.contourArea(cnt))
    #     if area > 500:
    #     # if area > 200:
    #         refine_contours.append(cnt)

    for cnt in all_contours:
        area = abs(cv.contourArea(cnt))
        refine_contours.append(cnt)
    # print(len(refine_contours))
    ## refine_contours = ratio_rect(refine_contours)
    # print("hh")
    # print(len(refine_contours))

    cv.drawContours(rlt, refine_contours, -1, (255, 255, 255), thickness=cv.FILLED)
    return rlt


def morph(image):
    blur = cv.GaussianBlur(image, (3, 3), 0)
    blur = blur[:, :, 0]
    ret, th = cv.threshold(blur, 0, 255, cv.THRESH_OTSU)
    # mb=cv.medianBlur(img,3)
    dilatation_size = 1
    dilatation_type = 0
    element = cv.getStructuringElement(dilatation_type, (2 * dilatation_size + 1, 2 * dilatation_size + 1))
    out = cv.morphologyEx(th, cv.MORPH_CLOSE, element)
    element2 = cv.getStructuringElement(dilatation_type, (5, 5))
    blur = cv.morphologyEx(out, cv.MORPH_OPEN, element)

    # blur=cv.medianBlur(blur,5)

    # blur = cv.dilate(blur, element2)
    # # # cv.imwrite("pic/dlt-mblur-dilate-7-p8.bmp",blur)
    #
    # src_gray = blur
    #
    # # src_gray = img[:, :, 0]
    # src = src_gray.copy()
    return blur


pic = cv.imread("pred/time_series/del.png")
rlt = remove_small_only(pic)
# rlt = imfill_noise(pic)
# rlt_2=morph(rlt)
cv.imwrite("pred/time_series/del_remove_small_only_2000.png", rlt)
