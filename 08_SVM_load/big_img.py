import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
from PIL import Image

# load SVM model from .dat file
# load BoW dict from dictionarl.xml file

USE_DENSE_SIFT = True

sift = cv.xfeatures2d.SIFT_create()
bowDiction = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2))

fs = cv.FileStorage("E:\code\hotkey\\07_SVM_train\dict_dense.xml", cv.FILE_STORAGE_READ)
dictionary = fs.getNode("cluster_centers")

bowDiction.setVocabulary(dictionary.mat())
fs.release()
# print(np.shape(dictonary))
svm = cv.ml.SVM_create()
# svm.setKernel(cv.ml.SVM_RBF)
# svm.setType(cv.ml.SVM_C_SVC)
# svm.setC(2.67)
# svm.setGamma(5.383)

# svm.load("/home/tf/code/classification/test1/svm_data.dat")
svm = cv.ml.SVM_load("E:\code\hotkey\\07_SVM_train\svm_data_dense.dat")
# print(svm.getGamma())
# print(svm.getVarCount())
# sv = svm.getSupportVectors()

step_size = 10
Out_path = "out"


def feature_extractor(pth):
    im = cv.imread(pth)
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    if USE_DENSE_SIFT:
        kpp = [cv.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
               for x in range(0, gray.shape[1], step_size)]
        return bowDiction.compute(gray, kpp)
    else:
        return bowDiction.compute(gray, sift.detect(gray))


def pred_one(img_path):
    query_desc = feature_extractor(img_path)
    qq = []
    qq.extend(query_desc)
    qq_final = np.asarray(qq, np.float32)
    result = svm.predict(qq_final)[1][0][0]
    return result


def combine_one(imgs_list, img_path, imgwidth, imgheight):
    im = Image.fromarray(imgs_list[0])
    width, height = im.size
    row_res = imgheight % height
    col_res = imgwidth % width
    img_row = int(imgheight / height) if row_res == 0 else int(imgheight / height) + 1
    # every row in big image contains img_row images
    img_col = int(imgwidth / width) if col_res == 0 else int(imgwidth / width) + 1
    blank = Image.new("RGB", (imgheight, imgwidth))
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
    if Path(Out_path).is_dir():
        pass
    else:
        Path(Out_path).mkdir()
    out_path = Out_path + "/" + Path(img_path).stem+".bmp"
    blank.save(out_path)


# slice_one("8.tiff")
IMG_Path = Path("slice_rlt")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.bmp")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

rlt_list = []
small_list=[]
for j in IMG_Str:
    rlt=int(pred_one(j))
    if rlt>0:
        rlt_small=np.full((50,50,3),255,dtype=np.uint8)
    else:
        rlt_small=np.zeros((50,50,3),dtype=np.uint8)
    small_list.append(rlt_small)
    rlt_list.append(rlt)
combine_one(small_list,"8.tiff",1500,1500)
print(rlt_list)
print(len(rlt_list))
rlt_np = np.asarray(rlt_list)
print("pos: %d" % ((rlt_np != 0).sum()))
