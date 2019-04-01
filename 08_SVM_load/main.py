import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
import time

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


IMG_Path = Path("E:\code\hotkey\\07_SVM_train\\train\\neg_train")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.bmp")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

rlt_list = []
for j in IMG_Str:
    rlt_list.append(pred_one(j))
print(rlt_list)
print(len(rlt_list))
rlt_np = np.asarray(rlt_list)
print("pos: %d" % ((rlt_np != 0).sum()))
