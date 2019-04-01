import cv2 as cv
import numpy as np
import time
from pathlib import Path
import natsort

## sift+BOW+SVM
USE_DENSE_SIFT = True

IMG_Path = Path("train")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*/*.bmp")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

sift = cv.xfeatures2d.SIFT_create()
descriptors_unclustered = []
dictonarySize = 1000
BOW = cv.BOWKMeansTrainer(dictonarySize)

time_A = time.time()
no_feature = []
step_size = 10
for p in IMG_Str:
    image = cv.imread(p)
    gray = image[:, :, 0]
    if USE_DENSE_SIFT:
        kp = [cv.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
              for x in range(0, gray.shape[1], step_size)]
        desc = sift.compute(gray, kp)
        BOW.add(desc[1])

    else:
        kp, desc = sift.detectAndCompute(gray, None)
        if desc is None:
            no_feature.append(p)
            continue
        BOW.add(desc)

time_use = time.time() - time_A
print("SIFT time: %f second" % time_use)

IMG_Str = list(set(IMG_Str) - set(no_feature))
neg_num = 0
pos_num = 0
for j in IMG_Str:
    if "neg" in j:
        neg_num += 1
    else:
        pos_num += 1
time_B = time.time()
dictonary = BOW.cluster()
fs = cv.FileStorage("dict_dense.xml", cv.FILE_STORAGE_WRITE)
fs.write(name="cluster_centers", val=dictonary)
fs.release()

time_use = time.time() - time_B
print("BOW kmeans time: %f" % time_use)

sift2 = cv.xfeatures2d.SIFT_create()
bowDiction = cv.BOWImgDescriptorExtractor(sift2, cv.BFMatcher(cv.NORM_L2))
# bowDiction=cv.BOWImgDescriptorExtractor(cv.Feature2D(sift2),cv.BFMatcher(cv.NORM_L2))
bowDiction.setVocabulary(dictonary)


# print(np.shape(dictonary))

def feature_extractor(pth):
    im = cv.imread(pth)
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    if USE_DENSE_SIFT:
        kpp = [cv.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
               for x in range(0, gray.shape[1], step_size)]
        return bowDiction.compute(gray, kpp)
    else:
        return bowDiction.compute(gray, sift.detect(gray))


training_desc = []
feature_norm = []
# query=cv.imread("/home/tf/Downloads/data/001.jpg")
query_path = "01-006.bmp"
query_desc = feature_extractor(query_path)
qq = []
qq.extend(query_desc)

time_C = time.time()
for p in IMG_Str:
    # fe=feature_extractor(p)
    # cv.normalize(feature_extractor(p),feature_norm,1.0,0.0,cv.NORM_MINMAX)
    training_desc.extend(feature_extractor(p))
time_use = time.time() - time_C
print("BOW build time: %f" % time_use)

trainData = np.asarray(training_desc, np.float32)
# trainData=np.reshape(trainData_beta, dictonarySize)
labels_neg = np.repeat(np.array([0]), neg_num)
labels_pos = np.repeat(np.array([1]), pos_num)
labels = np.append(labels_neg, labels_pos)[:, np.newaxis]

time_D = time.time()
svm = cv.ml.SVM_create()
# svm.setKernel(cv.ml.SVM_RBF)
# svm.setType(cv.ml.SVM_C_SVC)
# svm.setC(2.67)
# svm.setGamma(5.383)

svm.trainAuto(trainData, cv.ml.ROW_SAMPLE, labels)

time_use = time.time() - time_D
print("SVM train time: %f" % time_use)
svm.save('svm_data_dense.dat')

qq_final = np.asarray(qq, np.float32)
result = svm.predict(qq_final)[1]
print(result)
