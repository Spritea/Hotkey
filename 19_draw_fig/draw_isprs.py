import numpy as np
import matplotlib.pyplot as plt

category = 4
title_list=['Imp surf','Building','Low veg','Tree','Car']
dataset='Vaihingen/'
x1 = np.load(dataset+"fcn8s/rec_" + str(category) + ".npy")
y1 = np.load(dataset+"fcn8s/pre_" + str(category) + ".npy")
x2 = np.load(dataset+"gcn/rec_" + str(category) + ".npy")
y2 = np.load(dataset+"gcn/pre_" + str(category) + ".npy")
x3 = np.load(dataset+"frrnB/rec_" + str(category) + ".npy")
y3 = np.load(dataset+"frrnB/pre_" + str(category) + ".npy")
x4 = np.load(dataset+"deeplabv3/rec_" + str(category) + ".npy")
y4 = np.load(dataset+"deeplabv3/pre_" + str(category) + ".npy")
x5 = np.load(dataset+"deeplabv3_plus/rec_" + str(category) + ".npy")
y5 = np.load(dataset+"deeplabv3_plus/pre_" + str(category) + ".npy")
x6 = np.load(dataset+"mv3_1_true_2_res50/rec_" + str(category) + ".npy")
y6 = np.load(dataset+"mv3_1_true_2_res50/pre_" + str(category) + ".npy")

plt.plot(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6)
# plt.plot(x4,y4)

plt.xlabel('Recall')
plt.ylabel('Precision')

plt.legend(['FCN-8s', 'GCN', 'FRRN-B', 'DeepLabv3', 'DeepLabv3+', 'Ours'], loc=3)
# plt.legend(['FCN-8s'],loc=3)

# plt.legend(handles=l1,loc='3')
plt.xlim((0.5, 1))
plt.ylim((0.5, 1))
plt.title(title_list[category])

plt.show()
