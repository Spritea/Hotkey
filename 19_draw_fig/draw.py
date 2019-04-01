import numpy as np
import matplotlib.pyplot as plt

# x=np.array([1,	0.7407,	0.7118,	0.6893,	0.6685	,0.6472,	0.6236	,0.5947,	0.554	,0.482])
# y=np.array([0.0423,	0.6029,	0.6351,	0.6574,	0.6762,	0.6937,	0.7116,	0.7312,	0.755,	0.7886])
#np中的p,r是反的

x1=np.load("pr/fcn/pre.npy")
y1=np.load("pr/fcn/rec.npy")
x2=np.load("pr/deeplabv3/pre.npy")
y2=np.load("pr/deeplabv3/rec.npy")
x3=np.load("pr/refine/pre.npy")
y3=np.load("pr/refine/rec.npy")
x4=np.load("pr/scnn/pre.npy")
y4=np.load("pr/scnn/rec.npy")
x5=np.load("pr/lanenet/pre.npy")
y5=np.load("pr/lanenet/rec.npy")

x1=np.r_[x1,0.00005]
y1=np.r_[y1,0.9963]
x2=np.r_[x2,0.0004]
y2=np.r_[y2,0.9914]
x3=np.r_[x3,0.0039]
y3=np.r_[y3,0.9961]
x4=np.r_[x4,0.0019]
y4=np.r_[y4,0.9393]
x5=np.r_[x5,0.0928]
y5=np.r_[y5,0.9719]


plt.plot(x1, y1, x2, y2,x3, y3, x4, y4,x5, y5)
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.legend(['FCN', 'DeepLabv3','RefineNet','SCNN','LaneNet'],loc=3)
# plt.legend(handles=l1,loc='3')

plt.show()