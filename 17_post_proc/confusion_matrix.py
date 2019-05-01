import numpy as np

hist=np.load("confusion_matrix_np/hist_scnn.npy")
hist2=np.load("confusion_matrix_np/hist_our.npy")
diag=np.diag(hist)
tp_line=diag[1]+diag[2]+diag[3]+diag[4]
gt=hist.sum(axis=1)
gt_line=gt[1]+gt[2]+gt[3]+gt[4]
acc_in_scnn_code=tp_line/gt_line
tp_all=diag.sum()
print("special acc:%f" %acc_in_scnn_code)
print("kkk")