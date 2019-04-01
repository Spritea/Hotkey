import cv2 as cv
import util
import time

out=cv.imread("center_gt_0000_center_remove_small.tif")
gt=cv.imread("center_gt_0001_bp.tif")
# out=cv.imread("p1/center_remove_small.png")
# gt=cv.imread("p1/center_gt.png")
label_values = [[0, 0, 0], [255, 255, 255]]
num_classes = len(label_values)
ta=time.time()
gt = util.reverse_one_hot(util.one_hot_it(gt, label_values))
# out = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
output_image = util.reverse_one_hot(util.one_hot_it(out, label_values))
# out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
accuracy, class_accuracies, prec, rec, f1, iou ,kp= util.evaluate_segmentation(pred=output_image, label=gt,num_classes=num_classes)
tt=time.time()-ta
print("accuracy: %f"%accuracy)
print("kappa: %f"%kp)
print("precision: %f" % prec)
print("recall: %f" % rec)
print("F1: %f " % f1)
print("mIoU: %f " % iou)
print("time: %.2f"%tt)
# print("specimen: SAR01-00673")