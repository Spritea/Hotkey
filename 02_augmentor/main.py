import cv2 as cv
import numpy as np
from pathlib import Path
import natsort
import random
import Augmentor

IMG_path="off\gt"
p=Augmentor.Pipeline(IMG_path)
# p.ground_truth("test1\gt")
p.rotate90(probability=1)
p.process()
p.rotate180(probability=1)
p.process()
p.rotate270(probability=1)
p.process()
q1=Augmentor.Pipeline(IMG_path)
q1.flip_left_right(probability=1)
q1.process()
q2=Augmentor.Pipeline(IMG_path)
q2.flip_top_bottom(probability=1)
q2.process()
