# 869 pixels = 490 km
# 1 sq. pixel = 0.4 sq. km
from collections import Counter

import cv2
import numpy as np
from focal_loss import BinaryFocalLoss
from tensorflow.keras.models import load_model
import tensorflow as tf
from keras import backend as K


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = tf.compat.v1.layers.flatten(y_true)
    y_pred_f = tf.compat.v1.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def IOU_coef(y_true, y_pred):
    T = K.flatten(y_true)
    P = K.flatten(y_pred)
    inter = K.sum(T*P)
    IoU = (inter+1.0)/(K.sum(T)+K.sum(P)-inter+1.0)
    return IoU

dependencies = {
    'dice_coef': dice_coef,
    'IOU_coef': IOU_coef,
    'BinaryFocalLoss': BinaryFocalLoss(gamma=2)
}

i1 = "./data_final/data/png_full/mask/d1.png"
i2 = "./data_final/data/png_full/mask/d1.png"
img = "./data_final/data/png_full/img/WTR00001_K5_NIA0184.png"

cv2.imshow("demo", cv2.resize(cv2.imread(img, 0), (500, 500)))

m1 = load_model('binary_CE.h5', custom_objects=dependencies, compile=False)
m2 = load_model('FL.h5', custom_objects=dependencies, compile=False)
m3 = load_model('iou.h5', custom_objects=dependencies, compile=False)
m4 = load_model('dice.h5', custom_objects=dependencies, compile=False)

a = cv2.resize(m1.predict((cv2.resize(cv2.imread(img, 0), (128, 128)) / 255).reshape(-1, 128, 128, 1))[0], (500, 500))
a = np.asarray(((a - a.min()) * (1/(a.max() - a.min()) * 255)).astype('uint8'))

b = cv2.resize(m2.predict((cv2.resize(cv2.imread(img, 0), (128, 128)) / 255).reshape(-1, 128, 128, 1))[0], (500, 500))
b = np.asarray(((b - b.min()) * (1/(b.max() - b.min()) * 255)).astype('uint8'))

# c = cv2.resize(m3.predict((cv2.resize(cv2.imread(img, 0), (128, 128)) / 255).reshape(-1, 128, 128, 1))[0], (500, 500))
# c = np.asarray(((c - c.min()) * (1/(c.max() - c.min()) * 255)).astype('uint8'))
#
# d = cv2.resize(m4.predict((cv2.resize(cv2.imread(img, 0), (128, 128)) / 255).reshape(-1, 128, 128, 1))[0], (500, 500))
# d = np.asarray(((d - d.min()) * (1/(d.max() - d.min()) * 255)).astype('uint8'))



ret1,th1 = cv2.threshold(a,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("test1", th1)

ret2,th2 = cv2.threshold(b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("test2", th2)

# ret3,th3 = cv2.threshold(c,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow("test3", c)
#
# ret4,th4 = cv2.threshold(d,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow("test4", d)

diff = th1-th2

cv2.imshow("diff", diff)
fl_area = np.count_nonzero(diff == 1)

print("FLOOD AFFECTED AREA =", fl_area*0.4, "sq. km")

cv2.waitKey(0)