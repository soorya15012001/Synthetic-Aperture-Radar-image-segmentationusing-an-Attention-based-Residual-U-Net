from collections import Counter

import tensorflow as tf
import os
from keras.activations import sigmoid
from tensorflow.keras.utils import plot_model
import numpy as np
import os
import rasterio
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import cv2
from tqdm import tqdm
from focal_loss import BinaryFocalLoss
from sklearn.utils import class_weight


class DataGen(tf.keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=2, image_size=2000):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size

    def __load__(self, id_name):
        id_name = id_name.replace('.tif', '')
        image_path = os.path.join(self.path, "img", str(id_name)) + ".tif"
        mask_path = os.path.join(self.path, "mask", str(id_name) + "_label") + ".tif"

        # with rasterio.open(image_path, 'r') as ds:
        #     mask = ds.read()
        #     mask = mask.reshape(1024, 1024, 1)
        #
        # with rasterio.open(mask_path, 'r') as ds:
        #     image = ds.read()
        #     image = image.reshape(1024, 1024, 1)

        image = cv2.resize(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2GRAY), (128, 128)).reshape(128, 128, 1)
        mask = cv2.resize(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_RGB2GRAY), (128, 128)).reshape(128, 128, 1)

        ## Normalizaing
        image = image / 255.0
        mask = mask / 255.0

        # print(image.shape)
        # print(mask.shape)
        return image, mask

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size

        files_batch = self.ids[index * self.batch_size: (index + 1) * self.batch_size]

        image = []
        mask = []

        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)

        image = np.array(image)
        mask = np.array(mask)

        return image, mask

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))


def bn_act(x, act=True):
    x = BatchNormalization()(x)
    if act == True:
        x = Activation("relu")(x)
    return x


def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(conv)
    return conv


def first(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size, padding, strides)

    sht = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(x)
    sht = bn_act(sht, act=False)
    res = Add()([conv, sht])
    return res


def residual(x, filters, kernal_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernal_size, padding=padding,
                     strides=strides)  # Only this layer changes shape
    res = conv_block(res, filters, kernel_size=kernal_size, padding=padding,
                     strides=1)  # This does not change any shape

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = Add()([shortcut, res])
    return output


def up_concat(x, direct):
    up = UpSampling2D((2, 2))(x)  # upscales rows and columns
    # print(up.shape)
    up = Concatenate()([up, direct])  # upscales channels
    return up



def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = tf.compat.v1.layers.flatten(y_true)
    y_pred_f = tf.compat.v1.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

from keras import backend as K
def IOU_coef(y_true, y_pred):
    T = K.flatten(y_true)
    P = K.flatten(y_pred)
    inter = K.sum(T*P)
    IoU = (inter+1.0)/(K.sum(T)+K.sum(P)-inter+1.0)
    return IoU

def IOU_loss(y_true, y_pred):
    return -IOU_coef(y_true, y_pred)

def down(x, channels):
    x0, x1, x2, x3 = x.shape
    dwconv_3 = SeparableConv2D(filters=channels, kernel_size=3, dilation_rate=2, strides=2, padding="same")(x)
    dwconv_3 = BatchNormalization()(dwconv_3)

    dwconv_7 = SeparableConv2D(filters=channels, kernel_size=3, dilation_rate=4, strides=2, padding="same")(x)
    dwconv_7 = BatchNormalization()(dwconv_7)

    paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
    # print("DWconv_3", dwconv_3.shape)
    # print("DWconv_7", dwconv_7[1.shape)

    if dwconv_3.shape[1] != int(x1 / 2) or dwconv_7.shape[1] != int(x1 / 2):
        if dwconv_3.shape[1] > dwconv_7.shape[1]:
            dwconv_3 = dwconv_3[:, 0:-1, 0:-1, :]
            dwconv_7 = tf.pad(dwconv_7, paddings, "REFLECT")
        elif dwconv_7.shape[1] > dwconv_3.shape[1]:
            dwconv_3 = tf.pad(dwconv_3, paddings, "REFLECT")
            dwconv_7 = dwconv_7[:, 0:-1, 0:-1, :]

    if dwconv_3.shape[1] == int(x1 / 2) and dwconv_7.shape[1] == int(x1 / 2):
        dwconv_3 = tf.pad(dwconv_3, paddings, "REFLECT")
        dwconv_7 = tf.pad(dwconv_7, paddings, "REFLECT")

    f = Concatenate()([dwconv_3, dwconv_7])
    return f


def spatial(x, dilation=1):
    x1, x2, x3 = x.shape[1], x.shape[2], x.shape[1]
    x = down(x, x3)
    x = SeparableConv2D(filters=x3 * 2, kernel_size=2, dilation_rate=dilation)(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    return x


def gate(x, channels, reduction_ratio=16):
    g = Dense(channels // reduction_ratio, use_bias=False)(x)
    g = BatchNormalization()(g)
    g = Activation('relu')(g)
    g = Dense(channels, use_bias=False)(g)
    return g


def channel(x, reduction_ratio=16):
    x1, x2, x3 = x.shape[1], x.shape[3], x.shape[2]
    x = tf.reshape(Flatten()(x), [-1, x1, x2, x3])
    x_avg = tf.reshape(Flatten()(GlobalAvgPool2D()(x)), [-1, 1, x1, 1])
    x_max = tf.reshape(Flatten()(GlobalMaxPool2D()(x)), [-1, 1, x1, 1])
    x = tf.keras.layers.concatenate([x_avg, x_max], axis=1)
    x = Conv2D(1, kernel_size=(2, 1))(x)
    x = Flatten()(x)
    x = gate(x, x1, reduction_ratio)
    x = tf.reshape(Flatten()(x), [-1, x1, 1, 1])
    return x


def attn_mech(x, f):
    att_c = channel(x, reduction_ratio=16)
    att_s = spatial(x)
    w = sigmoid(tf.multiply(att_c, att_s))
    w = Conv2D(f, (1, 1))(w)
    return w


def resunet(input_size=(128, 128, 1)):
    f = 64

    inputs = Input(input_size)
    print("Input:", inputs.shape)

    fir = first(inputs, f)
    print("0:", fir.shape)

    dr1 = residual(fir, f * 2, strides=2)
    print("1:", dr1.shape)

    dr2 = residual(dr1, f * 4, strides=2)
    print("2:", dr2.shape)

    dr3 = residual(dr2, f * 8, strides=2)
    print("3:", dr3.shape)

    dr4 = residual(dr3, f * 16, strides=2)
    m1 = conv_block(dr4, f * 16, strides=1)
    m2 = conv_block(m1, f * 16, strides=1)

    ur1 = up_concat(m2, dr3)
    ur1 = attn_mech(ur1, f * 8)
    print("3:", ur1.shape)
    ur1 = tf.multiply(dr3, ur1)
    ur1 = residual(ur1, f * 16)

    ur2 = up_concat(ur1, dr2)
    ur2 = attn_mech(ur2, f * 4)
    print("2:", ur2.shape)
    ur2 = tf.multiply(dr2, ur2)
    ur2 = residual(ur2, f * 8)

    ur3 = up_concat(ur2, dr1)
    ur3 = attn_mech(ur3, f * 2)
    print("1:", ur3.shape)
    ur3 = tf.multiply(dr1, ur3)
    ur3 = residual(ur3, f * 4)

    ur4 = up_concat(ur3, fir)
    ur4 = attn_mech(ur4, f)
    print("0:", ur4.shape)
    ur4 = tf.multiply(fir, ur4)
    ur4 = residual(ur4, f * 2)

    output = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(ur4)
    print("Output:", output.shape)

    model = Model(inputs, output)
    return model


def create_npy():
    Xtrain = []
    Ytrain = []
    X = "./data_final/data/png_full/img/"
    Y = "./data_final/data/png_full/mask/"
    for i, j in tqdm(zip(os.listdir(X), os.listdir(Y))):
        image = np.expand_dims(cv2.resize(cv2.imread(X + i, 0), (128, 128)) / 255, axis=-1)
        mask = np.expand_dims(cv2.resize(cv2.imread(Y + j, 0), (128, 128)) / 255, axis=-1).round()

        Xtrain.append(image)
        Ytrain.append(mask)
    np.savez_compressed('Train_data', Xtrain=Xtrain, Ytrain=Ytrain)


# create_npy()

def train():
    adam = Adam()
    # plot_model(model, to_file="resnet.png", show_shapes=True, rankdir='TB', show_layer_names=False)

    data = np.load('Train_data.npz')
    Xt = data["Xtrain"]
    Yt = data["Ytrain"]
    print(Xt.shape)
    print(Yt.shape)

    # model0 = resunet()
    # model0.compile(optimizer=adam, loss='binary_crossentropy', metrics=[dice_coef, IOU_coef], sample_weight_mode="temporal")
    # model0.fit(Xt, Yt, batch_size=4, epochs=10, validation_split=0.1)
    # model0.save('binary_CE.h5')

    # model1 = resunet()
    # model1.compile(optimizer=adam, loss=BinaryFocalLoss(gamma=2), metrics=[dice_coef, IOU_coef])
    # model1.fit(Xt, Yt, batch_size=4, epochs=10, validation_split=0.1)
    # model1.save('FL.h5')

    model3 = resunet()
    model3.compile(optimizer=adam, loss=[IOU_loss], metrics=[dice_coef, IOU_coef])
    model3.fit(Xt, Yt, batch_size=4, epochs=10, validation_split=0.1)
    model3.save('iou.h5')

    model2 = resunet()
    model2.compile(optimizer=adam, loss=[dice_coef_loss], metrics=[dice_coef, IOU_coef])
    model2.fit(Xt, Yt, batch_size=4, epochs=10, validation_split=0.1)
    model2.save('dice.h5')




train()