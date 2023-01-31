# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This script includes utility functions
for compiling a tensorflow model
'''

# pylint: disable=E0401 E0611

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

def conv_block(filters, x):
    conv_layer = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    conv_layer = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv_layer)
    return conv_layer

def down_block(x, filters):
    conv1 = conv_block(filters, x)
    downsample_layer = MaxPooling2D(pool_size=(2, 2))(conv1)
    return conv1, downsample_layer

def up_concat_block(x1, x2, filters):
    conv2D_Transpose_Layer = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(x1)
    concat_layer = concatenate([conv2D_Transpose_Layer, x2], axis=3)
    upsample_layer = conv_block(filters, concat_layer)
    return upsample_layer

def make_model_a(IMAGE_SIZE, learning_rate):

    inputs = Input([*IMAGE_SIZE, 1])
    pix = [16, 32, 64, 128, 256, 512]

    conv1, down_block_1 = down_block(inputs, pix[0])
    conv2, down_block_2 = down_block(down_block_1, pix[1])
    conv3, down_block_3 = down_block(down_block_2, pix[2])
    conv4, down_block_4 = down_block(down_block_3, pix[3])
    conv5, down_block_5 = down_block(down_block_4, pix[4])

    bridge = conv_block(pix[5], down_block_5)

    up_block_6 = up_concat_block(bridge, conv5, pix[4])
    up_block_7 = up_concat_block(up_block_6, conv4, pix[3])
    up_block_8 = up_concat_block(up_block_7, conv3, pix[2])
    up_block_9 = up_concat_block(up_block_8, conv2, pix[1])
    up_block_10 = up_concat_block(up_block_9, conv1, pix[0])

    conv10 = Conv2D(2, (1, 1), activation='linear')(up_block_10)

    # construct model
    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MSE_flat)
    return model

def MSE_flat(y_true, y_pred):
    y_true_f = keras.layers.Flatten()(y_true)
    y_pred_f = keras.layers.Flatten()(y_pred)
    return tf.math.reduce_mean(tf.square(y_true_f - y_pred_f))
