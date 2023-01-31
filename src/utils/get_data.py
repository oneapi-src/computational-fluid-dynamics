# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This script includes util functions
for data load and preprocessing
'''

import os
import random
import numpy as np
import tensorflow as tf

seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def generate_image_label_batch(dataset, batch_size):
    batch_dataset = dataset.batch(batch_size)
    return batch_dataset

def extract_features_batch(serialized_batch):
    shape = (128, 256)
    features = tf.io.parse_example(serialized_batch,
                                   features={'boundary': tf.io.FixedLenFeature([], tf.string),
                                             'sflow': tf.io.FixedLenFeature([], tf.string)})
    boundary = tf.io.decode_raw(features['boundary'], tf.uint8)
    sflow = tf.io.decode_raw(features['sflow'], tf.float32)
    boundary = tf.reshape(boundary, [shape[0], shape[1], 1])
    sflow = tf.reshape(sflow, [shape[0], shape[1], 2])
    boundary = tf.cast(boundary, tf.float32)
    sflow = tf.cast(sflow, tf.float32)

    return boundary, sflow

def flow_input(tfrecord_filename):

    raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)
    parsed_dataset = raw_dataset.map(map_func=extract_features_batch)

    return parsed_dataset

def get_train_test_split(parsed_data, data_size):
    parsed_train = parsed_data.take(data_size)
    parsed_test = parsed_data.skip(data_size)
    return parsed_train, parsed_test

def get_batches(tfrecord_filename, data_size, batch_size):
    parsed_data = flow_input(tfrecord_filename)
    parsed_train, parsed_test = get_train_test_split(parsed_data, data_size)
    batch_data_train = generate_image_label_batch(parsed_train, batch_size)
    batch_data_test = generate_image_label_batch(parsed_test, batch_size)
    return batch_data_train, batch_data_test
