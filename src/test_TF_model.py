# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Script for running inference on TF model
'''

# pylint: disable=W0612 E1129

import os.path
import time
import random
import pathlib
from glob import glob as glb
import logging
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from utils.get_data import flow_input, get_train_test_split
from utils.plot_images import plot_images
from tqdm import tqdm

def stack_data(data_batch, counter, batch_size):
    '''Define a read_image function'''
    data_elements = []
    for ctr in range(counter, counter+batch_size):
        data_elements.append(data_batch[ctr])
    data_elements = np.stack(data_elements, axis=0)
    return data_elements

def main(FLAGS):
    # Set Seed Values
    seed_value = 0
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    # Set FLAG values
    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)
    logger = logging.getLogger()
    
    # create lossfile path if doesnt exist
    path = pathlib.Path(FLAGS.lossfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    lossfile = FLAGS.lossfile
    
    batch_size = FLAGS.batch_size
    modelfile = FLAGS.modelfile

    # Set data size for training the model
    data_size = 2553

    logger.info("Load Data, Train-Test Split and Generate Batches")
    # Record File Definition
    tfrecord_filename = glb('./data/*.tfrecords')
    
    # Batchify and Train Test Split
    parsed_data = flow_input(tfrecord_filename)
    parsed_train, parsed_test = get_train_test_split(parsed_data, data_size)
    boundaries = list(zip(*parsed_test))[0]
    labels = list(zip(*parsed_test))[1]
    
    with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session() as sess:
            logger.info("load graph")
            with tf.io.gfile.GFile(modelfile, "rb") as f:  # noqa: F841
                graph_def = tf.compat.v1.GraphDef()
                loaded = graph_def.ParseFromString(f.read())                
                sess.graph.as_default()
                tf.import_graph_def(graph_def, input_map=None, return_elements=None,
                                    name="", op_dict=None, producer_op_list=None)
                l_input = graph.get_tensor_by_name('x:0')  # Input Tensor
                l_output = graph.get_tensor_by_name('Identity:0')  # Output Tensor
                tf.compat.v1.global_variables_initializer()  # initialize_all_variables
                
                times = []
                MSE = []
                for counter in tqdm(range(0, len(boundaries), batch_size)):
                    test_boundaries = stack_data(boundaries, counter, batch_size)
                    test_labels = stack_data(labels, counter, batch_size)
                    
                    # Inference Call
                    start_time = time.time()
                    Session_out = sess.run(l_output, feed_dict={l_input: test_boundaries})
                    x = time.time()-start_time
                    times.append(x)
                    
                    if FLAGS.plot_images:
                        plot_images(Session_out, test_boundaries, test_labels, batch_size, counter)
                    MSE_inf = np.square(np.subtract(Session_out.reshape(-1), test_labels.reshape(-1))).mean()
                    MSE.append((counter, MSE_inf))
                MSE_df = pd.DataFrame(MSE, columns=['batch_num', 'MSE'])
                MSE_df.to_csv(lossfile, index=False)
                logger.info("Time taken for inference : %f", sum(times)/len(times))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="logfile.log",
                        help="log file to output benchmarking results to")
    
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default="4",
                        help="batch size for running training/inference workloads")

    parser.add_argument('-m',
                        '--modelfile',
                        type=str,
                        default="modelfile.pb",
                        help="modelfile to store the results to")

    parser.add_argument('-p',
                        '--plot_images',
                        default=False,
                        action="store_true",
                        help="output comparative images")

    parser.add_argument('-lossf',
                        '--lossfile',
                        type=str,
                        default="lossfile.csv",
                        help="loss file to store the loss trend to")

    FLAGS = parser.parse_args()
    main(FLAGS)
