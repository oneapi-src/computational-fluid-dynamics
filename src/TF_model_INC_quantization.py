# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This script is for creating the
INT8 quantized version of the FP32 model
using Intel Neural Compressor (INC)
'''

# pylint: disable=E0401 W0612 C0411

import os.path
import random
import pathlib
import logging
from glob import glob as glb
import argparse
import numpy as np
import tensorflow
from utils.get_data import flow_input, get_train_test_split
from neural_compressor.experimental import Quantization, common


def main(FLAGS):
    # Set Seed Values
    seed_value = 0
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tensorflow.random.set_seed(seed_value)
    
    # Set FLAG values
    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)
    logger = logging.getLogger()
    
    modelfile = FLAGS.modelfile
    outputfile = FLAGS.output_path
    
    # Set data size for training the model
    data_size = 2553

    logger.info("Load Data, Train-Test Split and Generate Batches")
    # Record File Definition
    tfrecord_filename = glb('./data/*.tfrecords')
    
    # Batchify and Train Test Split
    parsed_data = flow_input(tfrecord_filename)
    parsed_test, parsed_train = get_train_test_split(parsed_data, data_size)

    print("Loading Model")
    quantizer = Quantization('./src/unet_quant_INC.yaml')
    quantizer.model = modelfile
    quantizer.calib_dataloader = common.DataLoader(parsed_test)
    q_model = quantizer.fit()
    q_model.save(outputfile)
    logger.info("Quantization of model complete")

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

    parser.add_argument('-o',
                        '--output_path',
                        type=str,
                        default="batch_size_4_quant",
                        help="save quantized model to directory")

    FLAGS = parser.parse_args()
    main(FLAGS)
