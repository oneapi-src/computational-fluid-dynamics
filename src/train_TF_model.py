# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Script for training TF model
'''

# pylint: disable=W0612 R0903

import os.path
import time
import random
import pathlib
from glob import glob as glb
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow
from utils.get_data import get_batches
from utils.train import make_model_a

def main(FLAGS):
    # Set Seed Values
    seed_value = 0
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tensorflow.random.set_seed(seed_value)
    
    # Set FLAG values
    
    # create logfile path if doesnt exist
    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)
    logger = logging.getLogger()
    # logfile = FLAGS.logfile
    
    # create modelfile path if doesnt exist
    path = pathlib.Path(FLAGS.modelfile)
    path.mkdir(parents=True, exist_ok=True)
    modelfile = FLAGS.modelfile
    
    # create lossfile path if doesnt exist
    path = pathlib.Path(FLAGS.lossfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    lossfile = FLAGS.lossfile
    
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs

    # Set data size for training the model
    data_size = 2560
    learning_rate = 0.0001  # standard learning_rate

    # Define Loss Callback Function
    batch_end_loss = []
    
    class SaveBatchLoss(tensorflow.keras.callbacks.Callback):
        '''
        Class definition for compiling loss
        '''
        def on_train_batch_end(self, batch, logs=None):
            batch_end_loss.append(logs['loss'])

    logger.info("Load Data, Train-Test Split and Generate Batches")
    # Record File Definition
    tfrecord_filename = glb('./data/*.tfrecords')
    
    # Batchify and Train Test Split
    batch_data_train, batch_data_test = get_batches(tfrecord_filename, data_size, batch_size)
    IMAGE_SIZE = [128, 256]
    
    # Call Model Compile
    logger.info("Compiling Model")
    model = make_model_a(IMAGE_SIZE, learning_rate)

    # Train Model
    logger.info("Training Model on batch size %s", str(batch_size))
    start = time.time()
    model.fit(batch_data_train, epochs=epochs, callbacks=SaveBatchLoss())
    end = time.time()
    train_time = end-start
    logger.info("Training time = %s", train_time)
    logger.info("Loss after training = %s", batch_end_loss[-1])

    # Save Model
    logger.info("Saving Model")
    model.save(modelfile)

    # Save loss
    df = pd.DataFrame({'loss': batch_end_loss})
    df.to_csv(lossfile)

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

    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default="20",
                        help="epochs for training")

    parser.add_argument('-m',
                        '--modelfile',
                        type=str,
                        default="modelfile",
                        help="modelfile to store the results to")

    parser.add_argument('-lossf',
                        '--lossfile',
                        type=str,
                        default="lossfile.csv",
                        help="loss file to store the loss trend to")

    FLAGS = parser.parse_args()
    main(FLAGS)
